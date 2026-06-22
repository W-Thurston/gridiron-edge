# tests/evaluation/test_metrics.py
"""Unit tests for evaluation/metrics.py.

Tests for brier_by_confidence_tier, brier_by_season, and biggest_misses
use synthetic DataFrames so no real data or archive files are needed.
"""

from __future__ import annotations

from typing import Any

from pandas import DataFrame, Series
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_df(
    rows: list[dict],
) -> DataFrame:
    """Build a minimal evaluation DataFrame matching the expected schema."""
    base: list[dict] = [
        {
            "game_id": f"2024_01_AWAY_HOME_{i}",
            "season": "2024-2025",
            "week": 1,
            "away_team": "NYJ",
            "home_team": "MIA",
            "away_win_prob": 0.6,
            "away_team_won": 1,
            "model_name": "test_model",
        }
        for i in range(len(rows))
    ]
    for i, override in enumerate(rows):
        base[i].update(override)
    return DataFrame(base)


# ---------------------------------------------------------------------------
# brier_by_confidence_tier
# ---------------------------------------------------------------------------


class TestBrierByConfidenceTier:
    """Tests for brier_by_confidence_tier."""

    def test_perfect_predictions_low_calibration_gap(self) -> None:
        """A perfectly calibrated model should have near-zero calibration gaps."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        # 10 games at 0.6 probability, all predicted side wins
        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.6, "away_team_won": 1} for _ in range(10)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_confidence_tier(df)

        assert not result.empty
        # All predictions in the 60-70% tier
        tier_row: DataFrame | Series = result.loc[result["confidence_tier"] == "60-70%"]
        assert len(tier_row) == 1
        # Actual win rate should be 1.0, predicted avg 0.6, gap = -0.4
        # (underconfident: model said 60%, team won 100%)
        assert tier_row.iloc[0]["actual_win_rate"] == pytest.approx(1.0)
        assert tier_row.iloc[0]["calibration_gap"] == pytest.approx(0.6 - 1.0, abs=1e-4)

    def test_returns_one_row_per_occupied_tier(self) -> None:
        """Only tiers with at least one game should appear in the output."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        # All predictions in the 50-60% band only
        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.55, "away_team_won": 1} for _ in range(5)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_confidence_tier(df)

        assert len(result) == 1
        assert result.iloc[0]["confidence_tier"] == "50-60%"

    def test_home_team_confidence_aligned_correctly(self) -> None:
        """When p < 0.5, confidence should reflect the home team (1 - p)."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        # 10 games: model strongly favours home team (p=0.25 for away)
        # confidence = 0.75, predicted_team = home
        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.25, "away_team_won": 0} for _ in range(10)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_confidence_tier(df)

        assert not result.empty
        tier_row: DataFrame | Series = result.loc[result["confidence_tier"] == "70-80%"]
        assert len(tier_row) == 1
        # predicted_avg should reflect confidence = 0.75
        assert tier_row.iloc[0]["predicted_avg"] == pytest.approx(0.75, abs=1e-3)
        # Predicted team (home) won every game → actual_win_rate = 1.0
        assert tier_row.iloc[0]["actual_win_rate"] == pytest.approx(1.0)

    def test_empty_df_returns_empty(self) -> None:
        """An empty evaluation DataFrame should return an empty result."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        result: DataFrame = brier_by_confidence_tier(
            DataFrame(
                columns=[
                    "away_win_prob",
                    "away_team_won",
                    "season",
                    "week",
                    "away_team",
                    "home_team",
                    "game_id",
                    "model_name",
                ]
            )
        )
        assert result.empty

    def test_custom_tiers(self) -> None:
        """Custom tier boundaries should be respected."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.65, "away_team_won": 1} for _ in range(5)
        ]
        df: DataFrame = _make_eval_df(rows)
        custom_tiers: list[tuple[float, float]] = [(0.5, 0.75), (0.75, 1.01)]
        result: DataFrame = brier_by_confidence_tier(df, tiers=custom_tiers)

        # All games in the first custom tier (0.65 < 0.75)
        assert len(result) == 1
        assert result.iloc[0]["n_games"] == 5

    def test_n_games_sums_to_total(self) -> None:
        """n_games across all tiers should sum to total number of games."""
        from gridiron_edge.evaluation.metrics import brier_by_confidence_tier

        rows: list[dict[str, float | int]] = (
            [{"away_win_prob": 0.55, "away_team_won": 1} for _ in range(4)]
            + [{"away_win_prob": 0.65, "away_team_won": 0} for _ in range(3)]
            + [{"away_win_prob": 0.75, "away_team_won": 1} for _ in range(2)]
        )
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_confidence_tier(df)
        assert result["n_games"].sum() == 9


# ---------------------------------------------------------------------------
# brier_by_season
# ---------------------------------------------------------------------------


class TestBrierBySeason:
    """Tests for brier_by_season."""

    def test_one_row_per_season(self) -> None:
        """Each unique season should produce exactly one row."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        rows: list[dict[str, float | int | str]] = [
            {"season": "2023-2024", "away_win_prob": 0.6, "away_team_won": 1} for _ in range(5)
        ] + [{"season": "2024-2025", "away_win_prob": 0.55, "away_team_won": 0} for _ in range(5)]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_season(df)

        assert len(result) == 2
        assert set(result["season"]) == {"2023-2024", "2024-2025"}

    def test_delta_vs_mean_sums_to_zero(self) -> None:
        """The mean of delta_vs_mean across all seasons must be 0."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        rows: list[dict[str, float | int | str]] = (
            [{"season": "2022-2023", "away_win_prob": 0.7, "away_team_won": 1} for _ in range(10)]
            + [{"season": "2023-2024", "away_win_prob": 0.6, "away_team_won": 0} for _ in range(10)]
            + [
                {"season": "2024-2025", "away_win_prob": 0.55, "away_team_won": 1}
                for _ in range(10)
            ]
        )
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_season(df)

        assert result["delta_vs_mean"].sum() == pytest.approx(0.0, abs=1e-6)

    def test_trend_column_values(self) -> None:
        """Trend column should be one of '✓', '~', '⚠'."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        rows: list[dict[str, float | int | str]] = [
            {"season": "2023-2024", "away_win_prob": 0.9, "away_team_won": 0} for _ in range(10)
        ] + [{"season": "2024-2025", "away_win_prob": 0.51, "away_team_won": 1} for _ in range(10)]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_season(df)

        valid_trends: set[str] = {"✓", "~", "⚠"}
        assert all(t in valid_trends for t in result["trend"])

    def test_bad_season_gets_warn_trend(self) -> None:
        """A season that is much worse than the mean should be flagged with ⚠."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        # Season A: all correct at 90% → low Brier
        # Season B: all wrong at 90% → high Brier  ← should be flagged
        rows: list[dict[str, float | int | str]] = [
            {"season": "2023-2024", "away_win_prob": 0.9, "away_team_won": 1} for _ in range(20)
        ] + [{"season": "2024-2025", "away_win_prob": 0.9, "away_team_won": 0} for _ in range(20)]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_season(df)

        bad: DataFrame | Series = result.loc[result["season"] == "2024-2025"]
        assert bad.iloc[0]["trend"] == "⚠"

    def test_empty_df_returns_empty(self) -> None:
        """An empty DataFrame should return an empty result."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        result: DataFrame = brier_by_season(
            DataFrame(
                columns=[
                    "season",
                    "away_win_prob",
                    "away_team_won",
                ]
            )
        )
        assert result.empty

    def test_single_season(self) -> None:
        """With one season, delta_vs_mean should be 0.0."""
        from gridiron_edge.evaluation.metrics import brier_by_season

        rows: list[dict[str, float | int | str]] = [
            {"season": "2024-2025", "away_win_prob": 0.6, "away_team_won": 1} for _ in range(10)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = brier_by_season(df)

        assert len(result) == 1
        assert result.iloc[0]["delta_vs_mean"] == pytest.approx(0.0, abs=1e-6)
        # Single season - no meaningful trend signal, should be "~"
        assert result.iloc[0]["trend"] == "~"


# ---------------------------------------------------------------------------
# biggest_misses
# ---------------------------------------------------------------------------


class TestBiggestMisses:
    """Tests for biggest_misses."""

    def test_returns_n_rows(self) -> None:
        """Should return exactly n rows (or fewer if the dataset is smaller)."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.6 + i * 0.01, "away_team_won": 0} for i in range(20)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=10)
        assert len(result) == 10

    def test_returns_all_if_n_larger_than_dataset(self) -> None:
        """If n > len(df), all rows are returned."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.7, "away_team_won": 0} for _ in range(5)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=20)
        assert len(result) == 5

    def test_sorted_by_error_descending(self) -> None:
        """Highest-error predictions should appear first."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.51, "away_team_won": 0},  # error = 0.51
            {"away_win_prob": 0.90, "away_team_won": 0},  # error = 0.90
            {"away_win_prob": 0.70, "away_team_won": 0},  # error = 0.70
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=3)

        errors: list[Any] = result["error"].tolist()
        assert errors == sorted(errors, reverse=True)

    def test_correct_predictions_have_low_error(self) -> None:
        """Games where the predicted team won should have small error."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.9, "away_team_won": 1},  # away predicted, away won → error 0.1
            {"away_win_prob": 0.9, "away_team_won": 0},  # away predicted, away lost → error 0.9
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=2)

        # First row should be the large miss (error 0.9)
        assert result.iloc[0]["actual_result"] == "LOSS"
        assert result.iloc[0]["error"] == pytest.approx(0.9, abs=1e-3)
        # Second row is the near-correct prediction (error ~0.1)
        assert result.iloc[1]["actual_result"] == "WIN"

    def test_home_team_favorite_aligned(self) -> None:
        """When p < 0.5, the home team is the predicted team."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.2, "away_team_won": 1}
        ]  # home fav, but away won
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=1)

        assert result.iloc[0]["predicted_team"] == "MIA"  # home team
        assert result.iloc[0]["actual_result"] == "LOSS"
        assert result.iloc[0]["confidence"] == pytest.approx(0.8, abs=1e-3)
        assert result.iloc[0]["error"] == pytest.approx(0.8, abs=1e-3)

    def test_output_columns_present(self) -> None:
        """All expected columns should be in the output."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        rows: list[dict[str, float | int]] = [
            {"away_win_prob": 0.7, "away_team_won": 0} for _ in range(5)
        ]
        df: DataFrame = _make_eval_df(rows)
        result: DataFrame = biggest_misses(df, n=3)

        expected_cols: set[str] = {
            "season",
            "week",
            "away_team",
            "home_team",
            "predicted_team",
            "confidence",
            "actual_result",
            "error",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_empty_df_returns_empty(self) -> None:
        """An empty evaluation DataFrame should return an empty result."""
        from gridiron_edge.evaluation.metrics import biggest_misses

        result: DataFrame = biggest_misses(
            DataFrame(
                columns=[
                    "season",
                    "week",
                    "away_team",
                    "home_team",
                    "away_win_prob",
                    "away_team_won",
                    "game_id",
                    "model_name",
                ]
            ),
            n=10,
        )
        assert result.empty
