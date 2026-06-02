# tests/unit/models/test_total.py
"""Unit tests for total.py — W2 Phase C total points model."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.models.game_prediction.post_process import (
    projected_scores,
)

# ---------------------------------------------------------------------------
# TestProjectedScores
# ---------------------------------------------------------------------------


class TestProjectedScores:
    """Tests for projected_scores()."""

    def test_pickem(self) -> None:
        """Spread of 0 → both teams score half the total."""
        home, away = projected_scores(0.0, 44.0)
        assert home == pytest.approx(22.0)
        assert away == pytest.approx(22.0)

    def test_home_favored(self) -> None:
        """Negative spread → home scores more than away."""
        home, away = projected_scores(-7.0, 44.0)
        assert home > away
        assert home == pytest.approx(25.5)
        assert away == pytest.approx(18.5)

    def test_away_favored(self) -> None:
        """Positive spread → away scores more than home."""
        home, away = projected_scores(7.0, 44.0)
        assert away > home
        assert home == pytest.approx(18.5)
        assert away == pytest.approx(25.5)

    def test_sum_equals_total(self) -> None:
        """Projected scores always sum to model_total."""
        for spread in [-14.0, -7.0, -3.0, 0.0, 3.0, 7.0, 14.0]:
            for total in [34.0, 44.0, 54.0]:
                home, away = projected_scores(spread, total)
                assert (home + away) == pytest.approx(total)

    def test_difference_equals_margin(self) -> None:
        """Projected home - away = -spread (home margin)."""
        for spread in [-14.0, -7.0, 0.0, 7.0, 14.0]:
            home, away = projected_scores(spread, 44.0)
            # spread is negative when home is favored
            # home margin = home - away = -spread
            assert (home - away) == pytest.approx(-spread)

    def test_symmetry(self) -> None:
        """Flipping spread sign flips scores."""
        home_a, away_a = projected_scores(-7.0, 44.0)
        home_b, away_b = projected_scores(7.0, 44.0)
        assert home_a == pytest.approx(away_b)
        assert away_a == pytest.approx(home_b)


# ---------------------------------------------------------------------------
# TestEnrichWithTotal
# ---------------------------------------------------------------------------


class TestEnrichWithTotal:
    """Tests for enrich_predictions() Phase C columns."""

    def _make_df_with_total(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["G1", "G2", "G3"],
                "home_win_prob": [0.70, 0.50, 0.30],
                "away_win_prob": [0.30, 0.50, 0.70],
                "model_total": [44.5, 41.0, 48.0],
            }
        )

    def _make_df_without_total(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "home_win_prob": [0.70, 0.50],
                "away_win_prob": [0.30, 0.50],
            }
        )

    def test_adds_score_columns_when_total_present(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import (
            enrich_predictions,
        )

        enriched: DataFrame = enrich_predictions(
            self._make_df_with_total(),
            recalibrate=False,
        )
        assert "projected_home_score" in enriched.columns
        assert "projected_away_score" in enriched.columns

    def test_scores_sum_to_total(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import (
            enrich_predictions,
        )

        enriched: DataFrame = enrich_predictions(
            self._make_df_with_total(),
            recalibrate=False,
        )
        for _, row in enriched.iterrows():
            total = row["projected_home_score"] + row["projected_away_score"]
            assert total == pytest.approx(row["model_total"], abs=0.01)

    def test_score_difference_matches_spread(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import (
            enrich_predictions,
        )

        enriched: DataFrame = enrich_predictions(
            self._make_df_with_total(),
            recalibrate=False,
        )
        for _, row in enriched.iterrows():
            margin = row["projected_home_score"] - row["projected_away_score"]
            assert margin == pytest.approx(-row["model_spread"], abs=0.01)

    def test_skips_when_no_total_column(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import (
            enrich_predictions,
        )

        enriched: DataFrame = enrich_predictions(
            self._make_df_without_total(),
            recalibrate=False,
        )
        assert "projected_home_score" not in enriched.columns
        assert "projected_away_score" not in enriched.columns

    def test_phase_ab_columns_still_present(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import (
            enrich_predictions,
        )

        enriched: DataFrame = enrich_predictions(
            self._make_df_with_total(),
            recalibrate=False,
        )
        for col in ["model_spread", "margin_std", "win_prob_lo", "win_prob_hi", "confidence_tier"]:
            assert col in enriched.columns
