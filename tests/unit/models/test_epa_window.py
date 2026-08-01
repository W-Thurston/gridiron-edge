# tests/unit/models/test_epa_window.py

"""Unit tests for _rebuild_features_with_window (rolling EPA recomputation).

Used by GamesTrainer's hyperparameter search over EPA window sizes.
Tests cover:
- Window=4 fast path (no disk read)
- Different window sizes produce different EPA values
- Output schema preserved
- No lookahead leakage
- Graceful handling of missing epa_by_game.parquet
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.team.epa import (
    HomeAwayEpaFeature,
)
from gridiron_edge.models.game_prediction._epa_window import (
    _rebuild_features_with_window,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_modeling_df() -> pd.DataFrame:
    """Return canonical modeling rows with window-four EPA values."""
    rng: Generator = np.random.default_rng(0)
    row_count = 400

    seasons = (
        ["2006-2007"] * 80
        + ["2007-2008"] * 80
        + ["2008-2009"] * 80
        + ["2023-2024"] * 80
        + ["2024-2025"] * 80
    )

    frame = pd.DataFrame(
        {
            "GAME_ID": [f"game_{index}" for index in range(row_count)],
            "AWAY_TEAM": rng.choice(
                ["KC", "SF", "BUF", "PHI"],
                row_count,
            ).tolist(),
            "HOME_TEAM": rng.choice(
                ["DAL", "NYG", "MIA", "LAR"],
                row_count,
            ).tolist(),
            "YEAR": seasons,
            "WEEK_NUM": rng.integers(
                1,
                19,
                row_count,
            ).tolist(),
            "GAME_DATE": [f"{season[:4]}-09-01" for season in seasons],
            "HOME_WIN": rng.choice(
                [0, 1],
                row_count,
            ).tolist(),
            "ACTUAL_MARGIN": rng.uniform(
                -30.0,
                30.0,
                row_count,
            ).tolist(),
            "ACTUAL_TOTAL": rng.uniform(
                20.0,
                70.0,
                row_count,
            ).tolist(),
            "AWAY_ELO": rng.uniform(
                1400.0,
                1600.0,
                row_count,
            ).tolist(),
            "HOME_ELO": rng.uniform(
                1400.0,
                1600.0,
                row_count,
            ).tolist(),
        }
    )

    feature_values = pd.DataFrame(
        {
            output: rng.uniform(
                -0.2,
                0.3,
                row_count,
            )
            for output in (HomeAwayEpaFeature.spec.produces)
        }
    )

    frame = pd.concat(
        [
            frame,
            feature_values,
        ],
        axis=1,
    )

    assert frame.columns.is_unique
    assert frame["GAME_ID"].is_unique

    return frame


@pytest.fixture()
def synthetic_epa_by_game() -> pd.DataFrame:
    """Minimal epa_by_game DataFrame for window rebuild tests."""
    rng: Generator = np.random.default_rng(1)
    teams: list[str] = ["KC", "SF", "BUF", "PHI", "DAL", "NYG", "MIA", "LAR"]
    rows: list[dict[str, int | str]] = []
    for season in [2006, 2007, 2008, 2023, 2024]:
        for week in range(1, 19):
            for team in teams:
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_A_B",
                        "season": season,
                        "week": week,
                        "team": team,
                        "off_epa_per_play": rng.uniform(-0.2, 0.3),
                        "off_pass_epa": rng.uniform(-0.3, 0.4),
                        "off_rush_epa": rng.uniform(-0.2, 0.2),
                        "off_success_rate": rng.uniform(0.3, 0.6),
                        "off_pass_success_rate": rng.uniform(0.30, 0.60),
                        "off_rush_success_rate": rng.uniform(0.30, 0.50),
                        "off_explosive_rate": rng.uniform(0.03, 0.12),
                        "off_third_down_pct": rng.uniform(0.25, 0.50),
                        "off_redzone_td_pct": rng.uniform(0.40, 0.70),
                        "off_turnover_rate": rng.uniform(0.01, 0.06),
                        "off_sack_rate": rng.uniform(0.04, 0.10),
                        "def_epa_per_play": rng.uniform(-0.3, 0.2),
                        "def_pass_epa": rng.uniform(-0.4, 0.3),
                        "def_rush_epa": rng.uniform(-0.2, 0.2),
                        "def_success_rate": rng.uniform(0.3, 0.6),
                        "def_pass_success_rate": rng.uniform(0.30, 0.60),
                        "def_rush_success_rate": rng.uniform(0.30, 0.50),
                        "def_explosive_rate": rng.uniform(0.03, 0.12),
                        "def_third_down_pct": rng.uniform(0.25, 0.50),
                        "def_redzone_td_pct": rng.uniform(0.40, 0.70),
                        "def_turnover_rate": rng.uniform(0.01, 0.06),
                        "def_sack_rate": rng.uniform(0.04, 0.10),
                        "off_plays": rng.integers(50, 80),
                        "off_yards_per_play": rng.uniform(4.0, 7.0),
                        "off_redzone_attempts": rng.integers(2, 10),
                        "off_int_rate": rng.uniform(0.01, 0.05),
                        "off_penalty_rate": rng.uniform(0.02, 0.08),
                        "off_avg_score_diff": rng.uniform(-10.0, 10.0),
                        "off_close_game_pct": rng.uniform(0.3, 0.8),
                        "def_plays": rng.integers(50, 80),
                        "def_yards_per_play": rng.uniform(4.0, 7.0),
                        "def_redzone_attempts": rng.integers(2, 10),
                        "def_int_rate": rng.uniform(0.01, 0.05),
                        "def_penalty_rate": rng.uniform(0.02, 0.08),
                        "def_avg_score_diff": rng.uniform(-10.0, 10.0),
                        "def_close_game_pct": rng.uniform(0.3, 0.8),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def mini_repo(tmp_path: Path, synthetic_epa_by_game: pd.DataFrame) -> Path:
    """Minimal repository structure with epa_by_game.parquet."""
    cleaned: Path = tmp_path / "data" / "cleaned"
    cleaned.mkdir(parents=True)
    synthetic_epa_by_game.to_parquet(cleaned / "epa_by_game.parquet", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# _rebuild_features_with_window
# ---------------------------------------------------------------------------


class TestRebuildFeaturesWithWindow:
    """Tests for the rolling EPA window rebuild function."""

    def test_window_4_returns_df_unchanged(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """Window=4 is the fast path - should return identical DataFrame."""
        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=4, repo=mini_repo
        )
        pd.testing.assert_frame_equal(result, synthetic_modeling_df)

    def test_different_window_changes_epa_columns(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """Window != 4 should produce different EPA column values."""
        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=2, repo=mini_repo
        )
        assert not result["AWAY_OFF_EPA_PER_PLAY"].equals(
            synthetic_modeling_df["AWAY_OFF_EPA_PER_PLAY"]
        )

    def test_output_has_all_canonical_epa_columns(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        result = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=3,
            repo=mini_repo,
        )

        for column in HomeAwayEpaFeature.spec.produces:
            assert column in result.columns
            assert result.columns.tolist().count(column) == 1

    def test_output_row_count_matches_input(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """Row count should be identical to the input."""
        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=6, repo=mini_repo
        )
        assert len(result) == len(synthetic_modeling_df)

    def test_no_lookahead_leakage(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """First-season week-1 rows should have NaN EPA (no prior games to average)."""
        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=3, repo=mini_repo
        )
        first_season = result["YEAR"].min()
        week1_first_season = result.loc[
            (result["WEEK_NUM"] == 1) & (result["YEAR"] == first_season)
        ]
        if len(week1_first_season) > 0:
            assert week1_first_season["AWAY_OFF_EPA_PER_PLAY"].isna().all(), (
                "First-season week-1 rows should have NaN EPA - rolling window requires prior games"
            )
            assert week1_first_season["HOME_OFF_EPA_PER_PLAY"].isna().all(), (
                "First-season week-1 rows should have NaN EPA - rolling window requires prior games"
            )

    def test_empty_epa_source_replaces_values_with_nulls(
        self,
        synthetic_modeling_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Unavailable EPA must not reuse the persisted window-four values."""
        cleaned: Path = tmp_path / "data" / "cleaned"
        cleaned.mkdir(parents=True)

        pd.DataFrame().to_parquet(
            cleaned / "epa_by_game.parquet",
            index=False,
        )

        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=2,
            repo=tmp_path,
        )

        assert len(result) == len(synthetic_modeling_df)

        for column in HomeAwayEpaFeature.spec.produces:
            assert column in result.columns
            assert result[column].isna().all()

    def test_rebuild_preserves_game_and_target_columns(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        result = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=2,
            repo=mini_repo,
        )

        preserved = [
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "AWAY_TEAM",
            "HOME_TEAM",
            "HOME_WIN",
            "ACTUAL_MARGIN",
            "ACTUAL_TOTAL",
        ]

        pd.testing.assert_frame_equal(
            result.loc[:, preserved],
            synthetic_modeling_df.loc[
                :,
                preserved,
            ],
        )

    def test_rebuild_preserves_one_row_per_game(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        result = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=6,
            repo=mini_repo,
        )

        assert len(result) == len(synthetic_modeling_df)
        assert result["GAME_ID"].is_unique

    def test_rebuild_does_not_mutate_input(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        expected = synthetic_modeling_df.copy(deep=True)

        _rebuild_features_with_window(
            synthetic_modeling_df,
            window=2,
            repo=mini_repo,
        )

        pd.testing.assert_frame_equal(
            synthetic_modeling_df,
            expected,
        )

    def test_rebuilt_frame_excludes_retired_orientation(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        result = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=3,
            repo=mini_repo,
        )

        retired = {
            "TEAM_A",
            "TEAM_B",
            "HOME_FIELD",
            "RESULT",
        }

        assert not (retired & set(result.columns))

    def test_helper_source_excludes_retired_orientation(
        self,
    ) -> None:
        import inspect

        source = inspect.getsource(_rebuild_features_with_window)

        assert "TEAM_A" not in source
        assert "TEAM_B" not in source
        assert "HOME_FIELD" not in source
        assert "RESULT" not in source

    def test_window_4_returns_same_object(
        self,
        synthetic_modeling_df: pd.DataFrame,
        mini_repo: Path,
    ) -> None:
        result = _rebuild_features_with_window(
            synthetic_modeling_df,
            window=4,
            repo=mini_repo,
        )

        assert result is synthetic_modeling_df
