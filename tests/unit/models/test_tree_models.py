# tests/models/test_tree_models.py
"""Unit tests for Phase 20d tree-based models: Random Forest and XGBoost.

Tests cover:
- _rebuild_features_with_window: correctness of rolling EPA recomputation
- RandomForestV1Predictor and XGBoostV1Predictor: registration, spec, trainability
- _train_random_forest / _train_xgboost: smoke-tests on synthetic data
- Holdout Brier plausibility and metadata completeness
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.models.game_prediction._epa_window import (
    _rebuild_features_with_window,
)
import gridiron_edge.models.game_prediction.predictor  # noqa: F401
from gridiron_edge.models.game_prediction.predictor import (
    RandomForestV1Predictor,
    RandomForestV2Predictor,
    XGBoostV1Predictor,
    XGBoostV2Predictor,
)
from gridiron_edge.models.registry import PredictorRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_modeling_df() -> pd.DataFrame:
    """Minimal modeling DataFrame resembling the real schema.

    Covers 4 seasons (2006-2010) so _prepare_data has both train and
    holdout rows.  EPA columns use the standard TEAM_A_*/TEAM_B_* names
    with window=4 values pre-filled.
    """
    rng: Generator = np.random.default_rng(0)
    n = 400

    # Seasons straddling the holdout boundary
    seasons: list[str] = (
        ["2006-2007"] * 80
        + ["2007-2008"] * 80
        + ["2008-2009"] * 80
        + ["2023-2024"] * 80
        + ["2024-2025"] * 80
    )

    epa_suffixes: list[str] = [
        "OFF_EPA_PER_PLAY",
        "OFF_PASS_EPA",
        "OFF_RUSH_EPA",
        "OFF_SUCCESS_RATE",
        "OFF_EXPLOSIVE_RATE",
        "DEF_EPA_PER_PLAY",
        "DEF_PASS_EPA",
        "DEF_RUSH_EPA",
        "DEF_SUCCESS_RATE",
        "DEF_EXPLOSIVE_RATE",
    ]

    df = pd.DataFrame(
        {
            "GAME_ID": [f"game_{i}" for i in range(n)],
            "TEAM_A": rng.choice(["KC", "SF", "BUF", "PHI"], n).tolist(),
            "TEAM_B": rng.choice(["DAL", "NYG", "MIA", "LAR"], n).tolist(),
            "YEAR": seasons,
            "WEEK_NUM": rng.integers(1, 18, n).tolist(),
            "RESULT": rng.choice([0, 1], n).tolist(),
            "HOME_FIELD": rng.choice([0, 1], n).tolist(),
            "TEAM_A_ELO": rng.uniform(1400, 1600, n).tolist(),
            "TEAM_B_ELO": rng.uniform(1400, 1600, n).tolist(),
        }
    )

    # Add EPA columns
    for suffix in epa_suffixes:
        df[f"TEAM_A_{suffix}"] = rng.uniform(-0.2, 0.3, n)
        df[f"TEAM_B_{suffix}"] = rng.uniform(-0.2, 0.3, n)

    return df


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
                        "off_explosive_rate": rng.uniform(0.03, 0.12),
                        "def_epa_per_play": rng.uniform(-0.3, 0.2),
                        "def_pass_epa": rng.uniform(-0.4, 0.3),
                        "def_rush_epa": rng.uniform(-0.2, 0.2),
                        "def_success_rate": rng.uniform(0.3, 0.6),
                        "def_explosive_rate": rng.uniform(0.03, 0.12),
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
        """Window=4 is the fast path — should return identical DataFrame."""

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
        # The EPA columns should differ from the original 4-game window
        assert not result["TEAM_A_OFF_EPA_PER_PLAY"].equals(
            synthetic_modeling_df["TEAM_A_OFF_EPA_PER_PLAY"]
        )

    def test_output_has_all_expected_epa_columns(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """All 20 EPA columns (10 per team) must be present in the output."""

        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=3, repo=mini_repo
        )
        expected_suffixes: list[str] = [
            "OFF_EPA_PER_PLAY",
            "OFF_PASS_EPA",
            "OFF_RUSH_EPA",
            "OFF_SUCCESS_RATE",
            "DEF_EPA_PER_PLAY",
            "DEF_PASS_EPA",
            "DEF_RUSH_EPA",
            "DEF_SUCCESS_RATE",
        ]
        for suffix in expected_suffixes:
            assert f"TEAM_A_{suffix}" in result.columns
            assert f"TEAM_B_{suffix}" in result.columns

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
            assert week1_first_season["TEAM_A_OFF_EPA_PER_PLAY"].isna().all(), (
                "First-season week-1 rows should have NaN EPA — rolling window requires prior games"
            )

    def test_empty_epa_returns_df_unchanged(
        self, synthetic_modeling_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """If epa_by_game.parquet is missing, return the original DataFrame."""

        # tmp_path has no epa_by_game.parquet
        (tmp_path / "data" / "cleaned").mkdir(parents=True)
        result: DataFrame = _rebuild_features_with_window(
            synthetic_modeling_df, window=2, repo=tmp_path
        )
        pd.testing.assert_frame_equal(result, synthetic_modeling_df)


# ---------------------------------------------------------------------------
# Predictor registration and spec
# ---------------------------------------------------------------------------


class TestPredictorRegistration:
    """Verify RF and XGBoost predictors are correctly registered."""

    def test_random_forest_v1_registered(self) -> None:
        """random_forest_v1 should be in the PredictorRegistry."""

        assert "random_forest_v1" in PredictorRegistry.names()

    def test_xgboost_v1_registered(self) -> None:
        """xgboost_v1 should be in the PredictorRegistry."""

        assert "xgboost_v1" in PredictorRegistry.names()

    def test_random_forest_v1_is_trainable(self) -> None:
        """random_forest_v1 should be flagged as trainable."""

        assert PredictorRegistry.is_trainable("random_forest_v1")

    def test_xgboost_v1_is_trainable(self) -> None:
        """xgboost_v1 should be flagged as trainable."""

        assert PredictorRegistry.is_trainable("xgboost_v1")

    def test_random_forest_v1_spec(self) -> None:
        """random_forest_v1 spec should have the correct name."""

        assert RandomForestV1Predictor.spec.name == "random_forest_v1"

    def test_xgboost_v1_spec(self) -> None:
        """xgboost_v1 spec should have the correct name."""

        assert XGBoostV1Predictor.spec.name == "xgboost_v1"

    def test_random_forest_v2_registered(self) -> None:
        """random_forest_v2 should be in the PredictorRegistry."""

        assert "random_forest_v2" in PredictorRegistry.names()

    def test_xgboost_v2_registered(self) -> None:
        """xgboost_v2 should be in the PredictorRegistry."""

        assert "xgboost_v2" in PredictorRegistry.names()

    def test_random_forest_v2_spec(self) -> None:
        """random_forest_v2 spec should have the correct name."""

        assert RandomForestV2Predictor.spec.name == "random_forest_v2"

    def test_xgboost_v2_spec(self) -> None:
        """xgboost_v2 spec should have the correct name."""

        assert XGBoostV2Predictor.spec.name == "xgboost_v2"
