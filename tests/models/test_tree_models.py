# tests/models/test_tree_models.py
"""Unit tests for Phase 20d tree-based models: Random Forest and XGBoost.

Tests cover:
- _rebuild_features_with_window: correctness of rolling EPA recomputation
- RandomForestV1Predictor and XGBoostV1Predictor: registration, spec, trainability
- _train_random_forest / _train_xgboost: smoke-tests on synthetic data
- Holdout Brier plausibility and metadata completeness
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    rng = np.random.default_rng(0)
    n = 400

    # Seasons straddling the holdout boundary
    seasons = (
        ["2006-2007"] * 80
        + ["2007-2008"] * 80
        + ["2008-2009"] * 80
        + ["2023-2024"] * 80
        + ["2024-2025"] * 80
    )

    epa_suffixes = [
        "OFF_EPA_PER_PLAY",
        "OFF_PASS_EPA",
        "OFF_RUSH_EPA",
        "OFF_SUCCESS_RATE",
        "DEF_EPA_PER_PLAY",
        "DEF_PASS_EPA",
        "DEF_RUSH_EPA",
        "DEF_SUCCESS_RATE",
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
    rng = np.random.default_rng(1)
    teams = ["KC", "SF", "BUF", "PHI", "DAL", "NYG", "MIA", "LAR"]
    rows = []
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
                        "def_epa_per_play": rng.uniform(-0.3, 0.2),
                        "def_pass_epa": rng.uniform(-0.4, 0.3),
                        "def_rush_epa": rng.uniform(-0.2, 0.2),
                        "def_success_rate": rng.uniform(0.3, 0.6),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def mini_repo(tmp_path: Path, synthetic_epa_by_game: pd.DataFrame) -> Path:
    """Minimal repository structure with epa_by_game.parquet."""
    cleaned = tmp_path / "data" / "cleaned"
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
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        result = _rebuild_features_with_window(synthetic_modeling_df, window=4, repo=mini_repo)
        pd.testing.assert_frame_equal(result, synthetic_modeling_df)

    def test_different_window_changes_epa_columns(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """Window != 4 should produce different EPA column values."""
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        result = _rebuild_features_with_window(synthetic_modeling_df, window=2, repo=mini_repo)
        # The EPA columns should differ from the original 4-game window
        assert not result["TEAM_A_OFF_EPA_PER_PLAY"].equals(
            synthetic_modeling_df["TEAM_A_OFF_EPA_PER_PLAY"]
        )

    def test_output_has_all_expected_epa_columns(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """All 16 EPA columns (8 per team) must be present in the output."""
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        result = _rebuild_features_with_window(synthetic_modeling_df, window=3, repo=mini_repo)
        expected_suffixes = [
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
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        result = _rebuild_features_with_window(synthetic_modeling_df, window=6, repo=mini_repo)
        assert len(result) == len(synthetic_modeling_df)

    def test_no_lookahead_leakage(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path
    ) -> None:
        """Week-1 rows should have NaN EPA (no prior games to average)."""
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        result = _rebuild_features_with_window(synthetic_modeling_df, window=3, repo=mini_repo)
        week1 = result.loc[result["WEEK_NUM"] == 1]
        if len(week1) > 0:
            # All week-1 rows should have NaN EPA (nothing to roll over)
            assert week1["TEAM_A_OFF_EPA_PER_PLAY"].isna().all(), (
                "Week-1 rows should have NaN EPA — rolling window requires prior games"
            )

    def test_empty_epa_returns_df_unchanged(
        self, synthetic_modeling_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """If epa_by_game.parquet is missing, return the original DataFrame."""
        from gridiron_edge.models.game_prediction.tree import (
            _rebuild_features_with_window,
        )

        # tmp_path has no epa_by_game.parquet
        (tmp_path / "data" / "cleaned").mkdir(parents=True)
        result = _rebuild_features_with_window(synthetic_modeling_df, window=2, repo=tmp_path)
        pd.testing.assert_frame_equal(result, synthetic_modeling_df)


# ---------------------------------------------------------------------------
# Predictor registration and spec
# ---------------------------------------------------------------------------


class TestPredictorRegistration:
    """Verify RF and XGBoost predictors are correctly registered."""

    def test_random_forest_v1_registered(self) -> None:
        """random_forest_v1 should be in the PredictorRegistry."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert "random_forest_v1" in PredictorRegistry.names()

    def test_xgboost_v1_registered(self) -> None:
        """xgboost_v1 should be in the PredictorRegistry."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert "xgboost_v1" in PredictorRegistry.names()

    def test_random_forest_v1_is_trainable(self) -> None:
        """random_forest_v1 should be flagged as trainable."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert PredictorRegistry.is_trainable("random_forest_v1")

    def test_xgboost_v1_is_trainable(self) -> None:
        """xgboost_v1 should be flagged as trainable."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert PredictorRegistry.is_trainable("xgboost_v1")

    def test_random_forest_v1_spec(self) -> None:
        """random_forest_v1 spec should have the correct name."""
        from gridiron_edge.models.game_prediction.predictor import RandomForestV1Predictor

        assert RandomForestV1Predictor.spec.name == "random_forest_v1"

    def test_xgboost_v1_spec(self) -> None:
        """xgboost_v1 spec should have the correct name."""
        from gridiron_edge.models.game_prediction.predictor import XGBoostV1Predictor

        assert XGBoostV1Predictor.spec.name == "xgboost_v1"

    def test_random_forest_v2_registered(self) -> None:
        """random_forest_v2 should be in the PredictorRegistry."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert "random_forest_v2" in PredictorRegistry.names()

    def test_xgboost_v2_registered(self) -> None:
        """xgboost_v2 should be in the PredictorRegistry."""
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        assert "xgboost_v2" in PredictorRegistry.names()

    def test_random_forest_v2_spec(self) -> None:
        """random_forest_v2 spec should have the correct name."""
        from gridiron_edge.models.game_prediction.predictor import RandomForestV2Predictor

        assert RandomForestV2Predictor.spec.name == "random_forest_v2"

    def test_xgboost_v2_spec(self) -> None:
        """xgboost_v2 spec should have the correct name."""
        from gridiron_edge.models.game_prediction.predictor import XGBoostV2Predictor

        assert XGBoostV2Predictor.spec.name == "xgboost_v2"


# ---------------------------------------------------------------------------
# Training smoke tests (tiny n_iter / n_estimators for speed)
# ---------------------------------------------------------------------------


class TestRandomForestV1Training:
    """Smoke tests for _train_random_forest on synthetic data."""

    def test_train_returns_metadata(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training should return a ModelMetadata with a finite holdout Brier."""
        from gridiron_edge.models.game_prediction import predictor as pred_mod

        # Speed up: restrict window options to just 4 so no rebuild is needed
        monkeypatch.setattr(pred_mod, "_EPA_WINDOW_OPTIONS", [4])

        metadata = pred_mod._train_random_forest(
            synthetic_modeling_df,
            model_version="random_forest_v1",
            feature_fn=pred_mod._make_combined_features,
            feature_names=pred_mod._COMBINED_FEATURES,
            repo=mini_repo,
        )

        assert metadata.model_version == "random_forest_v1"
        assert 0.0 < metadata.holdout_brier < 1.0
        assert "n_estimators" in metadata.parameters
        assert "epa_window" in metadata.parameters
        assert "calibration_method" in metadata.parameters
        assert metadata.parameters["calibration_method"] == "isotonic"
        assert "top10_feature_importances" in metadata.parameters
        assert len(metadata.parameters["top10_feature_importances"]) <= 10

    def test_train_saves_artifact(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training should save a loadable artifact to the ArtifactStore."""
        from gridiron_edge.models.artifact import ArtifactStore
        from gridiron_edge.models.game_prediction import predictor as pred_mod

        monkeypatch.setattr(pred_mod, "_EPA_WINDOW_OPTIONS", [4])

        pred_mod._train_random_forest(
            synthetic_modeling_df,
            model_version="random_forest_v1",
            feature_fn=pred_mod._make_combined_features,
            feature_names=pred_mod._COMBINED_FEATURES,
            repo=mini_repo,
        )

        store = ArtifactStore(mini_repo)
        assert store.is_trained("random_forest_v1")
        pipeline = store.load("random_forest_v1")
        assert pipeline is not None


class TestXGBoostV1Training:
    """Smoke tests for _train_xgboost on synthetic data."""

    def test_train_returns_metadata(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training should return a ModelMetadata with a finite holdout Brier."""
        from gridiron_edge.models.game_prediction import predictor as pred_mod

        monkeypatch.setattr(pred_mod, "_EPA_WINDOW_OPTIONS", [4])

        metadata = pred_mod._train_xgboost(
            synthetic_modeling_df,
            model_version="xgboost_v1",
            feature_fn=pred_mod._make_combined_features,
            feature_names=pred_mod._COMBINED_FEATURES,
            repo=mini_repo,
        )

        assert metadata.model_version == "xgboost_v1"
        assert 0.0 < metadata.holdout_brier < 1.0
        assert "n_estimators" in metadata.parameters
        assert "learning_rate" in metadata.parameters
        assert "epa_window" in metadata.parameters
        assert "holdout_ece" in metadata.parameters
        assert "calibration_applied" in metadata.parameters
        assert "top10_feature_importances" in metadata.parameters
        assert len(metadata.parameters["top10_feature_importances"]) <= 10

    def test_train_saves_artifact(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training should save a loadable artifact to the ArtifactStore."""
        from gridiron_edge.models.artifact import ArtifactStore
        from gridiron_edge.models.game_prediction import predictor as pred_mod

        monkeypatch.setattr(pred_mod, "_EPA_WINDOW_OPTIONS", [4])

        pred_mod._train_xgboost(
            synthetic_modeling_df,
            model_version="xgboost_v1",
            feature_fn=pred_mod._make_combined_features,
            feature_names=pred_mod._COMBINED_FEATURES,
            repo=mini_repo,
        )

        store = ArtifactStore(mini_repo)
        assert store.is_trained("xgboost_v1")

    def test_brier_plausible(
        self, synthetic_modeling_df: pd.DataFrame, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Holdout Brier should be in a sensible range for random data."""
        from gridiron_edge.models.game_prediction import predictor as pred_mod

        monkeypatch.setattr(pred_mod, "_EPA_WINDOW_OPTIONS", [4])

        metadata = pred_mod._train_xgboost(
            synthetic_modeling_df,
            model_version="xgboost_v1",
            feature_fn=pred_mod._make_combined_features,
            feature_names=pred_mod._COMBINED_FEATURES,
            repo=mini_repo,
        )
        # Random data: Brier should be close to 0.25 (coin flip = 0.25)
        # Allow a generous range since small datasets can overfit
        assert 0.15 < metadata.holdout_brier < 0.35
