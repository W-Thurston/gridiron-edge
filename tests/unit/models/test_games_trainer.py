# tests/unit/models/test_games_trainer.py

"""Tests for GamesTrainer infrastructure.

Covers the static surface of the new trainer infrastructure:
    - GameModelType enum values.
    - GameModelSpec construction + frozen behavior.
    - _create_model for all valid (model_type, task) combinations.
    - _create_model raises for invalid combos (logistic + regression).
    - _get_param_grid returns nonempty grids for all valid combos.
    - _n_iter_for returns the expected counts.
    - WinProbTrainer.spec / TotalTrainer.spec correctness.
    - GamesTrainer.train() rejects unsupported model_type for its spec.

End-to-end fit-and-save smoke tests against real modeling data are
deferred to slow integration tests; this unit-test file exercises the
static surface only.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pandas as pd
import pytest

from gridiron_edge.models.game_prediction.base import (
    GameModelMetadata,
    GameModelSpec,
    GameModelType,
    _create_model,
    _get_param_grid,
    _n_iter_for,
)
from gridiron_edge.models.game_prediction.game_schema import (
    HOME_WIN_TARGET,
)
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer

# ---------------------------------------------------------------------------
# GameModelType
# ---------------------------------------------------------------------------


class TestGameModelType:
    """Enum value contract."""

    def test_values(self) -> None:
        assert GameModelType.LOGISTIC.value == "logistic"
        assert GameModelType.RANDOM_FOREST.value == "random_forest"
        assert GameModelType.XGBOOST.value == "xgboost"

    def test_is_string_enum(self) -> None:
        assert isinstance(GameModelType.LOGISTIC, str)
        assert GameModelType.LOGISTIC == "logistic"


# ---------------------------------------------------------------------------
# GameModelSpec
# ---------------------------------------------------------------------------


class TestGameModelSpec:
    """Dataclass shape, defaults, and immutability."""

    def test_construction(self) -> None:
        spec = GameModelSpec(
            name="win_prob",
            task="classification",
            target_col=HOME_WIN_TARGET,
            feature_set={
                GameModelType.LOGISTIC: object(),
            },
        )

        assert spec.name == "win_prob"
        assert spec.task == "classification"
        assert spec.target_col == HOME_WIN_TARGET
        assert spec.description == ""

    def test_frozen(self) -> None:
        spec = GameModelSpec(
            name="win_prob",
            task="classification",
            target_col=HOME_WIN_TARGET,
            feature_set={},
        )
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _create_model
# ---------------------------------------------------------------------------


class TestCreateModel:
    """Factory returns the right estimator + scaler per (model_type, task)."""

    @pytest.mark.parametrize(
        ("model_type", "task"),
        [
            (GameModelType.LOGISTIC, "classification"),
            (GameModelType.RANDOM_FOREST, "classification"),
            (GameModelType.XGBOOST, "classification"),
            (GameModelType.RANDOM_FOREST, "regression"),
            (GameModelType.XGBOOST, "regression"),
        ],
    )
    def test_returns_estimator_and_scaler(self, model_type: GameModelType, task: str) -> None:
        model, scaler = _create_model(model_type, task)
        assert model is not None
        if task == "classification" and model_type == GameModelType.LOGISTIC:
            assert scaler is not None
        else:
            assert scaler is None

    def test_logistic_regression_raises(self) -> None:
        with pytest.raises(ValueError, match="not a regression estimator"):
            _create_model(GameModelType.LOGISTIC, "regression")

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            _create_model(GameModelType.RANDOM_FOREST, "ranking")

    def test_rf_classification_is_calibrated(self) -> None:
        """RF-classification must be unconditionally wrapped in isotonic calibration."""
        # pyrefly: ignore [missing-import]
        from sklearn.calibration import CalibratedClassifierCV

        model, _ = _create_model(GameModelType.RANDOM_FOREST, "classification")
        assert isinstance(model, CalibratedClassifierCV)

    def test_rf_regression_is_plain(self) -> None:
        # pyrefly: ignore [missing-import]
        from sklearn.ensemble import RandomForestRegressor

        model, _ = _create_model(GameModelType.RANDOM_FOREST, "regression")
        assert isinstance(model, RandomForestRegressor)


# ---------------------------------------------------------------------------
# _get_param_grid
# ---------------------------------------------------------------------------


class TestGetParamGrid:
    """HP grid shape per (model_type, task)."""

    @pytest.mark.parametrize(
        ("model_type", "task"),
        [
            (GameModelType.LOGISTIC, "classification"),
            (GameModelType.RANDOM_FOREST, "classification"),
            (GameModelType.XGBOOST, "classification"),
            (GameModelType.RANDOM_FOREST, "regression"),
            (GameModelType.XGBOOST, "regression"),
        ],
    )
    def test_returns_nonempty_grid(self, model_type: GameModelType, task: str) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(model_type, task)
        assert isinstance(grid, list)
        assert len(grid) > 0
        assert all(isinstance(d, dict) for d in grid)

    def test_classification_grids_include_epa_window(self) -> None:
        grid = _get_param_grid(GameModelType.RANDOM_FOREST, "classification")
        assert all("epa_window" in d for d in grid)

    def test_regression_grids_exclude_epa_window(self) -> None:
        grid = _get_param_grid(GameModelType.RANDOM_FOREST, "regression")
        assert all("epa_window" not in d for d in grid)

    def test_logistic_regression_raises(self) -> None:
        with pytest.raises(ValueError, match="not a regression estimator"):
            _get_param_grid(GameModelType.LOGISTIC, "regression")


# ---------------------------------------------------------------------------
# _n_iter_for
# ---------------------------------------------------------------------------


class TestNIterFor:
    """Iteration count per (model_type, task)."""

    def test_logistic_classification_matches_epa_windows(self) -> None:
        from gridiron_edge.models.game_prediction._epa_window import (
            _EPA_WINDOW_OPTIONS,
        )

        assert _n_iter_for(GameModelType.LOGISTIC, "classification") == len(_EPA_WINDOW_OPTIONS)

    def test_rf_classification(self) -> None:
        assert _n_iter_for(GameModelType.RANDOM_FOREST, "classification") == 50

    def test_xgb_classification(self) -> None:
        assert _n_iter_for(GameModelType.XGBOOST, "classification") == 75

    def test_rf_regression(self) -> None:
        assert _n_iter_for(GameModelType.RANDOM_FOREST, "regression") == 50

    def test_xgb_regression(self) -> None:
        assert _n_iter_for(GameModelType.XGBOOST, "regression") == 50

    def test_logistic_regression_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            _n_iter_for(GameModelType.LOGISTIC, "regression")


# ---------------------------------------------------------------------------
# WinProbTrainer / TotalTrainer specs
# ---------------------------------------------------------------------------


class TestWinProbSpec:
    """WinProbTrainer.spec contract."""

    def test_name(self) -> None:
        assert WinProbTrainer().spec.name == "win_prob"

    def test_task(self) -> None:
        assert WinProbTrainer().spec.task == "classification"

    def test_target_col(self) -> None:
        assert WinProbTrainer().spec.target_col == HOME_WIN_TARGET

    def test_supports_all_three_classifiers(self) -> None:
        spec = WinProbTrainer().spec
        assert set(spec.feature_set.keys()) == {
            GameModelType.LOGISTIC,
            GameModelType.RANDOM_FOREST,
            GameModelType.XGBOOST,
        }


class TestTotalSpec:
    """TotalTrainer.spec contract."""

    def test_name(self) -> None:
        assert TotalTrainer().spec.name == "total"

    def test_task(self) -> None:
        assert TotalTrainer().spec.task == "regression"

    def test_target_col(self) -> None:
        assert TotalTrainer().spec.target_col == "actual_total"

    def test_excludes_logistic(self) -> None:
        spec = TotalTrainer().spec
        assert GameModelType.LOGISTIC not in spec.feature_set
        assert set(spec.feature_set.keys()) == {
            GameModelType.RANDOM_FOREST,
            GameModelType.XGBOOST,
        }


# ---------------------------------------------------------------------------
# GamesTrainer.train() spec validation
# ---------------------------------------------------------------------------


class TestTrainSpecValidation:
    """train() rejects model_type values not supported by spec."""

    def test_total_rejects_logistic(self) -> None:
        """TotalTrainer.train(model_type=LOGISTIC) raises before doing any work."""
        trainer = TotalTrainer()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="not supported by spec"):
            trainer.train(empty_df, model_type=GameModelType.LOGISTIC)


# ---------------------------------------------------------------------------
# GameModelMetadata smoke (also covered by test_metadata.py - included
# here to confirm the GamesTrainer-facing metadata contract is reachable).
# ---------------------------------------------------------------------------


class TestGameModelMetadataContract:
    """Confirm classification and regression metadata shape from this module."""

    def test_classification_construction(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-18T00:00:00",
            n_train_rows=100,
            n_holdout_rows=20,
            metrics={"brier": 0.22},
        )
        assert meta.task == "classification"
        assert meta.metrics["brier"] == pytest.approx(0.22)

    def test_regression_construction(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-18T00:00:00",
            n_train_rows=100,
            n_holdout_rows=20,
            metrics={"mae": 8.2, "rmse": 10.5, "r2": 0.31},
        )
        assert meta.task == "regression"
        assert meta.metrics["mae"] == pytest.approx(8.2)
        assert meta.metrics["rmse"] == pytest.approx(10.5)
        assert meta.metrics["r2"] == pytest.approx(0.31)


# ---------------------------------------------------------------------------
# Tests for TimeSeriesSplit inner CV (Unit 1b: game_base/H1, game_base/H2)
# ---------------------------------------------------------------------------


class TestInnerCVTemporalAwareness:
    """Verify inner CV uses TimeSeriesSplit instead of default StratifiedKFold.

    The outer HP search loop already uses TimeSeriesSplit. These tests ensure
    that the *inner* CVs nested inside the estimators are also temporally
    aware, closing game_base/H1 (LogisticRegressionCV) and game_base/H2
    (CalibratedClassifierCV).
    """

    def test_logistic_uses_timeseries_split(self) -> None:
        """LogisticRegressionCV.cv must be a TimeSeriesSplit instance."""
        # pyrefly: ignore [missing-import]
        from sklearn.model_selection import TimeSeriesSplit

        model, _ = _create_model(GameModelType.LOGISTIC, "classification")
        assert isinstance(model.cv, TimeSeriesSplit)

    def test_rf_calibration_uses_timeseries_split(self) -> None:
        """RF's CalibratedClassifierCV.cv must be a TimeSeriesSplit instance."""
        # pyrefly: ignore [missing-import]
        from sklearn.model_selection import TimeSeriesSplit

        model, _ = _create_model(GameModelType.RANDOM_FOREST, "classification")
        assert isinstance(model.cv, TimeSeriesSplit)

    def test_timeseries_split_n_splits_matches_cv_folds_for_logistic(self) -> None:
        """Logistic's inner CV n_splits should equal _CV_FOLDS for consistency."""
        from gridiron_edge.models.game_prediction.base import _CV_FOLDS

        model, _ = _create_model(GameModelType.LOGISTIC, "classification")
        assert model.cv.n_splits == _CV_FOLDS

    def test_calibration_n_splits_is_three(self) -> None:
        """RF calibration uses n_splits=3 (smaller than CV_FOLDS because
        the calibration curve fit is a simpler problem than HP selection)."""
        model, _ = _create_model(GameModelType.RANDOM_FOREST, "classification")
        assert model.cv.n_splits == 3
