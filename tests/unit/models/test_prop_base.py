# tests/unit/models/test_prop_base.py
"""Tests for gridiron_edge.models.prop_prediction.base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray
from numpy.random import Generator
import pandas as pd
from pandas import Series
import pytest

from gridiron_edge.models.prop_prediction.base import (
    _CV_FOLDS,
    _MIN_ATTEMPTS,
    PropModelMetadata,
    PropModelSpec,
    PropModelType,
    PropTrainer,
    evaluate_props,
)


class TestPropModelSpec:
    def test_required_fields(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2024-01-01T00:00:00Z",
            target_col="passing_yards",
            metrics={"mae": 25.0, "rmse": 32.0, "r2": 0.45},
        )
        assert meta.model_name == "qb_pass_yards"
        assert meta.model_type == "elasticnet"
        assert meta.task == "regression"
        assert meta.target_col == "passing_yards"
        assert meta.metrics["mae"] == pytest.approx(25.0)
        assert meta.metrics["rmse"] == pytest.approx(32.0)
        assert meta.metrics["r2"] == pytest.approx(0.45)

    def test_description_defaults_empty(self) -> None:
        spec = PropModelSpec(
            name="test",
            target_col="col",
            position_filter=["QB"],
        )
        assert spec.description == ""

    def test_frozen(self) -> None:
        spec = PropModelSpec(
            name="test",
            target_col="col",
            position_filter=["QB"],
        )
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


class TestPropModelMetadata:
    def test_required_fields(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2024-01-01T00:00:00Z",
            target_col="passing_yards",
            metrics={"mae": 25.0, "rmse": 32.0, "r2": 0.45},
        )
        assert meta.model_name == "qb_pass_yards"
        assert meta.model_type == "elasticnet"
        assert meta.task == "regression"
        assert meta.target_col == "passing_yards"
        assert meta.metrics["mae"] == pytest.approx(25.0)
        assert meta.metrics["rmse"] == pytest.approx(32.0)
        assert meta.metrics["r2"] == pytest.approx(0.45)

    def test_defaults(self) -> None:
        meta = PropModelMetadata(
            model_name="test",
            model_type="elasticnet",
            task="regression",
            trained_at="now",
            target_col="col",
            metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
        )
        # Inherited BaseModelMetadata defaults
        assert meta.schema_version == 3
        assert meta.training_seasons == []
        assert meta.holdout_seasons == []
        assert meta.parameters == {}
        assert meta.feature_columns == []
        assert meta.n_train_rows == 0
        assert meta.n_holdout_rows == 0
        assert meta.notes == ""

    def test_model_type_required(self) -> None:
        """model_type and task are required kwargs (Workstream 2)."""
        with pytest.raises(TypeError):
            PropModelMetadata(  # type: ignore[call-arg]
                model_name="test",
                trained_at="now",
                target_col="col",
                metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                task="regression",
            )
        with pytest.raises(TypeError):
            PropModelMetadata(  # type: ignore[call-arg]
                model_name="test",
                trained_at="now",
                target_col="col",
                metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                model_type="elasticnet",
            )

        # Both supplied → construction succeeds with the chosen model_type
        meta = PropModelMetadata(
            model_name="test",
            model_type="random_forest",
            task="regression",
            trained_at="now",
            target_col="col",
            metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
        )
        assert meta.model_type == "random_forest"
        assert meta.task == "regression"


class TestEvaluateProps:
    def test_perfect_prediction(self) -> None:
        y: ndarray = np.array([100.0, 200.0, 300.0])
        metrics: dict[str, float] = evaluate_props(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["median_ae"] == pytest.approx(0.0)

    def test_known_values(self) -> None:
        y_true: ndarray = np.array([250.0, 300.0, 200.0, 275.0, 320.0])
        y_pred: ndarray = np.array([240.0, 310.0, 190.0, 260.0, 305.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(12.0)
        assert metrics["rmse"] == pytest.approx(12.247, abs=0.01)
        assert metrics["r2"] == pytest.approx(0.914, abs=0.01)

    def test_constant_prediction(self) -> None:
        """Predicting the mean should give R²=0."""
        y_true: ndarray = np.array([100.0, 200.0, 300.0])
        y_pred: ndarray = np.array([200.0, 200.0, 200.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["r2"] == pytest.approx(0.0)

    def test_worse_than_mean(self) -> None:
        """Terrible predictions should give negative R²."""
        y_true: ndarray = np.array([100.0, 200.0, 300.0])
        y_pred: ndarray = np.array([500.0, 500.0, 500.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["r2"] < 0.0

    def test_single_value(self) -> None:
        """Single observation — R² should be 0 (ss_tot = 0)."""
        metrics: dict[str, float] = evaluate_props(np.array([100.0]), np.array([110.0]))
        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["r2"] == pytest.approx(0.0)


class TestMinAttempts:
    def test_passing_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["passing_yards"]
        assert col == "attempts"
        assert min_val == 10

    def test_rushing_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["rushing_yards"]
        assert col == "carries"
        assert min_val == 5

    def test_receiving_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["receiving_yards"]
        assert col == "targets"
        assert min_val == 2


class _StubTrainer(PropTrainer):
    """Minimal concrete trainer for base class tests."""

    @property
    def spec(self) -> PropModelSpec:
        return PropModelSpec(
            name="stub",
            target_col="rushing_yards",
            position_filter=["RB"],
            description="Test stub",
            clip_hi=250,
        )


class TestPredictNotFitted:
    def test_not_fitted_raises(self) -> None:
        trainer = _StubTrainer()
        # _model is None before train() is called
        dummy = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)


# ---------------------------------------------------------------------------
# Tests for _fit() TimeSeriesSplit CV discipline (Unit 1: prop_base/C1, C2)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic training data large enough for 5-fold TimeSeriesSplit.

    Target is a noisy linear function of feat_a so any reasonable
    regressor learns a non-trivial signal. Size chosen so each fold
    has enough samples to fit ElasticNet meaningfully.
    """
    rng: Generator = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": rng.normal(size=n),
        }
    )
    y: Series = pd.Series(x["feat_a"] * 10.0 + rng.normal(scale=2.0, size=n), name="y")
    return x, y


class TestFitCVDiscipline:
    """Verify _fit uses TimeSeriesSplit inner CV without touching holdout."""

    def test_returns_best_params_with_cv_mae(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """_fit returns best_params dict including cv_mae from inner CV."""
        x_train, y_train = synthetic_training_data
        trainer = _StubTrainer()
        params: dict[str, Any] = trainer._fit(x_train, y_train, model_type=PropModelType.ELASTICNET)

        assert "cv_mae" in params
        assert isinstance(params["cv_mae"], float)
        assert params["cv_mae"] > 0
        # ElasticNet HPs should also be present
        assert "alpha" in params
        assert "l1_ratio" in params

    def test_sets_model_and_scaler(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """_fit populates self._model and self._scaler for downstream _predict."""
        x_train, y_train = synthetic_training_data
        trainer = _StubTrainer()
        trainer._fit(x_train, y_train, model_type=PropModelType.ELASTICNET)

        assert trainer._model is not None
        # ElasticNet always gets a StandardScaler
        assert trainer._scaler is not None

    def test_fitted_model_predicts(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """End-to-end: _fit then _predict produces sensible outputs."""
        x_train, y_train = synthetic_training_data
        trainer = _StubTrainer()
        trainer._fit(x_train, y_train, model_type=PropModelType.ELASTICNET)

        preds: ndarray = trainer._predict(x_train.head(10))
        assert len(preds) == 10
        # Predictions clipped by stub spec to [0, 250]
        assert preds.min() >= 0
        assert preds.max() <= 250

    def test_raises_when_training_too_small_for_cv(self) -> None:
        """_fit raises when training set is too small for TimeSeriesSplit folds."""
        # TimeSeriesSplit requires at least n_splits+1 samples; we give 3.
        x_train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]})
        y_train: Series = pd.Series([1.0, 2.0, 3.0])
        trainer = _StubTrainer()

        with pytest.raises((RuntimeError, ValueError)):
            trainer._fit(x_train, y_train, model_type=PropModelType.ELASTICNET)

    def test_cv_folds_matches_games_trainer(self) -> None:
        """_CV_FOLDS in prop_base matches GamesTrainer's _CV_FOLDS.

        Structural-consistency check. If someone changes one side's fold
        count without considering the other, this test forces the conversation.
        """
        from gridiron_edge.models.game_prediction.base import _CV_FOLDS as GAMES_CV_FOLDS

        assert _CV_FOLDS == GAMES_CV_FOLDS


def test_train_and_save_persists_artifact(tmp_path: Path) -> None:
    """train_and_save writes a usable artifact via ArtifactStore."""
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.prop_prediction.base import (
        PropModelMetadata,
        PropModelType,
    )
    from gridiron_edge.models.prop_prediction.qb_pass_yards import (
        QBPassYardsTrainer,
    )

    trainer = QBPassYardsTrainer()

    # Inject fake fitted state so we can verify persistence without
    # running the full training pipeline (which requires player game
    # logs on disk).
    trainer._model = {"fake": "model"}
    trainer._scaler = {"fake": "scaler"}

    fake_meta = PropModelMetadata(
        model_name=trainer.spec.name,
        model_type=PropModelType.ELASTICNET.value,
        task="regression",
        trained_at="2026-06-20T00:00:00",
        target_col=trainer.spec.target_col,
        metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
    )

    # Monkey-patch `train` to skip the heavy path and exercise only
    # the persistence side of `train_and_save`.
    original_train = trainer.train
    try:
        trainer.train = lambda *_, **__: fake_meta  # type: ignore[assignment]
        trainer.train_and_save(repo=tmp_path)
    finally:
        trainer.train = original_train  # type: ignore[assignment]

    store = ArtifactStore(tmp_path)
    assert store.is_trained(trainer.spec.name, PropModelType.ELASTICNET.value)
    loaded = store.load(trainer.spec.name, PropModelType.ELASTICNET.value)
    assert loaded == {"fake": "model"}
