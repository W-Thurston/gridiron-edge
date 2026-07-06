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
        """Single observation - R² should be 0 (ss_tot = 0)."""
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


class TestPredictWithMeta:
    """Tests for PropTrainer.predict_with_meta - the single sanctioned
    prediction path outside of the internal fit-and-evaluate flow.

    Invariant: `meta.feature_columns` is the sole source of truth for
    which columns get passed to the fitted model. Callers must not
    re-derive a feature list at predict time.
    """

    def _fit_stub(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> tuple[_StubTrainer, PropModelMetadata]:
        """Helper: fit a stub trainer on synthetic data and hand-build meta
        so tests do not depend on train() / train_through() plumbing."""
        x_train, y_train = synthetic_training_data
        trainer = _StubTrainer()
        trainer._fit(x_train, y_train, model_type=PropModelType.ELASTICNET)

        meta = PropModelMetadata(
            model_name=trainer.spec.name,
            model_type=PropModelType.ELASTICNET.value,
            task="regression",
            trained_at="2026-06-22T12:00:00",
            target_col=trainer.spec.target_col,
            feature_columns=list(x_train.columns),
            metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
        )
        return trainer, meta

    def test_uses_meta_feature_columns_verbatim(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Extra columns in the prediction df must be ignored, not fed to the model."""
        trainer, meta = self._fit_stub(synthetic_training_data)
        x_train, _ = synthetic_training_data

        predict_df = x_train.head(5).copy()
        predict_df["extra_col"] = 999.0  # not in meta.feature_columns

        preds, predicted_df = trainer.predict_with_meta(predict_df, meta)

        assert len(preds) == 5
        assert list(predicted_df.columns) == list(predict_df.columns)  # unchanged
        # The prediction succeeded, which means the extra column was ignored.

    def test_raises_when_meta_column_missing_from_df(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Missing a fit-time column is a pipeline regression, not a NaN issue."""
        trainer, meta = self._fit_stub(synthetic_training_data)
        x_train, _ = synthetic_training_data

        predict_df = x_train.drop(columns=["feat_b"]).head(5)

        with pytest.raises(RuntimeError, match="missing columns present at fit time"):
            trainer.predict_with_meta(predict_df, meta)

    def test_drops_rows_with_nan_in_fit_time_cols(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """A row with NaN in a fit-time column is dropped before predict."""
        trainer, meta = self._fit_stub(synthetic_training_data)
        x_train, _ = synthetic_training_data

        predict_df = x_train.head(5).copy()
        predict_df.iloc[0, predict_df.columns.get_loc("feat_a")] = np.nan

        preds, predicted_df = trainer.predict_with_meta(predict_df, meta)

        assert len(preds) == 4
        assert len(predicted_df) == 4
        assert predicted_df.index.tolist() == [1, 2, 3, 4]

    def test_ignores_nan_in_non_fit_time_cols(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """A row with NaN in a column *not* in meta.feature_columns is kept."""
        trainer, meta = self._fit_stub(synthetic_training_data)
        x_train, _ = synthetic_training_data

        predict_df = x_train.head(5).copy()
        predict_df["extra_col"] = [1.0, np.nan, 3.0, 4.0, 5.0]

        preds, predicted_df = trainer.predict_with_meta(predict_df, meta)

        assert len(preds) == 5
        assert len(predicted_df) == 5

    def test_empty_df_returns_empty(
        self,
        synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """An empty prediction df returns empty arrays cleanly."""
        trainer, meta = self._fit_stub(synthetic_training_data)
        x_train, _ = synthetic_training_data

        empty_df = x_train.iloc[0:0]
        preds, predicted_df = trainer.predict_with_meta(empty_df, meta)

        assert len(preds) == 0
        assert predicted_df.empty


class TestTrainThroughFeatureColumnsInvariant:
    """meta.feature_columns must equal the exact columns passed to fit.

    This is the invariant predict_with_meta relies on. If train_through
    ever recorded a feature set that disagrees with what was fit,
    predict_with_meta would silently pass the wrong columns.
    """

    def test_meta_feature_columns_matches_fit_time_features(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Feed data where an era-boundary column is 100% NaN in the
        training slice. Assert it is (a) absent from meta.feature_columns
        and (b) absent from the columns the model was fit with.
        """
        import pandas as pd

        rng = np.random.default_rng(42)
        n_per_season = 100
        rows = []
        for season in range(2015, 2022):
            for _ in range(n_per_season):
                rows.append(
                    {
                        "season": season,
                        "week": 1,
                        "player_id": "a",
                        "game_id": f"g{season}_{_}",
                        "rushing_yards": float(rng.integers(20, 150)),
                        "carries": float(rng.integers(5, 25)),
                        # feat_common: populated all seasons
                        "feat_common": float(rng.normal()),
                        # feat_era: 100% NaN before 2020
                        "feat_era": (float(rng.normal()) if season >= 2020 else float("nan")),
                    }
                )
        synthetic = pd.DataFrame(rows)

        # Stub the feature pipeline to return our synthetic frame.
        monkeypatch.setattr(
            "gridiron_edge.models.prop_prediction.base.build_prop_features",
            lambda *_, **__: synthetic,
        )

        # Stub feature column list to include both columns.
        class _EraTrainer(_StubTrainer):
            def _feature_columns(self) -> list[str]:
                return ["feat_common", "feat_era"]

        trainer = _EraTrainer()

        # Cutoff 2019: train slice = 2015-2018 → feat_era is 100% NaN → dropped.
        meta = trainer.train_through(cutoff_season=2019, model_type=PropModelType.ELASTICNET)
        assert meta.feature_columns == ["feat_common"]

        # Cutoff 2021: train slice = 2015-2020 → feat_era ~14% NaN → kept.
        # (5/6 seasons are NaN so the raw ratio is >50% → still dropped.
        # That's actually the era-boundary case working correctly.)
        # We assert only that meta.feature_columns is a subset of the
        # trainer's declared feature columns.
        meta_late = trainer.train_through(cutoff_season=2021, model_type=PropModelType.ELASTICNET)
        assert set(meta_late.feature_columns).issubset({"feat_common", "feat_era"})


class TestExcludeFeaturePrefixes:
    """Structurally-invalid feature families are stripped before any
    NaN-based filtering.

    This exists because the 50% NaN threshold in `_filter_and_split` is
    unreliable at position boundaries: sporadic non-null rows in
    structurally-invalid features (trick plays, halfback passes) can
    push a column's NaN rate just under the threshold. The column is
    then kept in training but is ~100% NaN in the prediction slice,
    which collapses the holdout via dropna.
    """

    def test_default_is_empty_tuple(self) -> None:
        """Existing specs without an explicit list see no exclusion."""
        spec = PropModelSpec(
            name="test",
            target_col="col",
            position_filter=["QB"],
        )
        assert spec.exclude_feature_prefixes == ()

    def test_prepare_features_strips_excluded_prefixes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prefix in exclude_feature_prefixes must be dropped from
        available features even if the column is present in the df."""
        synthetic = pd.DataFrame(
            {
                "season": [2020, 2020, 2021, 2021],
                "week": [1, 2, 1, 2],
                "player_id": ["a"] * 4,
                "game_id": [f"g{i}" for i in range(4)],
                "rushing_yards": [50.0, 60.0, 55.0, 65.0],
                "carries": [12.0, 15.0, 13.0, 14.0],
                # Structurally-invalid for this stub's position:
                "target_share_L3_mean": [float("nan")] * 4,
                "air_yards_share_L3_std": [0.1, 0.2, 0.3, 0.4],
            }
        )

        class _StubTrainerWithExclusion(_StubTrainer):
            @property
            def spec(self) -> PropModelSpec:
                return PropModelSpec(
                    name="stub",
                    target_col="rushing_yards",
                    position_filter=["RB"],
                    exclude_feature_prefixes=(
                        "target_share",
                        "air_yards_share",
                    ),
                )

            def _feature_columns(self) -> list[str]:
                return [
                    "carries",
                    "target_share_L3_mean",
                    "air_yards_share_L3_std",
                ]

        monkeypatch.setattr(
            "gridiron_edge.models.prop_prediction.base.build_prop_features",
            lambda *_, **__: synthetic,
        )

        trainer = _StubTrainerWithExclusion()
        _features_df, available = trainer._prepare_features()

        assert available == ["carries"]
        assert "target_share_L3_mean" not in available
        assert "air_yards_share_L3_std" not in available

    def test_prepare_features_no_exclusion_keeps_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When exclude_feature_prefixes is empty, nothing is stripped."""
        synthetic = pd.DataFrame(
            {
                "season": [2020, 2020],
                "week": [1, 2],
                "player_id": ["a", "a"],
                "game_id": ["g1", "g2"],
                "rushing_yards": [50.0, 60.0],
                "carries": [12.0, 15.0],
                "target_share_L3_mean": [0.1, 0.2],
            }
        )

        class _NoExclusionStub(_StubTrainer):
            def _feature_columns(self) -> list[str]:
                return ["carries", "target_share_L3_mean"]

        monkeypatch.setattr(
            "gridiron_edge.models.prop_prediction.base.build_prop_features",
            lambda *_, **__: synthetic,
        )

        trainer = _NoExclusionStub()
        _features_df, available = trainer._prepare_features()

        assert set(available) == {"carries", "target_share_L3_mean"}
