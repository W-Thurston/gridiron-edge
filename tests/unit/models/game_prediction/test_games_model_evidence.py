# tests/unit/models/game_prediction/test_games_model_evidence.py
"""Integration tests for evidence-aware upcoming statistical prediction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
)
from gridiron_edge.models.game_prediction._columns import (
    _SCHEMA_VERSION,
    FeatureSet,
)
from gridiron_edge.models.game_prediction.base import GameModelMetadata
from gridiron_edge.models.game_prediction.model import (
    GamesModel,
    TotalRandomForestModel,
    TotalXGBoostModel,
    WinProbLogisticModel,
    WinProbRandomForestModel,
    WinProbXGBoostModel,
)
from gridiron_edge.models.game_prediction.post_process import (
    PredictionPostProcessingResolution,
    enrich_predictions_with_resolution,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    StatisticalPredictionExecution,
)

_COMMIT = "a" * 40
_DIGEST = "b" * 64
_REGISTRY_PATH = "data/output/calibration/game_model_calibration.json"


def _revision() -> SourceRevision:
    return SourceRevision(
        commit=_COMMIT,
        tracked_worktree_clean=True,
    )


def _sources() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest="c" * 64,
            size_bytes=100,
        ),
        SourceArtifactReference(
            relative_path=_REGISTRY_PATH,
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
    )


def _artifact(
    kind: PredictionArtifactKind,
    *,
    state: PredictionArtifactState,
    model_name: str = "win_prob",
    model_type: str = "random_forest",
) -> BinaryArtifactReference:
    filenames = {
        PredictionArtifactKind.EXTERNAL_CALIBRATOR: "calibrator.joblib",
        PredictionArtifactKind.MODEL: "model.joblib",
        PredictionArtifactKind.MODEL_METADATA: "metadata.json",
        PredictionArtifactKind.SCALER: "scaler.joblib",
    }
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=(f"data/models/{model_name}/{model_type}/{filenames[kind]}"),
        state=state,
        content_digest=(_DIGEST if state is PredictionArtifactState.PRESENT else None),
        size_bytes=100 if state is PredictionArtifactState.PRESENT else None,
    )


def _artifacts(
    *,
    scaler: PredictionArtifactState = PredictionArtifactState.ABSENT,
    calibrator: PredictionArtifactState = PredictionArtifactState.ABSENT,
    model_name: str = "win_prob",
    model_type: str = "random_forest",
) -> tuple[BinaryArtifactReference, ...]:
    return (
        _artifact(
            PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            state=calibrator,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.MODEL,
            state=PredictionArtifactState.PRESENT,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.MODEL_METADATA,
            state=PredictionArtifactState.PRESENT,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.SCALER,
            state=scaler,
            model_name=model_name,
            model_type=model_type,
        ),
    )


def _schedule() -> DataFrame:
    return DataFrame(
        {
            "GAME_ID": ["game-1", "game-2"],
            "YEAR": ["2026-2027", "2026-2027"],
            "WEEK_NUM": [3, 3],
            "AWAY_TEAM": ["Away One", "Away Two"],
            "HOME_TEAM": ["Home One", "Home Two"],
        }
    )


def _feature_output() -> DataFrame:
    return DataFrame(
        {
            "GAME_ID": ["game-1", "game-2"],
            "YEAR": ["2026-2027", "2026-2027"],
            "WEEK_NUM": [3, 3],
            "AWAY_TEAM": ["Away One", "Away Two"],
            "HOME_TEAM": ["Home One", "Home Two"],
            "AWAY_ELO": [1500.0, 1510.0],
            "HOME_ELO": [1520.0, 1490.0],
            "feature_a": [1.0, 2.0],
            "feature_b": [10.0, 20.0],
        }
    )


def _feature_set(
    *,
    columns: tuple[str, ...] = ("feature_a", "feature_b"),
) -> tuple[FeatureSet, MagicMock]:
    feature_fn = MagicMock(return_value=_feature_output().loc[:, list(columns)].copy())
    return (
        FeatureSet(
            name="test_features",
            feature_fn=feature_fn,
            feature_names=["feature_a", "feature_b"],
        ),
        feature_fn,
    )


def _metadata(
    *,
    model_name: str = "win_prob",
    model_type: str = "random_forest",
    task: str = "classification",
    feature_set_name: str = "test_features",
    feature_columns: list[str] | None = None,
    modeling_schema_version: object = _SCHEMA_VERSION,
) -> GameModelMetadata:
    return GameModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task=task,
        trained_at="2026-09-21T12:00:00",
        parameters={
            "feature_set": feature_set_name,
            "modeling_schema_version": modeling_schema_version,
            "calibration_applied": False,
        },
        feature_columns=(
            feature_columns if feature_columns is not None else ["feature_a", "feature_b"]
        ),
    )


def _resolution(
    *,
    calibrator: object | None = None,
) -> PredictionPostProcessingResolution:
    return PredictionPostProcessingResolution(
        sigma=12.0,
        sigma_source=CalibrationResolutionSource.MODEL_FALLBACK,
        margin_std=13.5,
        margin_std_source=CalibrationResolutionSource.MODEL_FALLBACK,
        registry_entry_updated_at=None,
        calibrator=calibrator,  # type: ignore[arg-type]
    )


def _configure_store(
    store_class: MagicMock,
    *,
    metadata: GameModelMetadata,
    estimator: MagicMock,
    scaler: object | None,
) -> MagicMock:
    store = store_class.return_value
    store.is_trained.return_value = True
    store.read_metadata.return_value = metadata
    store.load.return_value = estimator
    store.load_scaler.return_value = scaler
    return store


_EQUIVALENCE_CASES: tuple[tuple[type[GamesModel], str, str], ...] = (
    (WinProbLogisticModel, "win_prob", "logistic"),
    (WinProbRandomForestModel, "win_prob", "random_forest"),
    (WinProbXGBoostModel, "win_prob", "xgboost"),
    (TotalRandomForestModel, "total", "random_forest"),
    (TotalXGBoostModel, "total", "xgboost"),
)


class TestLegacyEvidencePredictionEquivalence:
    """Evidence instrumentation must not alter statistical predictions."""

    @pytest.mark.parametrize(
        ("model_class", "model_name", "model_type"),
        _EQUIVALENCE_CASES,
    )
    @patch("gridiron_edge.models.game_prediction.model.enrich_predictions")
    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_legacy_and_evidence_paths_return_identical_predictions(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        enrich_predictions: MagicMock,
        model_class: type[GamesModel],
        model_name: str,
        model_type: str,
        tmp_path: Path,
    ) -> None:
        schedule = _schedule()
        original_schedule = schedule.copy(deep=True)
        feature_output = _feature_output()
        original_feature_output = feature_output.copy(deep=True)
        feature_set, _ = _feature_set()
        estimator = MagicMock()
        resolution = _resolution()
        scaler: MagicMock | None = None

        if model_name == "win_prob":
            estimator.predict_proba.return_value = np.array(
                [[0.40, 0.60], [0.30, 0.70]],
                dtype=float,
            )
            task = "classification"
        else:
            estimator.predict.return_value = np.array([44.5, 47.0], dtype=float)
            task = "regression"

        if model_type == "logistic":
            scaler = MagicMock()
            scaler.transform.return_value = np.array(
                [[0.1, 1.0], [0.2, 2.0]],
                dtype=float,
            )

        _configure_store(
            store_class,
            metadata=_metadata(
                model_name=model_name,
                model_type=model_type,
                task=task,
            ),
            estimator=estimator,
            scaler=scaler,
        )
        run_features.return_value = feature_output
        artifact_references.return_value = _artifacts(
            scaler=(
                PredictionArtifactState.PRESENT
                if model_type == "logistic"
                else PredictionArtifactState.ABSENT
            ),
            model_name=model_name,
            model_type=model_type,
        )
        resolve_post_processing.return_value = resolution
        enrich_predictions.side_effect = lambda frame, **_kwargs: (
            enrich_predictions_with_resolution(frame, resolution=resolution)
        )
        model = model_class()

        with (
            patch.object(model, "_feature_fn", return_value=feature_set.feature_fn),
            patch.object(model, "prediction_feature_set", return_value=feature_set),
        ):
            legacy = model.predict_upcoming(schedule, repo=tmp_path)
            execution = model.predict_upcoming_with_evidence(
                schedule,
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        assert set(legacy.columns) == set(execution.predictions.columns)

        pd.testing.assert_frame_equal(
            legacy,
            execution.predictions.loc[:, legacy.columns],
            check_dtype=True,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(schedule, original_schedule)
        pd.testing.assert_frame_equal(feature_output, original_feature_output)
        assert tuple(computation.game_id for computation in execution.computations) == tuple(
            execution.predictions["GAME_ID"].astype(str)
        )

        output_column = "HOME_WIN_PROB" if model_name == "win_prob" else "model_total"
        assert tuple(
            computation.post_estimator_output for computation in execution.computations
        ) == pytest.approx(tuple(execution.predictions[output_column].astype(float)))


class TestClassificationEvidenceExecution:
    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_random_forest_preserves_exact_execution(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        estimator.predict_proba.return_value = np.array([[0.40, 0.60], [0.30, 0.70]])
        store = _configure_store(
            store_class,
            metadata=_metadata(),
            estimator=estimator,
            scaler=None,
        )
        feature_set, feature_fn = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts()
        resolve_post_processing.return_value = _resolution()
        model = WinProbRandomForestModel()

        with patch.object(
            model,
            "prediction_feature_set",
            return_value=feature_set,
        ):
            execution = model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        assert isinstance(execution, StatisticalPredictionExecution)
        assert tuple(value.game_id for value in execution.computations) == (
            "game-1",
            "game-2",
        )
        expected_inputs = np.array([[1.0, 10.0], [2.0, 20.0]])
        np.testing.assert_array_equal(
            estimator.predict_proba.call_args.args[0],
            expected_inputs,
        )
        assert execution.computations[0].raw_feature_values == (1.0, 10.0)
        assert execution.computations[0].transformed_feature_values == (
            1.0,
            10.0,
        )
        assert execution.computations[0].raw_estimator_output == pytest.approx(0.60)
        assert execution.computations[0].post_estimator_output == pytest.approx(0.60)
        assert execution.predictions["HOME_TEAM_WIN_PROB"].tolist() == [
            "60.0 %",
            "70.0 %",
        ]
        run_features.assert_called_once()
        feature_fn.assert_called_once()
        artifact_references.assert_called_once_with(
            store,
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )
        store.read_metadata.assert_called_once_with("win_prob", "random_forest")
        store.load.assert_called_once_with("win_prob", "random_forest")
        store.load_scaler.assert_called_once_with("win_prob", "random_forest")
        estimator.predict_proba.assert_called_once()
        resolve_post_processing.assert_called_once_with(
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )

    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_logistic_records_exact_scaler_output(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        tmp_path: Path,
    ) -> None:
        transformed: NDArray[np.float64] = np.array(
            [[0.1, 1.0], [0.2, 2.0]],
            dtype=float,
        )
        scaler = MagicMock()
        scaler.transform.return_value = transformed
        estimator = MagicMock()
        estimator.predict_proba.return_value = np.array([[0.40, 0.60], [0.30, 0.70]])
        _configure_store(
            store_class,
            metadata=_metadata(model_type="logistic"),
            estimator=estimator,
            scaler=scaler,
        )
        feature_set, _ = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts(
            scaler=PredictionArtifactState.PRESENT,
            model_type="logistic",
        )
        resolve_post_processing.return_value = _resolution()
        model = WinProbLogisticModel()

        with patch.object(
            model,
            "prediction_feature_set",
            return_value=feature_set,
        ):
            execution = model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        scaler.transform.assert_called_once()
        np.testing.assert_array_equal(
            estimator.predict_proba.call_args.args[0],
            transformed,
        )
        assert execution.computations[0].raw_feature_values == (1.0, 10.0)
        assert execution.computations[0].transformed_feature_values == (0.1, 1.0)
        estimator.predict_proba.assert_called_once()

    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_external_calibrator_distinguishes_raw_and_post_outputs(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        estimator.predict_proba.return_value = np.array([[0.40, 0.60], [0.30, 0.70]])
        _configure_store(
            store_class,
            metadata=_metadata(),
            estimator=estimator,
            scaler=None,
        )
        calibrator = MagicMock()
        calibrator.predict.return_value = np.array([0.57, 0.67])
        feature_set, _ = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts(
            calibrator=PredictionArtifactState.PRESENT,
        )
        resolve_post_processing.return_value = _resolution(calibrator=calibrator)
        model = WinProbRandomForestModel()

        with patch.object(
            model,
            "prediction_feature_set",
            return_value=feature_set,
        ):
            execution = model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        calibrator.predict.assert_called_once()
        assert execution.computations[0].raw_estimator_output == pytest.approx(0.60)
        assert execution.computations[0].post_estimator_output == pytest.approx(0.57)
        assert execution.predictions["HOME_TEAM_WIN_PROB"].tolist() == [
            "57.0 %",
            "67.0 %",
        ]
        assert execution.predictions["AWAY_TEAM_WIN_PROB"].tolist() == [
            "43.0 %",
            "33.0 %",
        ]


class TestClassificationFailureOrdering:
    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            (_metadata(model_name="wrong"), "model_name does not match"),
            (_metadata(model_type="xgboost"), "model_type does not match"),
            (_metadata(task="regression"), "task does not match"),
            (_metadata(feature_set_name="wrong"), "feature_set does not match"),
            (
                _metadata(feature_columns=["feature_b", "feature_a"]),
                "feature_columns do not match",
            ),
            (
                _metadata(modeling_schema_version=_SCHEMA_VERSION + 1),
                "modeling_schema_version does not match",
            ),
        ],
    )
    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_metadata_mismatch_blocks_estimator(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        metadata: GameModelMetadata,
        message: str,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        store = _configure_store(
            store_class,
            metadata=metadata,
            estimator=estimator,
            scaler=None,
        )
        feature_set, _ = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts()
        model = WinProbRandomForestModel()

        with (
            patch.object(
                model,
                "prediction_feature_set",
                return_value=feature_set,
            ),
            pytest.raises(ValueError, match=message),
        ):
            model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        store.load.assert_not_called()
        store.load_scaler.assert_not_called()
        estimator.predict_proba.assert_not_called()
        resolve_post_processing.assert_not_called()

    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_runtime_feature_order_blocks_estimator(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        store = _configure_store(
            store_class,
            metadata=_metadata(),
            estimator=estimator,
            scaler=None,
        )
        feature_set, _ = _feature_set(columns=("feature_b", "feature_a"))
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts()
        model = WinProbRandomForestModel()

        with (
            patch.object(
                model,
                "prediction_feature_set",
                return_value=feature_set,
            ),
            pytest.raises(ValueError, match="Runtime prediction feature columns"),
        ):
            model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        store.load.assert_not_called()
        estimator.predict_proba.assert_not_called()
        resolve_post_processing.assert_not_called()

    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_missing_logistic_scaler_blocks_estimator_call(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        _configure_store(
            store_class,
            metadata=_metadata(model_type="logistic"),
            estimator=estimator,
            scaler=None,
        )
        feature_set, _ = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts(model_type="logistic")
        model = WinProbLogisticModel()

        with (
            patch.object(
                model,
                "prediction_feature_set",
                return_value=feature_set,
            ),
            pytest.raises(ValueError, match="requires a persisted scaler"),
        ):
            model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        estimator.predict_proba.assert_not_called()
        resolve_post_processing.assert_not_called()


class TestRegressionEvidenceExecution:
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_total_preserves_exact_execution_without_post_processing(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        estimator.predict.return_value = np.array([44.5, 47.0])
        store = _configure_store(
            store_class,
            metadata=_metadata(
                model_name="total",
                task="regression",
            ),
            estimator=estimator,
            scaler=None,
        )
        feature_set, feature_fn = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts(
            model_name="total",
        )
        model = TotalRandomForestModel()

        with patch.object(
            model,
            "prediction_feature_set",
            return_value=feature_set,
        ):
            execution = model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        assert isinstance(execution, StatisticalPredictionExecution)
        expected_inputs = np.array([[1.0, 10.0], [2.0, 20.0]])
        np.testing.assert_array_equal(
            estimator.predict.call_args.args[0],
            expected_inputs,
        )
        assert execution.post_processing is None
        assert execution.computations[0].raw_feature_values == (1.0, 10.0)
        assert execution.computations[0].transformed_feature_values == (
            1.0,
            10.0,
        )
        assert execution.computations[0].raw_estimator_output == pytest.approx(44.5)
        assert execution.computations[0].post_estimator_output == pytest.approx(44.5)
        assert execution.predictions["model_total"].tolist() == [44.5, 47.0]
        feature_fn.assert_called_once()
        store.load_scaler.assert_called_once_with("total", "random_forest")
        estimator.predict.assert_called_once()


class TestModelLayerPersistenceBoundary:
    @patch("gridiron_edge.evaluation.forecast_store.write_forecast_events")
    @patch(
        "gridiron_edge.evaluation.prediction_input_evidence_store.write_prediction_input_evidence"
    )
    @patch("gridiron_edge.evaluation.prediction_input_evidence_store.write_binary_snapshot")
    @patch("gridiron_edge.models.game_prediction.model.resolve_prediction_post_processing")
    @patch("gridiron_edge.models.game_prediction.model._statistical_artifact_references")
    @patch("gridiron_edge.models.game_prediction.model.run_features")
    @patch("gridiron_edge.models.game_prediction.model.ArtifactStore")
    def test_model_execution_performs_no_persistence(
        self,
        store_class: MagicMock,
        run_features: MagicMock,
        artifact_references: MagicMock,
        resolve_post_processing: MagicMock,
        write_binary_snapshot: MagicMock,
        write_evidence: MagicMock,
        write_events: MagicMock,
        tmp_path: Path,
    ) -> None:
        estimator = MagicMock()
        estimator.predict_proba.return_value = np.array([[0.40, 0.60], [0.30, 0.70]])
        _configure_store(
            store_class,
            metadata=_metadata(),
            estimator=estimator,
            scaler=None,
        )
        feature_set, _ = _feature_set()
        run_features.return_value = _feature_output()
        artifact_references.return_value = _artifacts()
        resolve_post_processing.return_value = _resolution()
        model = WinProbRandomForestModel()

        with patch.object(
            model,
            "prediction_feature_set",
            return_value=feature_set,
        ):
            model.predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        write_binary_snapshot.assert_not_called()
        write_evidence.assert_not_called()
        write_events.assert_not_called()
