# tests/unit/models/game_prediction/test_statistical_prediction_execution.py
"""Tests for pure statistical prediction execution evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    create_prediction_feature_schema,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    build_statistical_prediction_execution,
)

COMMIT = "a" * 40
DIGEST = "b" * 64
REGISTRY_PATH = "data/output/calibration/game_model_calibration.json"


def _revision() -> SourceRevision:
    return SourceRevision(commit=COMMIT, tracked_worktree_clean=True)


def _sources() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest="c" * 64,
            size_bytes=100,
        ),
        SourceArtifactReference(
            relative_path=REGISTRY_PATH,
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
    )


def _artifact(
    kind: PredictionArtifactKind,
    *,
    state: PredictionArtifactState,
) -> BinaryArtifactReference:
    filenames = {
        PredictionArtifactKind.EXTERNAL_CALIBRATOR: "calibrator.joblib",
        PredictionArtifactKind.MODEL: "model.joblib",
        PredictionArtifactKind.MODEL_METADATA: "metadata.json",
        PredictionArtifactKind.SCALER: "scaler.joblib",
    }
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=("data/models/win_prob/random_forest/" + filenames[kind]),
        state=state,
        content_digest=DIGEST if state is PredictionArtifactState.PRESENT else None,
        size_bytes=100 if state is PredictionArtifactState.PRESENT else None,
    )


def _artifacts(
    *,
    scaler: PredictionArtifactState = PredictionArtifactState.ABSENT,
    calibrator: PredictionArtifactState = PredictionArtifactState.ABSENT,
) -> tuple[BinaryArtifactReference, ...]:
    return (
        _artifact(PredictionArtifactKind.EXTERNAL_CALIBRATOR, state=calibrator),
        _artifact(PredictionArtifactKind.MODEL, state=PredictionArtifactState.PRESENT),
        _artifact(
            PredictionArtifactKind.MODEL_METADATA,
            state=PredictionArtifactState.PRESENT,
        ),
        _artifact(PredictionArtifactKind.SCALER, state=scaler),
    )


def _schema(
    *,
    task: str = "classification",
    model_name: str = "win_prob",
    model_type: str = "random_forest",
):
    return create_prediction_feature_schema(
        model_name=model_name,
        model_type=model_type,
        task=task,
        modeling_schema_version=5,
        feature_set_name="test_features",
        ordered_columns=("feature_a", "feature_b"),
    )


def _post_processing(
    *,
    calibrator: PredictionArtifactState = PredictionArtifactState.ABSENT,
) -> PredictionPostProcessingEvidence:
    return PredictionPostProcessingEvidence(
        registry_reference=SourceArtifactReference(
            relative_path=REGISTRY_PATH,
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        registry_entry_updated_at=None,
        sigma=12.0,
        sigma_source=CalibrationResolutionSource.MODEL_FALLBACK,
        margin_std=13.5,
        margin_std_source=CalibrationResolutionSource.MODEL_FALLBACK,
        external_calibrator_state=calibrator,
        embedded_estimator_calibration=True,
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["game-2", "game-1"],
            "AWAY_WIN_PROB": [0.40, 0.45],
            "HOME_WIN_PROB": [0.60, 0.55],
        }
    )


def _classification_kwargs() -> dict[str, Any]:
    return {
        "game_ids": ("game-2", "game-1"),
        "raw_feature_values": ((2.0, 20.0), (1.0, 10.0)),
        "transformed_feature_values": ((2.0, 20.0), (1.0, 10.0)),
        "raw_estimator_outputs": (0.60, 0.55),
        "post_estimator_outputs": (0.60, 0.55),
        "source_revision": _revision(),
        "source_artifacts": _sources(),
        "binary_artifacts": _artifacts(),
        "feature_schema": _schema(),
        "post_processing": _post_processing(),
    }


def _build(
    predictions: pd.DataFrame | None = None,
    **overrides: Any,
):
    kwargs = _classification_kwargs()
    kwargs.update(overrides)
    return build_statistical_prediction_execution(
        _predictions() if predictions is None else predictions,
        **kwargs,
    )


class TestSuccessfulStatisticalExecution:
    def test_classification_preserves_exact_values_in_game_order(self) -> None:
        execution = _build()

        assert tuple(value.game_id for value in execution.computations) == (
            "game-1",
            "game-2",
        )
        first = execution.computations[0]
        assert first.raw_feature_values == (1.0, 10.0)
        assert first.transformed_feature_values == (1.0, 10.0)
        assert first.raw_estimator_output == pytest.approx(0.55)
        assert first.post_estimator_output == pytest.approx(0.55)
        assert execution.source_revision == _revision()
        assert execution.source_artifacts == _sources()
        assert execution.binary_artifacts == _artifacts()
        assert execution.feature_schema == _schema()
        assert execution.post_processing == _post_processing()

    def test_returns_defensive_prediction_copy(self) -> None:
        predictions = _predictions()
        execution = _build(predictions)

        predictions.loc[0, "HOME_WIN_PROB"] = 0.99

        assert execution.predictions.loc[0, "HOME_WIN_PROB"] == pytest.approx(0.60)

    def test_logistic_accepts_present_scaler_and_transformed_values(self) -> None:
        execution = _build(
            feature_schema=_schema(model_type="logistic"),
            binary_artifacts=_artifacts(scaler=PredictionArtifactState.PRESENT),
            transformed_feature_values=((0.2, 2.0), (0.1, 1.0)),
        )

        assert execution.computations[0].transformed_feature_values == (0.1, 1.0)

    def test_classification_accepts_present_external_calibrator(self) -> None:
        execution = _build(
            binary_artifacts=_artifacts(
                calibrator=PredictionArtifactState.PRESENT,
            ),
            post_processing=_post_processing(
                calibrator=PredictionArtifactState.PRESENT,
            ),
            post_estimator_outputs=(0.58, 0.53),
        )

        assert execution.computations[0].raw_estimator_output == pytest.approx(0.55)
        assert execution.computations[0].post_estimator_output == pytest.approx(0.53)

    def test_regression_accepts_unrestricted_finite_outputs(self) -> None:
        predictions = pd.DataFrame({"GAME_ID": ["game-2", "game-1"], "model_total": [47.0, 44.5]})
        execution = _build(
            predictions,
            feature_schema=_schema(
                task="regression",
                model_name="total",
                model_type="random_forest",
            ),
            post_processing=None,
            raw_estimator_outputs=(47.0, -4.5),
            post_estimator_outputs=(47.0, -4.5),
        )

        assert execution.computations[0].raw_estimator_output == pytest.approx(-4.5)
        assert execution.computations[0].post_estimator_output == pytest.approx(-4.5)


class TestPredictionAndCoverageValidation:
    def test_empty_predictions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _build(pd.DataFrame(columns=["GAME_ID"]))

    def test_missing_game_id_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing GAME_ID"):
            _build(pd.DataFrame({"HOME_WIN_PROB": [0.6, 0.55]}))

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_null_or_empty_prediction_game_id_is_rejected(self, value: object) -> None:
        predictions = _predictions().astype({"GAME_ID": object})
        predictions.loc[0, "GAME_ID"] = value

        with pytest.raises(ValueError, match="GAME_ID"):
            _build(predictions)

    def test_duplicate_prediction_game_ids_are_rejected(self) -> None:
        predictions = _predictions()
        predictions.loc[1, "GAME_ID"] = "game-2"

        with pytest.raises(ValueError, match="duplicate game IDs"):
            _build(predictions)

    def test_collection_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="align one-to-one"):
            _build(game_ids=("game-1",))

    def test_computation_coverage_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="coverage does not match"):
            _build(game_ids=("game-1", "unrelated-game"))

    def test_duplicate_computation_game_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate computation"):
            _build(game_ids=("game-1", "game-1"))


class TestFeatureAndOutputValidation:
    @pytest.mark.parametrize(
        ("argument", "value", "message"),
        [
            ("raw_feature_values", ((1.0,), (2.0,)), "Raw feature vector length"),
            (
                "transformed_feature_values",
                ((1.0,), (2.0,)),
                "Transformed feature vector length",
            ),
            (
                "raw_feature_values",
                ((float("nan"), 1.0), (2.0, 3.0)),
                "raw feature value",
            ),
            (
                "transformed_feature_values",
                ((float("inf"), 1.0), (2.0, 3.0)),
                "transformed feature value",
            ),
            (
                "raw_estimator_outputs",
                (float("nan"), 0.55),
                "raw_estimator_output",
            ),
            (
                "post_estimator_outputs",
                (0.60, float("inf")),
                "post_estimator_output",
            ),
        ],
    )
    def test_malformed_computation_values_are_rejected(
        self,
        argument: str,
        value: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _build(**{argument: value})

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("raw_estimator_outputs", (-0.01, 0.55)),
            ("raw_estimator_outputs", (1.01, 0.55)),
            ("post_estimator_outputs", (0.60, -0.01)),
            ("post_estimator_outputs", (0.60, 1.01)),
        ],
    )
    def test_classification_outputs_must_be_probabilities(
        self,
        argument: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            _build(**{argument: value})

    def test_absent_scaler_requires_identical_feature_values(self) -> None:
        with pytest.raises(ValueError, match="scaler is absent"):
            _build(transformed_feature_values=((0.2, 2.0), (0.1, 1.0)))


class TestArtifactAndFamilyValidation:
    @pytest.mark.parametrize(
        "kind",
        [PredictionArtifactKind.MODEL, PredictionArtifactKind.MODEL_METADATA],
    )
    def test_required_artifact_must_be_present(
        self,
        kind: PredictionArtifactKind,
    ) -> None:
        artifacts = tuple(
            replace(
                reference,
                state=PredictionArtifactState.ABSENT,
                content_digest=None,
                size_bytes=None,
            )
            if reference.kind is kind
            else reference
            for reference in _artifacts()
        )

        with pytest.raises(ValueError, match=f"{kind.value} artifact must be present"):
            _build(binary_artifacts=artifacts)

    def test_missing_artifact_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires exactly"):
            _build(binary_artifacts=_artifacts()[:-1])

    def test_unsorted_artifact_kinds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="ordered by unique artifact kind"):
            _build(binary_artifacts=tuple(reversed(_artifacts())))

    def test_logistic_requires_scaler(self) -> None:
        with pytest.raises(ValueError, match=r"Logistic.*present scaler"):
            _build(feature_schema=_schema(model_type="logistic"))

    def test_non_logistic_rejects_present_scaler(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"Non-logistic.*absent scaler",
        ):
            _build(binary_artifacts=_artifacts(scaler=PredictionArtifactState.PRESENT))

    def test_classification_requires_post_processing(self) -> None:
        with pytest.raises(ValueError, match="requires post_processing"):
            _build(post_processing=None)

    def test_regression_rejects_post_processing(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            _build(
                feature_schema=_schema(
                    task="regression",
                    model_name="total",
                    model_type="random_forest",
                )
            )

    def test_regression_rejects_external_calibrator(self) -> None:
        with pytest.raises(ValueError, match="absent external calibrator"):
            _build(
                feature_schema=_schema(
                    task="regression",
                    model_name="total",
                    model_type="random_forest",
                ),
                post_processing=None,
                binary_artifacts=_artifacts(
                    calibrator=PredictionArtifactState.PRESENT,
                ),
            )

    def test_calibrator_state_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="state does not match"):
            _build(
                binary_artifacts=_artifacts(
                    calibrator=PredictionArtifactState.PRESENT,
                )
            )

    def test_malformed_feature_schema_identity_is_rejected(self) -> None:
        malformed = replace(_schema(), schema_id="f" * 64)

        with pytest.raises(ValueError, match="schema_id does not match"):
            _build(feature_schema=malformed)


class TestStatisticalProvenanceValidation:
    def test_dirty_revision_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="clean tracked source revision"):
            _build(
                source_revision=SourceRevision(
                    commit=COMMIT,
                    tracked_worktree_clean=False,
                )
            )

    def test_invalid_commit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="lowercase Git commit SHA"):
            _build(
                source_revision=SourceRevision(
                    commit="not-a-commit",
                    tracked_worktree_clean=True,
                )
            )

    def test_unsorted_source_inventory_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            _build(source_artifacts=tuple(reversed(_sources())))
