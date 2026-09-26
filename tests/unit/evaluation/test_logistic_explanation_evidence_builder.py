"""Tests for building and persisting Logistic explanation evidence batches."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import uuid

import numpy as np
import pytest

from gridiron_edge.evaluation.logistic_explanation_evidence import sigmoid
from gridiron_edge.evaluation.logistic_explanation_evidence_builder import (
    build_and_write_logistic_explanation_batch,
)
from gridiron_edge.evaluation.logistic_explanation_evidence_store import (
    read_logistic_explanation_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    StatisticalPredictionEventEvidence,
    create_prediction_feature_schema,
    create_statistical_prediction_input_evidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import write_binary_snapshot

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
SEASON = "2026-2027"
WEEK = 2
COMMIT = "a" * 40
FEATURE_NAMES = ("ELO_DIFF", "OFF_EPA_PER_PLAY_DIFF")
COEFFICIENTS = (0.8, -0.4)
INTERCEPT = 0.1


class _StubEstimator:
    def __init__(self, coef: object, intercept: object) -> None:
        self.coef_ = coef
        self.intercept_ = intercept


def _snapshot(tmp_path: Path, obj: object) -> tuple[str, int]:
    import joblib

    source = tmp_path / f"source-{uuid.uuid4().hex}.joblib"
    joblib.dump(obj, source)
    digest = sha256(source.read_bytes()).hexdigest()
    size = source.stat().st_size
    write_binary_snapshot(source, expected_digest=digest, expected_size_bytes=size, repo=tmp_path)
    source.unlink()
    return digest, size


def _revision() -> SourceRevision:
    return SourceRevision(commit=COMMIT, tracked_worktree_clean=True)


def _source_artifacts() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionSourceState.PRESENT,
            content_digest="a" * 64,
            size_bytes=100,
        ),
    )


def _binary_artifacts(
    tmp_path: Path,
    *,
    model_obj: object,
) -> tuple[BinaryArtifactReference, ...]:
    model_digest, model_size = _snapshot(tmp_path, model_obj)
    scaler_digest, scaler_size = _snapshot(tmp_path, {"scaler": "stub"})
    return (
        BinaryArtifactReference(
            kind=PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            source_relative_path="data/models/win_prob/logistic/calibrator.joblib",
            state=PredictionArtifactState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL,
            source_relative_path="data/models/win_prob/logistic/model.joblib",
            state=PredictionArtifactState.PRESENT,
            content_digest=model_digest,
            size_bytes=model_size,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL_METADATA,
            source_relative_path="data/models/win_prob/logistic/metadata.json",
            state=PredictionArtifactState.PRESENT,
            content_digest="c" * 64,
            size_bytes=20,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.SCALER,
            source_relative_path="data/models/win_prob/logistic/scaler.joblib",
            state=PredictionArtifactState.PRESENT,
            content_digest=scaler_digest,
            size_bytes=scaler_size,
        ),
    )


def _feature_schema():
    return create_prediction_feature_schema(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        modeling_schema_version=5,
        epa_window=6,
        feature_set_name="combined_111",
        ordered_columns=FEATURE_NAMES,
    )


def _post_processing() -> PredictionPostProcessingEvidence:
    return PredictionPostProcessingEvidence(
        registry_reference=SourceArtifactReference(
            relative_path="data/output/calibration/game_model_calibration.json",
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        registry_entry_updated_at=None,
        sigma=None,
        sigma_source=CalibrationResolutionSource.NOT_USED,
        margin_std=None,
        margin_std_source=CalibrationResolutionSource.NOT_USED,
        external_calibrator_state=PredictionArtifactState.ABSENT,
        embedded_estimator_calibration=False,
    )


def _true_probability(transformed_values: tuple[float, ...]) -> float:
    log_odds = INTERCEPT + sum(
        coefficient * value
        for coefficient, value in zip(COEFFICIENTS, transformed_values, strict=True)
    )
    return sigmoid(log_odds)


def _statistical_events() -> tuple[StatisticalPredictionEventEvidence, ...]:
    event_1_transformed = (0.5, -0.25)
    event_2_transformed = (-0.3, 0.2)
    return (
        StatisticalPredictionEventEvidence(
            event_id="event-1",
            game_id="2026_02_A_B",
            raw_feature_values=(20.0, -0.05),
            transformed_feature_values=event_1_transformed,
            raw_estimator_output=_true_probability(event_1_transformed),
            post_estimator_output=_true_probability(event_1_transformed),
            final_outputs=(("home_win_prob", _true_probability(event_1_transformed)),),
        ),
        StatisticalPredictionEventEvidence(
            event_id="event-2",
            game_id="2026_02_C_D",
            raw_feature_values=(-12.0, 0.04),
            transformed_feature_values=event_2_transformed,
            raw_estimator_output=_true_probability(event_2_transformed),
            post_estimator_output=_true_probability(event_2_transformed),
            final_outputs=(("home_win_prob", _true_probability(event_2_transformed)),),
        ),
    )


def _write_evidence(
    tmp_path: Path,
    *,
    run_id: str = "run-1",
    model_obj: object | None = None,
    events: tuple[StatisticalPredictionEventEvidence, ...] | None = None,
):
    from gridiron_edge.evaluation.prediction_input_evidence_store import (
        write_prediction_input_evidence,
    )

    resolved_model = (
        _StubEstimator(
            coef=np.array([list(COEFFICIENTS)]),
            intercept=np.array([INTERCEPT]),
        )
        if model_obj is None
        else model_obj
    )
    evidence = create_statistical_prediction_input_evidence(
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
        model_name="win_prob",
        model_type="logistic",
        source_revision=_revision(),
        source_artifacts=_source_artifacts(),
        binary_artifacts=_binary_artifacts(tmp_path, model_obj=resolved_model),
        feature_schema=_feature_schema(),
        post_processing=_post_processing(),
        events=events if events is not None else _statistical_events(),
    )
    write_prediction_input_evidence(evidence, repo=tmp_path)
    return evidence


class TestBuildAndWriteLogisticExplanationBatch:
    def test_reconstructs_and_persists_within_tolerance(self, tmp_path: Path) -> None:
        evidence = _write_evidence(tmp_path)

        result = build_and_write_logistic_explanation_batch(
            run_id=evidence.run_id,
            generated_at=GENERATED_AT,
            repo=tmp_path,
        )

        assert result.event_count == 2
        assert result.max_reconciliation_error < 1e-9
        assert result.batch.evidence_id == evidence.evidence_id
        assert read_logistic_explanation_batch(result.manifest_path) == result.batch

        first_event = result.batch.events[0]
        assert [item.feature_name for item in first_event.contributions] == list(FEATURE_NAMES)
        assert first_event.contributions[0].coefficient == pytest.approx(COEFFICIENTS[0])

    def test_is_idempotent_on_rerun(self, tmp_path: Path) -> None:
        evidence = _write_evidence(tmp_path)

        first = build_and_write_logistic_explanation_batch(
            run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
        )
        second = build_and_write_logistic_explanation_batch(
            run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
        )

        assert first.batch.batch_id == second.batch.batch_id
        assert first.manifest_path == second.manifest_path

    def test_is_idempotent_on_rerun_with_different_timestamp(self, tmp_path: Path) -> None:
        """A real CLI rerun always passes a fresh `datetime.now(UTC)`.

        `generated_at` is excluded from `batch_id` (D56), so this must not
        raise "does not exactly replay input" even though the two calls'
        constructed (pre-write) batches differ in that one field.
        """
        evidence = _write_evidence(tmp_path)

        first = build_and_write_logistic_explanation_batch(
            run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
        )
        second = build_and_write_logistic_explanation_batch(
            run_id=evidence.run_id,
            generated_at=datetime(2027, 1, 1, tzinfo=UTC),
            repo=tmp_path,
        )

        assert first.batch.batch_id == second.batch.batch_id
        assert second.batch.generated_at == GENERATED_AT

    def test_rejects_run_with_no_logistic_evidence(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No win_prob/logistic"):
            build_and_write_logistic_explanation_batch(
                run_id="nonexistent-run", generated_at=GENERATED_AT, repo=tmp_path
            )

    def test_rejects_run_with_multiple_logistic_evidence(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prediction_input_evidence_store import (
            write_prediction_input_evidence,
        )

        evidence = _write_evidence(tmp_path, run_id="shared-run")
        second_evidence = create_statistical_prediction_input_evidence(
            run_id="shared-run",
            season=SEASON,
            week=WEEK + 1,
            generated_at=GENERATED_AT,
            model_name="win_prob",
            model_type="logistic",
            source_revision=_revision(),
            source_artifacts=_source_artifacts(),
            binary_artifacts=_binary_artifacts(
                tmp_path,
                model_obj=_StubEstimator(
                    coef=np.array([list(COEFFICIENTS)]), intercept=np.array([INTERCEPT])
                ),
            ),
            feature_schema=_feature_schema(),
            post_processing=_post_processing(),
            events=_statistical_events(),
        )
        write_prediction_input_evidence(second_evidence, repo=tmp_path)

        with pytest.raises(ValueError, match="Multiple win_prob/logistic"):
            build_and_write_logistic_explanation_batch(
                run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
            )

    def test_rejects_reloaded_object_without_coefficients(self, tmp_path: Path) -> None:
        evidence = _write_evidence(tmp_path, model_obj={"not_a_model": True})

        with pytest.raises(ValueError, match="does not expose coef_/intercept_"):
            build_and_write_logistic_explanation_batch(
                run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
            )

    def test_rejects_coefficient_count_mismatch(self, tmp_path: Path) -> None:
        mismatched = _StubEstimator(
            coef=np.array([[0.8, -0.4, 0.1]]),
            intercept=np.array([INTERCEPT]),
        )
        evidence = _write_evidence(tmp_path, model_obj=mismatched)

        with pytest.raises(ValueError, match="does not match"):
            build_and_write_logistic_explanation_batch(
                run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
            )

    def test_rejects_reconciliation_violation(self, tmp_path: Path) -> None:
        bad_events = (
            StatisticalPredictionEventEvidence(
                event_id="event-1",
                game_id="2026_02_A_B",
                raw_feature_values=(20.0, -0.05),
                transformed_feature_values=(0.5, -0.25),
                raw_estimator_output=0.01,
                post_estimator_output=0.01,
                final_outputs=(("home_win_prob", 0.01),),
            ),
        )
        evidence = _write_evidence(tmp_path, events=bad_events)

        with pytest.raises(ValueError, match="reconcile"):
            build_and_write_logistic_explanation_batch(
                run_id=evidence.run_id, generated_at=GENERATED_AT, repo=tmp_path
            )
