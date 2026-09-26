# src/gridiron_edge/evaluation/logistic_explanation_evidence_builder.py
"""Build and persist one Logistic explanation evidence batch for one run.

Reconstructs exact scaled-feature-by-coefficient log-odds contributions from
one run's already-persisted ``win_prob``/``logistic`` prediction-input
evidence. Performs no feature construction, scaling, or model inference of
its own: the transformed feature vector and estimator output are read
directly from that evidence. The model is reloaded from its exact immutable
binary snapshot (content-addressed under the prediction-input evidence
store), never from the live, mutable ``data/models/win_prob/logistic/``
artifact store, since a later champion retrain overwrites that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
    LogisticExplanationBatch,
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    create_logistic_explanation_batch,
    sigmoid,
)
from gridiron_edge.evaluation.logistic_explanation_evidence_store import (
    read_logistic_explanation_batch,
    write_logistic_explanation_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionExecutionKind,
    StatisticalPredictionEventEvidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    authenticate_binary_snapshot,
    list_prediction_input_evidence_by_run,
)


@dataclass(frozen=True, slots=True)
class LogisticExplanationBuildResult:
    """Accounting for one persisted and replay-verified explanation batch."""

    batch: LogisticExplanationBatch
    manifest_path: Path
    event_count: int
    max_reconciliation_error: float


def build_and_write_logistic_explanation_batch(
    *,
    run_id: str,
    generated_at: datetime,
    repo: Path,
) -> LogisticExplanationBuildResult:
    """Reconstruct and persist one run's Logistic explanation evidence.

    Args:
        run_id: Exact run identity whose ``win_prob``/``logistic``
            prediction-input evidence should be explained.
        generated_at: Timezone-aware UTC generation timestamp.
        repo: Repository root.

    Raises:
        ValueError: If the run has no (or more than one) win_prob/logistic
            persisted-estimator evidence, if the reloaded artifact is not a
            fitted Logistic estimator matching the evidence's feature schema,
            or if any event's reconstructed probability does not reconcile
            with its persisted ``raw_estimator_output`` within tolerance.
    """
    _require_utc(generated_at)
    normalized_run_id = _require_text(run_id, "run_id")

    evidence_candidates = [
        evidence
        for evidence in list_prediction_input_evidence_by_run(normalized_run_id, repo=repo)
        if evidence.model_name == "win_prob"
        and evidence.model_type == "logistic"
        and evidence.execution_kind is PredictionExecutionKind.PERSISTED_ESTIMATOR
    ]
    if not evidence_candidates:
        raise ValueError(
            "No win_prob/logistic persisted-estimator evidence found for "
            f"run_id={normalized_run_id!r}."
        )
    if len(evidence_candidates) > 1:
        raise ValueError(
            "Multiple win_prob/logistic persisted-estimator evidence artifacts found for "
            f"run_id={normalized_run_id!r}."
        )
    evidence = evidence_candidates[0]
    if evidence.feature_schema is None:
        raise ValueError("win_prob/logistic evidence is missing its feature_schema.")

    by_kind = {reference.kind: reference for reference in evidence.binary_artifacts}
    model_reference = by_kind.get(PredictionArtifactKind.MODEL)
    scaler_reference = by_kind.get(PredictionArtifactKind.SCALER)
    if model_reference is None or model_reference.state is not PredictionArtifactState.PRESENT:
        raise ValueError("win_prob/logistic evidence requires a present model artifact.")
    if scaler_reference is None or scaler_reference.state is not PredictionArtifactState.PRESENT:
        raise ValueError("win_prob/logistic evidence requires a present scaler artifact.")

    model_path = authenticate_binary_snapshot(model_reference, repo=repo)
    scaler_path = authenticate_binary_snapshot(scaler_reference, repo=repo)
    if model_path is None or scaler_path is None:
        raise ValueError("win_prob/logistic evidence binary snapshots are unexpectedly absent.")

    model = _load_joblib(model_path)
    # Loaded only to authenticate the snapshot is a loadable artifact; the
    # scaler's transform is never re-applied, since evidence already carries
    # the exact transformed values the estimator consumed.
    _load_joblib(scaler_path)

    ordered_columns = evidence.feature_schema.ordered_columns
    coefficients, intercept = _require_logistic_coefficients(
        model,
        expected_count=len(ordered_columns),
    )

    events = tuple(
        _explanation_event(
            statistical_event,
            feature_names=ordered_columns,
            coefficients=coefficients,
            intercept=intercept,
        )
        for statistical_event in evidence.statistical_events
    )

    assert model_reference.content_digest is not None
    assert scaler_reference.content_digest is not None
    batch = create_logistic_explanation_batch(
        run_id=evidence.run_id,
        evidence_id=evidence.evidence_id,
        model_content_digest=model_reference.content_digest,
        scaler_content_digest=scaler_reference.content_digest,
        feature_schema=evidence.feature_schema,
        generated_at=generated_at,
        events=events,
    )
    manifest_path = write_logistic_explanation_batch(batch, repo=repo)
    stored = read_logistic_explanation_batch(manifest_path)
    if stored != batch:
        raise ValueError("Stored logistic explanation batch does not exactly replay input.")

    max_error = max(
        abs(event.reconstructed_probability - event.raw_estimator_output) for event in events
    )
    return LogisticExplanationBuildResult(
        batch=batch,
        manifest_path=manifest_path,
        event_count=len(events),
        max_reconciliation_error=max_error,
    )


def _explanation_event(
    statistical_event: StatisticalPredictionEventEvidence,
    *,
    feature_names: tuple[str, ...],
    coefficients: tuple[float, ...],
    intercept: float,
) -> LogisticExplanationEvent:
    transformed = statistical_event.transformed_feature_values
    if len(transformed) != len(feature_names):
        raise ValueError(
            f"Event {statistical_event.event_id!r} transformed feature vector length "
            "does not match the feature schema."
        )

    contributions = tuple(
        LogisticFeatureContribution(
            feature_name=name,
            transformed_value=value,
            coefficient=coefficient,
            contribution=coefficient * value,
        )
        for name, value, coefficient in zip(feature_names, transformed, coefficients, strict=True)
    )
    reconstructed_log_odds = intercept + sum(item.contribution for item in contributions)
    reconstructed_probability = sigmoid(reconstructed_log_odds)

    return LogisticExplanationEvent(
        event_id=statistical_event.event_id,
        game_id=statistical_event.game_id,
        intercept=intercept,
        contributions=contributions,
        reconstructed_log_odds=reconstructed_log_odds,
        reconstructed_probability=reconstructed_probability,
        raw_estimator_output=statistical_event.raw_estimator_output,
        tolerance=LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
    )


def _require_logistic_coefficients(
    model: object,
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], float]:
    coefficients = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coefficients is None or intercept is None:
        raise ValueError(
            "Reloaded model does not expose coef_/intercept_; expected a fitted "
            "binary Logistic estimator."
        )
    coefficient_array = np.asarray(coefficients)
    intercept_array = np.asarray(intercept)
    if coefficient_array.shape != (1, expected_count):
        raise ValueError(
            f"Reloaded model coef_ shape {coefficient_array.shape} does not match "
            f"the feature schema's {expected_count} ordered columns."
        )
    if intercept_array.shape != (1,):
        raise ValueError(
            f"Reloaded model intercept_ shape {intercept_array.shape} is not a single "
            "binary-classification intercept."
        )
    return tuple(float(value) for value in coefficient_array[0]), float(intercept_array[0])


def _load_joblib(path: Path) -> object:
    import joblib  # type: ignore[import-untyped]

    return joblib.load(path)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC.")


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()
