# src/gridiron_edge/evaluation/logistic_explanation_evidence.py
"""Immutable Logistic Win explanation evidence contracts.

Persists exact scaled-feature-by-coefficient contributions in log-odds space
for one run's ``win_prob``/``logistic`` statistical prediction-input evidence.
Sibling to ``prediction_input_evidence.py`` and
``game_model_evaluation_report.py``, not an extension of either: this contract
decomposes one already-persisted estimator computation into per-feature
contributions, it does not re-run prediction or evaluate model quality.

A contribution is a linear decomposition of one fitted estimator's own
decision function (``coefficient * transformed_feature_value``), not a
causal effect estimate. It answers "how much did this feature move this
model's log-odds output," never "what would happen if this feature changed."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from typing import Final

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionFeatureSchema,
    prediction_feature_schema_id,
)

LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION: Final[int] = 1
LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE: Final[float] = 1e-9
_INTERNAL_CONSISTENCY_TOLERANCE: Final[float] = 1e-9
_DIGEST_PATTERN_LENGTH: Final[int] = 64


@dataclass(frozen=True, slots=True)
class LogisticFeatureContribution:
    """One feature's exact log-odds contribution to one event's prediction."""

    feature_name: str
    transformed_value: float
    coefficient: float
    contribution: float


@dataclass(frozen=True, slots=True)
class LogisticExplanationEvent:
    """Exact log-odds decomposition of one persisted Logistic prediction."""

    event_id: str
    game_id: str
    intercept: float
    contributions: tuple[LogisticFeatureContribution, ...]
    reconstructed_log_odds: float
    reconstructed_probability: float
    raw_estimator_output: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class LogisticExplanationBatch:
    """Frozen immutable batch of Logistic explanation evidence for one run."""

    schema_version: int
    batch_id: str
    run_id: str
    evidence_id: str
    model_content_digest: str
    scaler_content_digest: str
    feature_schema: PredictionFeatureSchema
    generated_at: datetime
    events: tuple[LogisticExplanationEvent, ...]


def logistic_explanation_batch_id(
    *,
    run_id: str,
    evidence_id: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    events: tuple[LogisticExplanationEvent, ...],
) -> str:
    """Return the SHA-256 identity of one canonical explanation batch payload.

    Deliberately excludes ``generated_at``: identity is determined entirely
    by the source run/evidence and the reconstructed events, so re-running
    ``explain-logistic`` against unchanged evidence reproduces the same
    ``batch_id`` (an idempotent write-or-replay) instead of a second batch
    that legitimately claims the same events (see `DECISIONS.md` D56 and
    D54's identical fix for comparable-games evidence).
    """
    payload = _identity_payload(
        run_id=run_id,
        evidence_id=evidence_id,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        events=events,
    )
    return _canonical_digest(payload)


def create_logistic_explanation_batch(
    *,
    run_id: str,
    evidence_id: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    generated_at: datetime,
    events: tuple[LogisticExplanationEvent, ...],
) -> LogisticExplanationBatch:
    """Create and validate one complete Logistic explanation batch."""
    batch_id = logistic_explanation_batch_id(
        run_id=run_id,
        evidence_id=evidence_id,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        events=events,
    )
    batch = LogisticExplanationBatch(
        schema_version=LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION,
        batch_id=batch_id,
        run_id=run_id,
        evidence_id=evidence_id,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        generated_at=generated_at,
        events=events,
    )
    validate_logistic_explanation_batch(batch)
    return batch


def validate_logistic_explanation_batch(batch: LogisticExplanationBatch) -> None:
    """Validate one batch's identity, invariants, and reconciliation."""
    if batch.schema_version != LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported logistic explanation evidence schema_version: {batch.schema_version}."
        )
    _digest(batch.batch_id, "batch_id")
    _text(batch.run_id, "run_id")
    _digest(batch.evidence_id, "evidence_id")
    _digest(batch.model_content_digest, "model_content_digest")
    _digest(batch.scaler_content_digest, "scaler_content_digest")
    _validate_feature_schema(batch.feature_schema)
    _utc(batch.generated_at, "generated_at")
    _validate_events(batch.events, feature_schema=batch.feature_schema)

    expected_id = logistic_explanation_batch_id(
        run_id=batch.run_id,
        evidence_id=batch.evidence_id,
        model_content_digest=batch.model_content_digest,
        scaler_content_digest=batch.scaler_content_digest,
        feature_schema=batch.feature_schema,
        events=batch.events,
    )
    if batch.batch_id != expected_id:
        raise ValueError("batch_id does not match canonical batch content.")


def logistic_explanation_batch_payload(
    batch: LogisticExplanationBatch,
) -> dict[str, object]:
    """Return the complete stable JSON-compatible batch representation."""
    validate_logistic_explanation_batch(batch)
    return {
        "schema_version": batch.schema_version,
        "batch_id": batch.batch_id,
        "generated_at": batch.generated_at.isoformat(),
        **_identity_payload(
            run_id=batch.run_id,
            evidence_id=batch.evidence_id,
            model_content_digest=batch.model_content_digest,
            scaler_content_digest=batch.scaler_content_digest,
            feature_schema=batch.feature_schema,
            events=batch.events,
        ),
    }


def _validate_feature_schema(schema: PredictionFeatureSchema) -> None:
    if not isinstance(schema, PredictionFeatureSchema):
        raise TypeError("feature_schema must be a PredictionFeatureSchema.")
    if schema.model_name != "win_prob" or schema.model_type != "logistic":
        raise ValueError(
            "Logistic explanation evidence requires a win_prob/logistic feature schema."
        )
    if schema.task != "classification":
        raise ValueError("Logistic explanation evidence requires a classification feature schema.")
    if not schema.ordered_columns:
        raise ValueError("feature_schema ordered_columns must not be empty.")
    expected_id = prediction_feature_schema_id(
        model_name=schema.model_name,
        model_type=schema.model_type,
        task=schema.task,
        modeling_schema_version=schema.modeling_schema_version,
        epa_window=schema.epa_window,
        feature_set_name=schema.feature_set_name,
        ordered_columns=schema.ordered_columns,
    )
    if schema.schema_id != expected_id:
        raise ValueError("feature_schema schema_id does not match canonical schema content.")


def _validate_events(
    events: tuple[LogisticExplanationEvent, ...],
    *,
    feature_schema: PredictionFeatureSchema,
) -> None:
    if not events:
        raise ValueError("events must not be empty.")
    keys = tuple((event.game_id, event.event_id) for event in events)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("events must be ordered by unique game_id and event_id.")
    event_ids = [key[1] for key in keys]
    game_ids = [key[0] for key in keys]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("events contains duplicate event IDs.")
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("events contains duplicate game IDs.")

    for event in events:
        _validate_event(event, feature_schema=feature_schema)


def _validate_event(
    event: LogisticExplanationEvent,
    *,
    feature_schema: PredictionFeatureSchema,
) -> None:
    if not isinstance(event, LogisticExplanationEvent):
        raise TypeError("events must contain LogisticExplanationEvent values.")
    _text(event.event_id, "event_id")
    _text(event.game_id, "game_id")
    _finite(event.intercept, "intercept")
    _finite(event.reconstructed_log_odds, "reconstructed_log_odds")
    _probability(event.reconstructed_probability, "reconstructed_probability")
    _probability(event.raw_estimator_output, "raw_estimator_output")
    if not isinstance(event.tolerance, int | float) or isinstance(event.tolerance, bool):
        raise ValueError("tolerance must be numeric.")
    if event.tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    contribution_names = tuple(item.feature_name for item in event.contributions)
    if contribution_names != feature_schema.ordered_columns:
        raise ValueError("event contributions must exactly match feature_schema ordered_columns.")

    running_log_odds = event.intercept
    for item in event.contributions:
        if not isinstance(item, LogisticFeatureContribution):
            raise TypeError("contributions must contain LogisticFeatureContribution values.")
        _text(item.feature_name, "feature_name")
        _finite(item.transformed_value, "transformed_value")
        _finite(item.coefficient, "coefficient")
        _finite(item.contribution, "contribution")
        expected_contribution = item.coefficient * item.transformed_value
        if not math.isclose(
            item.contribution,
            expected_contribution,
            rel_tol=0.0,
            abs_tol=_INTERNAL_CONSISTENCY_TOLERANCE,
        ):
            raise ValueError(
                f"contribution for {item.feature_name!r} does not equal "
                "coefficient * transformed_value."
            )
        running_log_odds += item.contribution

    if not math.isclose(
        running_log_odds,
        event.reconstructed_log_odds,
        rel_tol=0.0,
        abs_tol=_INTERNAL_CONSISTENCY_TOLERANCE,
    ):
        raise ValueError(
            "reconstructed_log_odds does not equal intercept plus summed contributions."
        )

    expected_probability = sigmoid(event.reconstructed_log_odds)
    if not math.isclose(
        expected_probability,
        event.reconstructed_probability,
        rel_tol=0.0,
        abs_tol=_INTERNAL_CONSISTENCY_TOLERANCE,
    ):
        raise ValueError(
            "reconstructed_probability does not equal sigmoid(reconstructed_log_odds)."
        )

    if abs(event.reconstructed_probability - event.raw_estimator_output) > event.tolerance:
        raise ValueError(
            "reconstructed_probability does not reconcile with raw_estimator_output "
            f"within tolerance: event_id={event.event_id!r}, "
            f"reconstructed={event.reconstructed_probability!r}, "
            f"raw={event.raw_estimator_output!r}, tolerance={event.tolerance!r}."
        )


def sigmoid(x: float) -> float:
    """Return the logistic sigmoid of one log-odds value, overflow-safe."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _identity_payload(
    *,
    run_id: str,
    evidence_id: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    events: tuple[LogisticExplanationEvent, ...],
) -> dict[str, object]:
    return {
        "run_id": _text(run_id, "run_id"),
        "evidence_id": _digest(evidence_id, "evidence_id"),
        "model_content_digest": _digest(model_content_digest, "model_content_digest"),
        "scaler_content_digest": _digest(scaler_content_digest, "scaler_content_digest"),
        "feature_schema": _feature_schema_payload(feature_schema),
        "events": [_event_payload(event) for event in events],
    }


def _feature_schema_payload(value: PredictionFeatureSchema) -> dict[str, object]:
    _validate_feature_schema(value)
    return {
        "schema_id": value.schema_id,
        "model_name": value.model_name,
        "model_type": value.model_type,
        "task": value.task,
        "modeling_schema_version": value.modeling_schema_version,
        "epa_window": value.epa_window,
        "feature_set_name": value.feature_set_name,
        "ordered_columns": list(value.ordered_columns),
    }


def _event_payload(value: LogisticExplanationEvent) -> dict[str, object]:
    return {
        "event_id": _text(value.event_id, "event_id"),
        "game_id": _text(value.game_id, "game_id"),
        "intercept": _finite(value.intercept, "intercept"),
        "contributions": [_contribution_payload(item) for item in value.contributions],
        "reconstructed_log_odds": _finite(value.reconstructed_log_odds, "reconstructed_log_odds"),
        "reconstructed_probability": _probability(
            value.reconstructed_probability, "reconstructed_probability"
        ),
        "raw_estimator_output": _probability(value.raw_estimator_output, "raw_estimator_output"),
        "tolerance": value.tolerance,
    }


def _contribution_payload(value: LogisticFeatureContribution) -> dict[str, object]:
    return {
        "feature_name": _text(value.feature_name, "feature_name"),
        "transformed_value": _finite(value.transformed_value, "transformed_value"),
        "coefficient": _finite(value.coefficient, "coefficient"),
        "contribution": _finite(value.contribution, "contribution"),
    }


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC.")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError(f"{label} must use UTC.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_PATTERN_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _probability(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0 or result > 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result
