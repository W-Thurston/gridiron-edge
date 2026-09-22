# src/gridiron_edge/evaluation/prediction_input_evidence.py
"""Immutable prediction-input evidence for selected weekly forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Final

from pandas import DataFrame

from gridiron_edge.evaluation.forecast_contracts import (
    INPUT_EVIDENCE_FORECAST_ROLES,
    ForecastRole,
)
from gridiron_edge.evaluation.forecast_store import validate_forecast_events

PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION: Final[int] = 1
_PROBABILITY_TOLERANCE: Final[float] = 1e-12
_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SEASON_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{4})$")
_ALLOWED_TASKS: Final[frozenset[str]] = frozenset({"classification", "regression"})
type JsonScalar = bool | int | float | str | None
type FinalOutputs = tuple[tuple[str, JsonScalar], ...]


class PredictionExecutionKind(StrEnum):
    """Computational mechanism used to produce one forecast family.

    This is distinct from PredictionModelSource, which records how a model
    was selected by prediction policy.
    """

    PERSISTED_ESTIMATOR = "persisted_estimator"
    ELO_FORMULA = "elo_formula"


class PredictionArtifactKind(StrEnum):
    """Supported immutable binary artifact kind."""

    MODEL = "model"
    MODEL_METADATA = "model_metadata"
    SCALER = "scaler"
    EXTERNAL_CALIBRATOR = "external_calibrator"
    ELO_LINEAGE = "elo_lineage"
    ELO_STATE = "elo_state"


class PredictionArtifactState(StrEnum):
    """Presence state of one optional or required binary artifact."""

    PRESENT = "present"
    ABSENT = "absent"


class PredictionSourceState(StrEnum):
    """Presence state of one canonical source artifact."""

    PRESENT = "present"
    ABSENT = "absent"


class CalibrationResolutionSource(StrEnum):
    """Source used to resolve one post-processing parameter."""

    PERSISTED_REGISTRY = "persisted_registry"
    MODEL_FALLBACK = "model_fallback"
    GLOBAL_DEFAULT = "global_default"
    NOT_USED = "not_used"


@dataclass(frozen=True, slots=True)
class SourceRevision:
    """Exact committed source revision used for prediction execution."""

    commit: str
    tracked_worktree_clean: bool


@dataclass(frozen=True, slots=True)
class SourceArtifactReference:
    """Exact identity or explicit absence of one canonical source artifact."""

    relative_path: str
    state: PredictionSourceState
    content_digest: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class BinaryArtifactReference:
    """Exact immutable snapshot identity or explicit optional absence."""

    kind: PredictionArtifactKind
    source_relative_path: str
    state: PredictionArtifactState
    content_digest: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class PredictionFeatureSchema:
    """Ordered estimator feature contract for one statistical model."""

    schema_id: str
    model_name: str
    model_type: str
    task: str
    modeling_schema_version: int
    feature_set_name: str
    ordered_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PredictionPostProcessingEvidence:
    """Exact Win post-processing inputs and their resolution provenance."""

    registry_reference: SourceArtifactReference
    registry_entry_updated_at: datetime | None
    sigma: float | None
    sigma_source: CalibrationResolutionSource
    margin_std: float | None
    margin_std_source: CalibrationResolutionSource
    external_calibrator_state: PredictionArtifactState
    embedded_estimator_calibration: bool


@dataclass(frozen=True, slots=True)
class StatisticalPredictionEventEvidence:
    """Exact event-level computation performed by one persisted estimator."""

    event_id: str
    game_id: str
    raw_feature_values: tuple[float, ...]
    transformed_feature_values: tuple[float, ...]
    raw_estimator_output: float
    post_estimator_output: float
    final_outputs: FinalOutputs


@dataclass(frozen=True, slots=True)
class EloPredictionEventEvidence:
    """Exact event-level computation performed by the Elo formula."""

    event_id: str
    game_id: str
    season: str
    week: int
    away_team: str
    home_team: str
    away_elo: float
    home_elo: float
    formula_id: str
    divisor: float
    away_win_probability: float
    home_win_probability: float
    final_outputs: FinalOutputs


@dataclass(frozen=True, slots=True)
class PredictionInputEvidence:
    """Immutable evidence for one run-scoped selected prediction family."""

    schema_version: int
    evidence_id: str
    run_id: str
    season: str
    week: int
    generated_at: datetime
    model_name: str
    model_type: str
    execution_kind: PredictionExecutionKind
    source_revision: SourceRevision
    source_artifacts: tuple[SourceArtifactReference, ...]
    binary_artifacts: tuple[BinaryArtifactReference, ...]
    feature_schema: PredictionFeatureSchema | None
    post_processing: PredictionPostProcessingEvidence | None
    statistical_events: tuple[StatisticalPredictionEventEvidence, ...]
    elo_events: tuple[EloPredictionEventEvidence, ...]


def prediction_feature_schema_id(
    *,
    model_name: str,
    model_type: str,
    task: str,
    modeling_schema_version: int,
    feature_set_name: str,
    ordered_columns: tuple[str, ...],
) -> str:
    """Return the SHA-256 identity of one ordered feature contract."""
    payload = _feature_schema_payload(
        model_name=model_name,
        model_type=model_type,
        task=task,
        modeling_schema_version=modeling_schema_version,
        feature_set_name=feature_set_name,
        ordered_columns=ordered_columns,
    )
    return _canonical_digest(payload)


def create_prediction_feature_schema(
    *,
    model_name: str,
    model_type: str,
    task: str,
    modeling_schema_version: int,
    feature_set_name: str,
    ordered_columns: tuple[str, ...],
) -> PredictionFeatureSchema:
    """Create and validate one ordered estimator feature contract."""
    schema = PredictionFeatureSchema(
        schema_id=prediction_feature_schema_id(
            model_name=model_name,
            model_type=model_type,
            task=task,
            modeling_schema_version=modeling_schema_version,
            feature_set_name=feature_set_name,
            ordered_columns=ordered_columns,
        ),
        model_name=model_name,
        model_type=model_type,
        task=task,
        modeling_schema_version=modeling_schema_version,
        feature_set_name=feature_set_name,
        ordered_columns=ordered_columns,
    )
    _validate_feature_schema(schema)
    return schema


def prediction_input_evidence_id(
    *,
    run_id: str,
    season: str,
    week: int,
    generated_at: datetime,
    model_name: str,
    model_type: str,
    execution_kind: PredictionExecutionKind,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
    feature_schema: PredictionFeatureSchema | None,
    post_processing: PredictionPostProcessingEvidence | None,
    statistical_events: tuple[StatisticalPredictionEventEvidence, ...],
    elo_events: tuple[EloPredictionEventEvidence, ...],
) -> str:
    """Return the SHA-256 identity of the complete canonical evidence payload."""
    payload = _identity_payload(
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name=model_name,
        model_type=model_type,
        execution_kind=execution_kind,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=feature_schema,
        post_processing=post_processing,
        statistical_events=statistical_events,
        elo_events=elo_events,
    )
    return _canonical_digest(payload)


def create_statistical_prediction_input_evidence(
    *,
    run_id: str,
    season: str,
    week: int,
    generated_at: datetime,
    model_name: str,
    model_type: str,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
    feature_schema: PredictionFeatureSchema,
    post_processing: PredictionPostProcessingEvidence | None,
    events: tuple[StatisticalPredictionEventEvidence, ...],
) -> PredictionInputEvidence:
    """Create complete schema-1 evidence for one persisted estimator family."""
    evidence_id = prediction_input_evidence_id(
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name=model_name,
        model_type=model_type,
        execution_kind=PredictionExecutionKind.PERSISTED_ESTIMATOR,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=feature_schema,
        post_processing=post_processing,
        statistical_events=events,
        elo_events=(),
    )
    evidence = PredictionInputEvidence(
        schema_version=PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION,
        evidence_id=evidence_id,
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name=model_name,
        model_type=model_type,
        execution_kind=PredictionExecutionKind.PERSISTED_ESTIMATOR,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=feature_schema,
        post_processing=post_processing,
        statistical_events=events,
        elo_events=(),
    )
    validate_prediction_input_evidence(evidence)
    return evidence


def create_elo_prediction_input_evidence(
    *,
    run_id: str,
    season: str,
    week: int,
    generated_at: datetime,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
    events: tuple[EloPredictionEventEvidence, ...],
) -> PredictionInputEvidence:
    """Create complete schema-1 evidence for one Elo formula family."""
    evidence_id = prediction_input_evidence_id(
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name="win_prob",
        model_type="elo",
        execution_kind=PredictionExecutionKind.ELO_FORMULA,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=None,
        post_processing=None,
        statistical_events=(),
        elo_events=events,
    )
    evidence = PredictionInputEvidence(
        schema_version=PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION,
        evidence_id=evidence_id,
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name="win_prob",
        model_type="elo",
        execution_kind=PredictionExecutionKind.ELO_FORMULA,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=None,
        post_processing=None,
        statistical_events=(),
        elo_events=events,
    )
    validate_prediction_input_evidence(evidence)
    return evidence


def validate_prediction_input_evidence(evidence: PredictionInputEvidence) -> None:
    """Validate one complete family evidence contract and embedded identity."""
    if evidence.schema_version != PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prediction-input evidence schema_version: {evidence.schema_version}."
        )
    _digest(evidence.evidence_id, "evidence_id")
    _text(evidence.run_id, "run_id")
    _season(evidence.season)
    _positive_integer(evidence.week, "week")
    _utc(evidence.generated_at, "generated_at")
    _text(evidence.model_name, "model_name")
    _text(evidence.model_type, "model_type")
    _require_enum(evidence.execution_kind, PredictionExecutionKind, "execution_kind")
    _validate_source_revision(evidence.source_revision)
    _validate_source_artifacts(evidence.source_artifacts)
    _validate_binary_artifacts(evidence.binary_artifacts)

    if evidence.execution_kind is PredictionExecutionKind.PERSISTED_ESTIMATOR:
        _validate_persisted_estimator_evidence(evidence)
    elif evidence.execution_kind is PredictionExecutionKind.ELO_FORMULA:
        _validate_elo_formula_evidence(evidence)
    else:  # Defensive for malformed runtime construction.
        raise ValueError("Unsupported prediction execution kind.")

    expected_id = prediction_input_evidence_id(
        run_id=evidence.run_id,
        season=evidence.season,
        week=evidence.week,
        generated_at=evidence.generated_at,
        model_name=evidence.model_name,
        model_type=evidence.model_type,
        execution_kind=evidence.execution_kind,
        source_revision=evidence.source_revision,
        source_artifacts=evidence.source_artifacts,
        binary_artifacts=evidence.binary_artifacts,
        feature_schema=evidence.feature_schema,
        post_processing=evidence.post_processing,
        statistical_events=evidence.statistical_events,
        elo_events=evidence.elo_events,
    )
    if evidence.evidence_id != expected_id:
        raise ValueError("evidence_id does not match canonical evidence content.")


def authenticate_prediction_input_evidence(
    evidence: PredictionInputEvidence,
    *,
    forecast_events: DataFrame,
) -> None:
    """Authenticate exact bidirectional coverage of one selected weekly family."""
    validate_prediction_input_evidence(evidence)
    events = validate_forecast_events(forecast_events)
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("Forecast events contain duplicate event IDs.")

    scoped = events.loc[
        (events["run_id"].astype(str) == evidence.run_id)
        & (events["season"].astype(str) == evidence.season)
        & (events["week"].astype(int) == evidence.week)
        & (events["model_name"].astype(str) == evidence.model_name)
        & (events["model_type"].astype(str) == evidence.model_type),
        :,
    ].copy()

    evidence_events = _all_event_evidence(evidence)
    expected_ids = tuple(sorted(event.event_id for event in evidence_events))
    actual_ids = tuple(sorted(scoped["event_id"].astype(str).tolist()))

    if not actual_ids:
        raise ValueError(
            "Forecast-event coverage does not match prediction-input evidence; "
            f"missing={list(expected_ids)}, unexpected=[]."
        )

    roles = tuple(sorted(scoped["role"].dropna().astype(str).unique().tolist()))
    if len(roles) != 1:
        raise ValueError("Prediction-input evidence forecast events must use one role.")

    try:
        role = ForecastRole(roles[0])
    except ValueError as exc:
        raise ValueError(
            "Prediction-input evidence contains an unsupported forecast role."
        ) from exc

    if role not in INPUT_EVIDENCE_FORECAST_ROLES:
        raise ValueError("Prediction-input evidence requires a live or development forecast role.")

    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(
            "Forecast-event coverage does not match prediction-input evidence; "
            f"missing={missing}, unexpected={unexpected}."
        )

    indexed = scoped.set_index("event_id", drop=False)
    for event in evidence_events:
        row = indexed.loc[event.event_id]
        if str(row["game_id"]) != event.game_id:
            raise ValueError("Forecast event game_id does not match evidence.")
        for output_name, expected_value in event.final_outputs:
            if output_name not in scoped.columns:
                raise ValueError(
                    f"Forecast events are missing evidenced output column: {output_name}"
                )
            actual_value = row[output_name]
            if not _scalar_equal(actual_value, expected_value):
                raise ValueError(
                    "Forecast event output does not match prediction-input evidence: "
                    f"event_id={event.event_id!r}, output={output_name!r}."
                )


def require_same_source_artifacts(
    before: tuple[SourceArtifactReference, ...],
    after: tuple[SourceArtifactReference, ...],
) -> None:
    """Reject any source presence, digest, size, or path drift."""
    _validate_source_artifacts(before)
    _validate_source_artifacts(after)
    if before != after:
        raise ValueError("Prediction source artifacts changed during execution.")


def prediction_input_evidence_payload(
    evidence: PredictionInputEvidence,
) -> dict[str, object]:
    """Return the complete stable JSON-compatible evidence representation."""
    validate_prediction_input_evidence(evidence)
    return {
        "schema_version": evidence.schema_version,
        "evidence_id": evidence.evidence_id,
        **_identity_payload(
            run_id=evidence.run_id,
            season=evidence.season,
            week=evidence.week,
            generated_at=evidence.generated_at,
            model_name=evidence.model_name,
            model_type=evidence.model_type,
            execution_kind=evidence.execution_kind,
            source_revision=evidence.source_revision,
            source_artifacts=evidence.source_artifacts,
            binary_artifacts=evidence.binary_artifacts,
            feature_schema=evidence.feature_schema,
            post_processing=evidence.post_processing,
            statistical_events=evidence.statistical_events,
            elo_events=evidence.elo_events,
        ),
    }


def _validate_persisted_estimator_evidence(evidence: PredictionInputEvidence) -> None:
    if evidence.feature_schema is None:
        raise ValueError("Persisted-estimator evidence requires feature_schema.")
    if not evidence.statistical_events or evidence.elo_events:
        raise ValueError("Persisted-estimator evidence requires only statistical event evidence.")
    _validate_feature_schema(evidence.feature_schema)
    if evidence.feature_schema.model_name != evidence.model_name:
        raise ValueError("Feature schema model_name does not match family evidence.")
    if evidence.feature_schema.model_type != evidence.model_type:
        raise ValueError("Feature schema model_type does not match family evidence.")

    by_kind = {reference.kind: reference for reference in evidence.binary_artifacts}
    required_kinds = {
        PredictionArtifactKind.MODEL,
        PredictionArtifactKind.MODEL_METADATA,
        PredictionArtifactKind.SCALER,
        PredictionArtifactKind.EXTERNAL_CALIBRATOR,
    }
    if set(by_kind) != required_kinds:
        raise ValueError(
            "Persisted-estimator evidence requires exactly model, model_metadata, "
            "scaler, and external_calibrator artifact records."
        )
    for kind in (PredictionArtifactKind.MODEL, PredictionArtifactKind.MODEL_METADATA):
        if by_kind[kind].state is not PredictionArtifactState.PRESENT:
            raise ValueError(f"Persisted-estimator {kind.value} artifact must be present.")

    if any(
        reference.kind in {PredictionArtifactKind.ELO_LINEAGE, PredictionArtifactKind.ELO_STATE}
        for reference in evidence.binary_artifacts
    ):
        raise ValueError("Persisted-estimator evidence cannot contain Elo artifacts.")

    _validate_statistical_events(
        evidence.statistical_events,
        expected_feature_count=len(evidence.feature_schema.ordered_columns),
        scaler_state=by_kind[PredictionArtifactKind.SCALER].state,
    )
    if evidence.feature_schema.task == "classification":
        if evidence.model_name != "win_prob":
            raise ValueError("Classification evidence must use model_name win_prob.")
        if evidence.post_processing is None:
            raise ValueError("Win classification evidence requires post_processing.")
        _validate_post_processing(evidence.post_processing, evidence.binary_artifacts)
    elif evidence.post_processing is not None:
        raise ValueError("Regression evidence must not contain Win post_processing.")


def _validate_elo_formula_evidence(evidence: PredictionInputEvidence) -> None:
    if evidence.model_name != "win_prob" or evidence.model_type != "elo":
        raise ValueError("Elo formula evidence requires model identity win_prob/elo.")
    if evidence.feature_schema is not None:
        raise ValueError("Elo formula evidence must not contain feature_schema.")
    if evidence.post_processing is not None:
        raise ValueError("Elo formula evidence must not contain post_processing.")
    if evidence.statistical_events or not evidence.elo_events:
        raise ValueError("Elo formula evidence requires only Elo event evidence.")

    by_kind = {reference.kind: reference for reference in evidence.binary_artifacts}
    if set(by_kind) != {
        PredictionArtifactKind.ELO_LINEAGE,
        PredictionArtifactKind.ELO_STATE,
    }:
        raise ValueError(
            "Elo formula evidence requires exactly elo_lineage and elo_state artifacts."
        )
    if any(
        reference.state is not PredictionArtifactState.PRESENT for reference in by_kind.values()
    ):
        raise ValueError("Elo lineage and Elo state artifacts must be present.")

    _validate_elo_events(
        evidence.elo_events,
        season=evidence.season,
        week=evidence.week,
    )


def _validate_source_revision(revision: SourceRevision) -> None:
    if not isinstance(revision, SourceRevision):
        raise TypeError("source_revision must be a SourceRevision.")
    if _COMMIT_PATTERN.fullmatch(revision.commit) is None:
        raise ValueError("source revision commit must be a lowercase 40-character Git SHA.")
    if revision.tracked_worktree_clean is not True:
        raise ValueError("source revision requires a clean tracked worktree.")


def _validate_source_artifacts(references: tuple[SourceArtifactReference, ...]) -> None:
    if not references:
        raise ValueError("source_artifacts must not be empty.")
    keys: list[str] = []
    for reference in references:
        if not isinstance(reference, SourceArtifactReference):
            raise TypeError("source_artifacts must contain SourceArtifactReference values.")
        _safe_relative_path(reference.relative_path, "source artifact path")
        _require_enum(reference.state, PredictionSourceState, "source artifact state")
        _validate_presence(
            state=reference.state.value,
            digest=reference.content_digest,
            size_bytes=reference.size_bytes,
            label="source artifact",
        )
        keys.append(reference.relative_path)
    if tuple(keys) != tuple(sorted(set(keys))):
        raise ValueError("source_artifacts must be ordered by unique relative path.")


def _validate_binary_artifacts(references: tuple[BinaryArtifactReference, ...]) -> None:
    if not references:
        raise ValueError("binary_artifacts must not be empty.")
    kinds: list[str] = []
    for reference in references:
        if not isinstance(reference, BinaryArtifactReference):
            raise TypeError("binary_artifacts must contain BinaryArtifactReference values.")
        _require_enum(reference.kind, PredictionArtifactKind, "binary artifact kind")
        _safe_relative_path(reference.source_relative_path, "binary artifact source path")
        _require_enum(reference.state, PredictionArtifactState, "binary artifact state")
        _validate_presence(
            state=reference.state.value,
            digest=reference.content_digest,
            size_bytes=reference.size_bytes,
            label="binary artifact",
        )
        kinds.append(reference.kind.value)
    if tuple(kinds) != tuple(sorted(set(kinds))):
        raise ValueError("binary_artifacts must be ordered by unique artifact kind.")


def _validate_feature_schema(schema: PredictionFeatureSchema) -> None:
    if not isinstance(schema, PredictionFeatureSchema):
        raise TypeError("feature_schema must be a PredictionFeatureSchema.")
    _digest(schema.schema_id, "feature schema_id")
    _text(schema.model_name, "feature schema model_name")
    _text(schema.model_type, "feature schema model_type")
    if schema.task not in _ALLOWED_TASKS:
        raise ValueError("feature schema task must be classification or regression.")
    _positive_integer(schema.modeling_schema_version, "modeling_schema_version")
    _text(schema.feature_set_name, "feature_set_name")
    if not schema.ordered_columns:
        raise ValueError("ordered feature columns must not be empty.")
    normalized = tuple(_text(value, "feature column") for value in schema.ordered_columns)
    if len(normalized) != len(set(normalized)):
        raise ValueError("ordered feature columns must not contain duplicates.")
    expected_id = prediction_feature_schema_id(
        model_name=schema.model_name,
        model_type=schema.model_type,
        task=schema.task,
        modeling_schema_version=schema.modeling_schema_version,
        feature_set_name=schema.feature_set_name,
        ordered_columns=schema.ordered_columns,
    )
    if schema.schema_id != expected_id:
        raise ValueError("feature schema_id does not match canonical schema content.")


def _validate_post_processing(
    evidence: PredictionPostProcessingEvidence,
    artifacts: tuple[BinaryArtifactReference, ...],
) -> None:
    if not isinstance(evidence, PredictionPostProcessingEvidence):
        raise TypeError("post_processing must be PredictionPostProcessingEvidence.")
    _validate_source_artifacts((evidence.registry_reference,))
    if evidence.registry_entry_updated_at is not None:
        _utc(evidence.registry_entry_updated_at, "registry_entry_updated_at")
    _validate_resolved_value(
        value=evidence.sigma,
        source=evidence.sigma_source,
        label="sigma",
        registry=evidence.registry_reference,
    )
    _validate_resolved_value(
        value=evidence.margin_std,
        source=evidence.margin_std_source,
        label="margin_std",
        registry=evidence.registry_reference,
    )
    _require_enum(
        evidence.external_calibrator_state,
        PredictionArtifactState,
        "external_calibrator_state",
    )
    if not isinstance(evidence.embedded_estimator_calibration, bool):
        raise ValueError("embedded_estimator_calibration must be a boolean.")
    external = next(
        reference
        for reference in artifacts
        if reference.kind is PredictionArtifactKind.EXTERNAL_CALIBRATOR
    )
    if external.state is not evidence.external_calibrator_state:
        raise ValueError(
            "External calibrator artifact state does not match post-processing evidence."
        )


def _validate_resolved_value(
    *,
    value: float | None,
    source: CalibrationResolutionSource,
    label: str,
    registry: SourceArtifactReference,
) -> None:
    _require_enum(source, CalibrationResolutionSource, f"{label}_source")
    if source is CalibrationResolutionSource.NOT_USED:
        if value is not None:
            raise ValueError(f"{label} must be null when its source is not_used.")
        return
    if value is None or not _finite(value, label) or value <= 0:
        raise ValueError(f"{label} must be a positive finite value when used.")
    if source is CalibrationResolutionSource.PERSISTED_REGISTRY and (
        registry.state is not PredictionSourceState.PRESENT
    ):
        raise ValueError(f"{label} persisted_registry source requires a present registry.")


def _validate_statistical_events(
    events: tuple[StatisticalPredictionEventEvidence, ...],
    *,
    expected_feature_count: int,
    scaler_state: PredictionArtifactState,
) -> None:
    _validate_event_order(events)
    for event in events:
        if not isinstance(event, StatisticalPredictionEventEvidence):
            raise TypeError("statistical_events contain an unsupported value.")
        _text(event.event_id, "event_id")
        _text(event.game_id, "game_id")
        if len(event.raw_feature_values) != expected_feature_count:
            raise ValueError("Raw feature vector length does not match feature schema.")
        if len(event.transformed_feature_values) != expected_feature_count:
            raise ValueError("Transformed feature vector length does not match feature schema.")
        for value in event.raw_feature_values:
            _finite(value, "raw feature value")
        for value in event.transformed_feature_values:
            _finite(value, "transformed feature value")
        _finite(event.raw_estimator_output, "raw_estimator_output")
        _finite(event.post_estimator_output, "post_estimator_output")
        _validate_final_outputs(event.final_outputs)
        if (
            scaler_state is PredictionArtifactState.ABSENT
            and event.raw_feature_values != event.transformed_feature_values
        ):
            raise ValueError("Raw and transformed feature values must match when scaler is absent.")


def _validate_elo_events(
    events: tuple[EloPredictionEventEvidence, ...],
    *,
    season: str,
    week: int,
) -> None:
    _validate_event_order(events)
    formula_keys: set[tuple[str, float]] = set()
    for event in events:
        if not isinstance(event, EloPredictionEventEvidence):
            raise TypeError("elo_events contain an unsupported value.")
        _text(event.event_id, "event_id")
        _text(event.game_id, "game_id")
        if _season(event.season) != season or _positive_integer(event.week, "week") != week:
            raise ValueError("Elo event scope does not match family evidence.")
        away_team = _text(event.away_team, "away_team")
        home_team = _text(event.home_team, "home_team")
        if away_team == home_team:
            raise ValueError("Elo event Away and Home teams must differ.")
        _finite(event.away_elo, "away_elo")
        _finite(event.home_elo, "home_elo")
        formula_id = _text(event.formula_id, "formula_id")
        divisor = _finite(event.divisor, "divisor")
        if divisor <= 0:
            raise ValueError("Elo divisor must be positive.")
        away_probability = _probability(event.away_win_probability, "away_win_probability")
        home_probability = _probability(event.home_win_probability, "home_win_probability")
        if not math.isclose(
            away_probability + home_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("Elo Away and Home probabilities must be complementary.")
        _validate_final_outputs(event.final_outputs)
        formula_keys.add((formula_id, divisor))
    if len(formula_keys) != 1:
        raise ValueError("Elo family evidence requires one formula identity and divisor.")


def _validate_event_order(events: tuple[object, ...]) -> None:
    if not events:
        raise ValueError("event evidence must not be empty.")
    keys = tuple(
        (getattr(event, "game_id", None), getattr(event, "event_id", None)) for event in events
    )
    if keys != tuple(sorted(set(keys))):
        raise ValueError("event evidence must be ordered by unique game_id and event_id.")
    event_ids = [key[1] for key in keys]
    game_ids = [key[0] for key in keys]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event evidence contains duplicate event IDs.")
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("event evidence contains duplicate game IDs.")


def _validate_final_outputs(outputs: FinalOutputs) -> None:
    if not outputs:
        raise ValueError("final_outputs must not be empty.")
    names: list[str] = []
    for name, value in outputs:
        names.append(_text(name, "final output name"))
        _json_scalar(value, f"final output {name}")
    if tuple(names) != tuple(sorted(set(names))):
        raise ValueError("final_outputs must be ordered by unique output name.")


def _identity_payload(
    *,
    run_id: str,
    season: str,
    week: int,
    generated_at: datetime,
    model_name: str,
    model_type: str,
    execution_kind: PredictionExecutionKind,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
    feature_schema: PredictionFeatureSchema | None,
    post_processing: PredictionPostProcessingEvidence | None,
    statistical_events: tuple[StatisticalPredictionEventEvidence, ...],
    elo_events: tuple[EloPredictionEventEvidence, ...],
) -> dict[str, object]:
    return {
        "run_id": _text(run_id, "run_id"),
        "season": _season(season),
        "week": _positive_integer(week, "week"),
        "generated_at": _utc(generated_at, "generated_at").isoformat(),
        "model_name": _text(model_name, "model_name"),
        "model_type": _text(model_type, "model_type"),
        "execution_kind": _enum_value(execution_kind, PredictionExecutionKind, "execution_kind"),
        "source_revision": _source_revision_payload(source_revision),
        "source_artifacts": [_source_artifact_payload(value) for value in source_artifacts],
        "binary_artifacts": [_binary_artifact_payload(value) for value in binary_artifacts],
        "feature_schema": None
        if feature_schema is None
        else _feature_schema_object(feature_schema),
        "post_processing": (
            None if post_processing is None else _post_processing_payload(post_processing)
        ),
        "statistical_events": [_statistical_event_payload(value) for value in statistical_events],
        "elo_events": [_elo_event_payload(value) for value in elo_events],
    }


def _feature_schema_payload(
    *,
    model_name: str,
    model_type: str,
    task: str,
    modeling_schema_version: int,
    feature_set_name: str,
    ordered_columns: tuple[str, ...],
) -> dict[str, object]:
    if task not in _ALLOWED_TASKS:
        raise ValueError("feature schema task must be classification or regression.")
    columns = tuple(_text(value, "feature column") for value in ordered_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("ordered feature columns must be nonempty and unique.")
    return {
        "model_name": _text(model_name, "model_name"),
        "model_type": _text(model_type, "model_type"),
        "task": task,
        "modeling_schema_version": _positive_integer(
            modeling_schema_version, "modeling_schema_version"
        ),
        "feature_set_name": _text(feature_set_name, "feature_set_name"),
        "ordered_columns": list(columns),
    }


def _source_revision_payload(value: SourceRevision) -> dict[str, object]:
    _validate_source_revision(value)
    return {
        "commit": value.commit,
        "tracked_worktree_clean": value.tracked_worktree_clean,
    }


def _source_artifact_payload(value: SourceArtifactReference) -> dict[str, object]:
    _validate_source_artifacts((value,))
    return {
        "relative_path": value.relative_path,
        "state": value.state.value,
        "content_digest": value.content_digest,
        "size_bytes": value.size_bytes,
    }


def _binary_artifact_payload(value: BinaryArtifactReference) -> dict[str, object]:
    # Validate directly because a one-item tuple is always correctly ordered.
    _require_enum(value.kind, PredictionArtifactKind, "binary artifact kind")
    _safe_relative_path(value.source_relative_path, "binary artifact source path")
    _require_enum(value.state, PredictionArtifactState, "binary artifact state")
    _validate_presence(
        state=value.state.value,
        digest=value.content_digest,
        size_bytes=value.size_bytes,
        label="binary artifact",
    )
    return {
        "kind": value.kind.value,
        "source_relative_path": value.source_relative_path,
        "state": value.state.value,
        "content_digest": value.content_digest,
        "size_bytes": value.size_bytes,
    }


def _feature_schema_object(value: PredictionFeatureSchema) -> dict[str, object]:
    _validate_feature_schema(value)
    return {
        "schema_id": value.schema_id,
        **_feature_schema_payload(
            model_name=value.model_name,
            model_type=value.model_type,
            task=value.task,
            modeling_schema_version=value.modeling_schema_version,
            feature_set_name=value.feature_set_name,
            ordered_columns=value.ordered_columns,
        ),
    }


def _post_processing_payload(value: PredictionPostProcessingEvidence) -> dict[str, object]:
    return {
        "registry_reference": _source_artifact_payload(value.registry_reference),
        "registry_entry_updated_at": (
            None
            if value.registry_entry_updated_at is None
            else _utc(value.registry_entry_updated_at, "registry_entry_updated_at").isoformat()
        ),
        "sigma": value.sigma,
        "sigma_source": _enum_value(
            value.sigma_source, CalibrationResolutionSource, "sigma_source"
        ),
        "margin_std": value.margin_std,
        "margin_std_source": _enum_value(
            value.margin_std_source,
            CalibrationResolutionSource,
            "margin_std_source",
        ),
        "external_calibrator_state": _enum_value(
            value.external_calibrator_state,
            PredictionArtifactState,
            "external_calibrator_state",
        ),
        "embedded_estimator_calibration": value.embedded_estimator_calibration,
    }


def _statistical_event_payload(
    value: StatisticalPredictionEventEvidence,
) -> dict[str, object]:
    _text(value.event_id, "event_id")
    _text(value.game_id, "game_id")

    raw_feature_values = [_finite(item, "raw feature value") for item in value.raw_feature_values]
    transformed_feature_values = [
        _finite(item, "transformed feature value") for item in value.transformed_feature_values
    ]
    raw_estimator_output = _finite(
        value.raw_estimator_output,
        "raw_estimator_output",
    )
    post_estimator_output = _finite(
        value.post_estimator_output,
        "post_estimator_output",
    )
    _validate_final_outputs(value.final_outputs)

    return {
        "event_id": value.event_id,
        "game_id": value.game_id,
        "raw_feature_values": raw_feature_values,
        "transformed_feature_values": transformed_feature_values,
        "raw_estimator_output": raw_estimator_output,
        "post_estimator_output": post_estimator_output,
        "final_outputs": [[name, scalar] for name, scalar in value.final_outputs],
    }


def _elo_event_payload(
    value: EloPredictionEventEvidence,
) -> dict[str, object]:
    event_id = _text(value.event_id, "event_id")
    game_id = _text(value.game_id, "game_id")
    season = _season(value.season)
    week = _positive_integer(value.week, "week")
    away_team = _text(value.away_team, "away_team")
    home_team = _text(value.home_team, "home_team")
    away_elo = _finite(value.away_elo, "away_elo")
    home_elo = _finite(value.home_elo, "home_elo")
    formula_id = _text(value.formula_id, "formula_id")
    divisor = _finite(value.divisor, "divisor")
    away_win_probability = _probability(
        value.away_win_probability,
        "away_win_probability",
    )
    home_win_probability = _probability(
        value.home_win_probability,
        "home_win_probability",
    )
    _validate_final_outputs(value.final_outputs)

    return {
        "event_id": event_id,
        "game_id": game_id,
        "season": season,
        "week": week,
        "away_team": away_team,
        "home_team": home_team,
        "away_elo": away_elo,
        "home_elo": home_elo,
        "formula_id": formula_id,
        "divisor": divisor,
        "away_win_probability": away_win_probability,
        "home_win_probability": home_win_probability,
        "final_outputs": [[name, scalar] for name, scalar in value.final_outputs],
    }


def _all_event_evidence(
    evidence: PredictionInputEvidence,
) -> tuple[StatisticalPredictionEventEvidence | EloPredictionEventEvidence, ...]:
    values: tuple[StatisticalPredictionEventEvidence | EloPredictionEventEvidence, ...]
    if evidence.execution_kind is PredictionExecutionKind.PERSISTED_ESTIMATOR:
        values = evidence.statistical_events
    else:
        values = evidence.elo_events
    return values


def _validate_presence(
    *,
    state: str,
    digest: str | None,
    size_bytes: int | None,
    label: str,
) -> None:
    if state == "present":
        if digest is None:
            raise ValueError(f"Present {label} requires content_digest.")
        _digest(digest, f"{label} content_digest")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Present {label} requires nonnegative size_bytes.")
    elif state == "absent":
        if digest is not None or size_bytes is not None:
            raise ValueError(f"Absent {label} must not contain digest or size.")
    else:
        raise ValueError(f"Unsupported {label} presence state: {state!r}.")


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _safe_relative_path(value: str, label: str) -> str:
    text = _text(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe repository-relative path.")
    return text


def _season(value: str) -> str:
    text = _text(value, "season")
    match = _SEASON_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("season must use canonical YYYY-YYYY format.")
    if int(match.group("end")) != int(match.group("start")) + 1:
        raise ValueError("season ending year must be one greater than starting year.")
    return text


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
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
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


def _json_scalar(value: object, label: str) -> JsonScalar:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value
    raise ValueError(f"{label} must be a JSON scalar.")


def _enum_value(value: object, expected: type[StrEnum], label: str) -> str:
    _require_enum(value, expected, label)
    return str(value)


def _require_enum(value: object, expected: type[StrEnum], label: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}.")


def _scalar_equal(actual: object, expected: JsonScalar) -> bool:
    if expected is None:
        return actual is None or (isinstance(actual, float) and math.isnan(actual))
    if isinstance(expected, float):
        return (
            isinstance(actual, int | float)
            and not isinstance(actual, bool)
            and float(actual) == expected
        )
    return actual == expected
