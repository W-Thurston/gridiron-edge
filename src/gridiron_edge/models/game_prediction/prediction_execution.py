# src/gridiron_edge/models/game_prediction/prediction_execution.py
"""Evidence-preserving execution results for live game prediction models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Final

from pandas import DataFrame, Series

from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionFeatureSchema,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    prediction_feature_schema_id,
)

ELO_FORMULA_ID: Final[str] = "elo_win_probability_v1"
_PROBABILITY_TOLERANCE: Final[float] = 1e-12
_ELO_STATE_PATH: Final[str] = "data/cleaned/NFL_Team_Elo.csv"
_ELO_LINEAGE_PATH: Final[str] = "data/cleaned/NFL_Team_Elo.metadata.json"


@dataclass(frozen=True, slots=True)
class StatisticalPredictionComputation:
    """Exact inputs and outputs for one persisted-estimator invocation."""

    game_id: str
    raw_feature_values: tuple[float, ...]
    transformed_feature_values: tuple[float, ...]
    raw_estimator_output: float
    post_estimator_output: float


@dataclass(frozen=True, slots=True)
class StatisticalPredictionExecution:
    """Statistical predictions and exact evidence from one execution."""

    predictions: DataFrame
    computations: tuple[StatisticalPredictionComputation, ...]
    source_revision: SourceRevision
    source_artifacts: tuple[SourceArtifactReference, ...]
    binary_artifacts: tuple[BinaryArtifactReference, ...]
    feature_schema: PredictionFeatureSchema
    post_processing: PredictionPostProcessingEvidence | None


def build_statistical_prediction_execution(
    predictions: DataFrame,
    *,
    game_ids: tuple[str, ...],
    raw_feature_values: tuple[tuple[float, ...], ...],
    transformed_feature_values: tuple[tuple[float, ...], ...],
    raw_estimator_outputs: tuple[float, ...],
    post_estimator_outputs: tuple[float, ...],
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
    feature_schema: PredictionFeatureSchema,
    post_processing: PredictionPostProcessingEvidence | None,
) -> StatisticalPredictionExecution:
    """Build and validate one evidence-preserving statistical execution."""
    _validate_revision(
        source_revision,
        label="Statistical execution",
    )
    _validate_sources(
        source_artifacts,
        label="Statistical execution",
    )
    _validate_statistical_feature_schema(feature_schema)

    artifacts = _validate_statistical_binary_artifacts(
        binary_artifacts,
        feature_schema=feature_schema,
        post_processing=post_processing,
    )

    if predictions.empty:
        raise ValueError("Statistical prediction execution must not be empty.")

    if "GAME_ID" not in predictions.columns:
        raise ValueError("Statistical predictions are missing GAME_ID.")

    prediction_ids = predictions["GAME_ID"]
    if prediction_ids.isna().any():
        raise ValueError("Statistical predictions contain null GAME_ID values.")

    normalized_prediction_ids = tuple(_text(value, "GAME_ID") for value in prediction_ids.tolist())
    if len(normalized_prediction_ids) != len(set(normalized_prediction_ids)):
        raise ValueError("Statistical predictions contain duplicate game IDs.")

    row_count = len(predictions)
    lengths = {
        len(game_ids),
        len(raw_feature_values),
        len(transformed_feature_values),
        len(raw_estimator_outputs),
        len(post_estimator_outputs),
    }
    if lengths != {row_count}:
        raise ValueError("Statistical execution inputs must align one-to-one with prediction rows.")

    feature_count = len(feature_schema.ordered_columns)
    scaler_state = artifacts[PredictionArtifactKind.SCALER].state

    computations = tuple(
        sorted(
            (
                _statistical_computation(
                    game_id=game_id,
                    raw_values=raw_values,
                    transformed_values=transformed_values,
                    raw_output=raw_output,
                    post_output=post_output,
                    feature_count=feature_count,
                    task=feature_schema.task,
                    scaler_state=scaler_state,
                )
                for (
                    game_id,
                    raw_values,
                    transformed_values,
                    raw_output,
                    post_output,
                ) in zip(
                    game_ids,
                    raw_feature_values,
                    transformed_feature_values,
                    raw_estimator_outputs,
                    post_estimator_outputs,
                    strict=True,
                )
            ),
            key=lambda value: value.game_id,
        )
    )

    computation_ids = tuple(value.game_id for value in computations)
    if len(computation_ids) != len(set(computation_ids)):
        raise ValueError("Statistical execution contains duplicate computation game IDs.")

    prediction_id_set = set(normalized_prediction_ids)
    computation_id_set = set(computation_ids)
    if computation_id_set != prediction_id_set:
        missing = sorted(prediction_id_set - computation_id_set)
        unexpected = sorted(computation_id_set - prediction_id_set)
        raise ValueError(
            "Statistical computation coverage does not match "
            "predictions; "
            f"missing={missing}, unexpected={unexpected}."
        )

    return StatisticalPredictionExecution(
        predictions=predictions.copy(deep=True),
        computations=computations,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=binary_artifacts,
        feature_schema=feature_schema,
        post_processing=post_processing,
    )


@dataclass(frozen=True, slots=True)
class EloPredictionComputation:
    """Exact inputs and outputs for one live Elo probability calculation."""

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


@dataclass(frozen=True, slots=True)
class EloPredictionExecution:
    """Live Elo predictions and exact evidence from the same computation."""

    predictions: DataFrame
    computations: tuple[EloPredictionComputation, ...]
    source_revision: SourceRevision
    source_artifacts: tuple[SourceArtifactReference, ...]
    binary_artifacts: tuple[BinaryArtifactReference, ...]


def build_elo_prediction_execution(
    predictions: DataFrame,
    *,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
    divisor: float,
    formula_id: str = ELO_FORMULA_ID,
) -> EloPredictionExecution:
    """Build and validate one evidence-preserving live Elo execution result."""
    _validate_revision(
        source_revision,
        label="Elo execution",
    )
    sources = _validate_sources(
        source_artifacts,
        label="Elo execution",
    )
    artifacts = _elo_binary_artifacts(sources)
    formula = _text(formula_id, "formula_id")
    resolved_divisor = _positive_finite(divisor, "divisor")
    if predictions.empty:
        raise ValueError("Elo prediction execution must not be empty.")

    required: set[str] = {
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_TEAM_ELO",
        "HOME_TEAM_ELO",
        "AWAY_WIN_PROB",
        "HOME_WIN_PROB",
    }
    missing: list[str] = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            "Elo predictions are missing execution-evidence columns: " + ", ".join(missing)
        )

    computations = tuple(
        sorted(
            (
                _computation(
                    row,
                    formula_id=formula,
                    divisor=resolved_divisor,
                )
                for _, row in predictions.iterrows()
            ),
            key=lambda value: value.game_id,
        )
    )
    game_ids = tuple(value.game_id for value in computations)
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("Elo prediction execution contains duplicate game IDs.")

    return EloPredictionExecution(
        predictions=predictions.copy(deep=True),
        computations=computations,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        binary_artifacts=artifacts,
    )


def _statistical_computation(
    *,
    game_id: object,
    raw_values: tuple[float, ...],
    transformed_values: tuple[float, ...],
    raw_output: object,
    post_output: object,
    feature_count: int,
    task: str,
    scaler_state: PredictionArtifactState,
) -> StatisticalPredictionComputation:
    """Validate one exact persisted-estimator computation."""
    resolved_game_id = _text(game_id, "game_id")

    if len(raw_values) != feature_count:
        raise ValueError("Raw feature vector length does not match the feature schema.")
    if len(transformed_values) != feature_count:
        raise ValueError("Transformed feature vector length does not match the feature schema.")

    raw = tuple(_finite(value, "raw feature value") for value in raw_values)
    transformed = tuple(_finite(value, "transformed feature value") for value in transformed_values)

    if scaler_state is PredictionArtifactState.ABSENT and transformed != raw:
        raise ValueError("Raw and transformed feature values must match when the scaler is absent.")

    resolved_raw_output = _finite(
        raw_output,
        "raw_estimator_output",
    )
    resolved_post_output = _finite(
        post_output,
        "post_estimator_output",
    )

    if task == "classification":
        resolved_raw_output = _probability(
            resolved_raw_output,
            "raw_estimator_output",
        )
        resolved_post_output = _probability(
            resolved_post_output,
            "post_estimator_output",
        )

    return StatisticalPredictionComputation(
        game_id=resolved_game_id,
        raw_feature_values=raw,
        transformed_feature_values=transformed,
        raw_estimator_output=resolved_raw_output,
        post_estimator_output=resolved_post_output,
    )


def _validate_statistical_feature_schema(
    schema: PredictionFeatureSchema,
) -> None:
    """Validate the exact ordered estimator feature contract."""
    if schema.task not in {"classification", "regression"}:
        raise ValueError("Statistical feature schema task must be classification or regression.")

    _text(
        schema.model_name,
        "feature schema model_name",
    )
    _text(
        schema.model_type,
        "feature schema model_type",
    )
    _positive_integer(
        schema.modeling_schema_version,
        "modeling_schema_version",
    )
    _positive_integer(
        schema.epa_window,
        "epa_window",
    )
    _text(
        schema.feature_set_name,
        "feature_set_name",
    )

    if not schema.ordered_columns:
        raise ValueError("Statistical feature schema columns must not be empty.")

    normalized_columns = tuple(_text(column, "feature column") for column in schema.ordered_columns)
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Statistical feature schema columns must be unique.")

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
        raise ValueError("Statistical feature schema_id does not match canonical schema content.")


def _validate_statistical_binary_artifacts(
    references: tuple[BinaryArtifactReference, ...],
    *,
    feature_schema: PredictionFeatureSchema,
    post_processing: PredictionPostProcessingEvidence | None,
) -> dict[PredictionArtifactKind, BinaryArtifactReference]:
    """Validate the exact statistical binary-artifact inventory."""
    expected_kinds = {
        PredictionArtifactKind.EXTERNAL_CALIBRATOR,
        PredictionArtifactKind.MODEL,
        PredictionArtifactKind.MODEL_METADATA,
        PredictionArtifactKind.SCALER,
    }

    kinds = tuple(reference.kind.value for reference in references)
    if kinds != tuple(sorted(set(kinds))):
        raise ValueError("Statistical binary artifacts must be ordered by unique artifact kind.")

    by_kind = {reference.kind: reference for reference in references}
    if set(by_kind) != expected_kinds:
        raise ValueError(
            "Statistical execution requires exactly model, "
            "model_metadata, scaler, and external_calibrator "
            "artifact records."
        )

    for reference in references:
        _validate_binary_reference(reference)

    _validate_required_statistical_artifacts(by_kind)
    _validate_statistical_scaler(
        feature_schema=feature_schema,
        artifacts=by_kind,
    )
    _validate_statistical_post_processing(
        feature_schema=feature_schema,
        post_processing=post_processing,
        artifacts=by_kind,
    )

    return by_kind


def _validate_required_statistical_artifacts(
    artifacts: dict[
        PredictionArtifactKind,
        BinaryArtifactReference,
    ],
) -> None:
    """Require the estimator and metadata artifacts."""
    for kind in (
        PredictionArtifactKind.MODEL,
        PredictionArtifactKind.MODEL_METADATA,
    ):
        if artifacts[kind].state is not PredictionArtifactState.PRESENT:
            raise ValueError(f"Statistical {kind.value} artifact must be present.")


def _validate_statistical_scaler(
    *,
    feature_schema: PredictionFeatureSchema,
    artifacts: dict[
        PredictionArtifactKind,
        BinaryArtifactReference,
    ],
) -> None:
    """Validate scaler presence against the estimator type."""
    scaler = artifacts[PredictionArtifactKind.SCALER]

    if feature_schema.model_type == "logistic":
        if scaler.state is not PredictionArtifactState.PRESENT:
            raise ValueError("Logistic statistical execution requires a present scaler.")
        return

    if scaler.state is not PredictionArtifactState.ABSENT:
        raise ValueError("Non-logistic statistical execution requires an absent scaler.")


def _validate_statistical_post_processing(
    *,
    feature_schema: PredictionFeatureSchema,
    post_processing: PredictionPostProcessingEvidence | None,
    artifacts: dict[
        PredictionArtifactKind,
        BinaryArtifactReference,
    ],
) -> None:
    """Validate task-specific post-processing and calibrator state."""
    external = artifacts[PredictionArtifactKind.EXTERNAL_CALIBRATOR]

    if feature_schema.task == "classification":
        _validate_classification_post_processing(
            feature_schema=feature_schema,
            post_processing=post_processing,
            external_calibrator=external,
        )
        return

    if post_processing is not None:
        raise ValueError("Regression execution must not contain Win post_processing.")

    if external.state is not PredictionArtifactState.ABSENT:
        raise ValueError("Regression execution requires an absent external calibrator.")


def _validate_classification_post_processing(
    *,
    feature_schema: PredictionFeatureSchema,
    post_processing: PredictionPostProcessingEvidence | None,
    external_calibrator: BinaryArtifactReference,
) -> None:
    """Validate Win-classification post-processing evidence."""
    if feature_schema.model_name != "win_prob":
        raise ValueError("Classification execution requires model_name win_prob.")

    if post_processing is None:
        raise ValueError("Classification execution requires post_processing evidence.")

    if external_calibrator.state is not post_processing.external_calibrator_state:
        raise ValueError("External calibrator state does not match post-processing evidence.")


def _validate_binary_reference(
    reference: BinaryArtifactReference,
) -> None:
    """Validate one statistical binary-artifact reference."""
    relative_path = _text(
        reference.source_relative_path,
        "binary artifact source path",
    )
    parts = relative_path.split("/")
    if relative_path.startswith("/") or ".." in parts:
        raise ValueError("Binary artifact source path must be safe and relative.")

    if reference.state is PredictionArtifactState.PRESENT:
        digest = reference.content_digest
        size_bytes = reference.size_bytes

        if digest is None or size_bytes is None:
            raise ValueError("Present binary artifact requires digest and size.")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Binary artifact digest must be lowercase SHA-256.")
        if size_bytes < 1:
            raise ValueError("Present binary artifact size must be positive.")
        return

    if reference.content_digest is not None or reference.size_bytes is not None:
        raise ValueError("Absent binary artifact must not contain digest or size.")


def _computation(
    row: Series,
    *,
    formula_id: str,
    divisor: float,
) -> EloPredictionComputation:
    game_id = _text(row["GAME_ID"], "GAME_ID")
    season = _text(row["YEAR"], "YEAR")
    week = _positive_integer(row["WEEK_NUM"], "WEEK_NUM")
    away_team = _text(row["AWAY_TEAM"], "AWAY_TEAM")
    home_team = _text(row["HOME_TEAM"], "HOME_TEAM")
    if away_team == home_team:
        raise ValueError("Elo Away and Home teams must differ.")
    away_elo = _finite(row["AWAY_TEAM_ELO"], "AWAY_TEAM_ELO")
    home_elo = _finite(row["HOME_TEAM_ELO"], "HOME_TEAM_ELO")
    away_probability = _probability(row["AWAY_WIN_PROB"], "AWAY_WIN_PROB")
    home_probability = _probability(row["HOME_WIN_PROB"], "HOME_WIN_PROB")
    if not math.isclose(
        away_probability + home_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("Elo Away and Home probabilities must be complementary.")
    return EloPredictionComputation(
        game_id=game_id,
        season=season,
        week=week,
        away_team=away_team,
        home_team=home_team,
        away_elo=away_elo,
        home_elo=home_elo,
        formula_id=formula_id,
        divisor=divisor,
        away_win_probability=away_probability,
        home_win_probability=home_probability,
    )


def _elo_binary_artifacts(
    sources: dict[str, SourceArtifactReference],
) -> tuple[BinaryArtifactReference, ...]:
    lineage = _required_present_source(sources, _ELO_LINEAGE_PATH)
    state = _required_present_source(sources, _ELO_STATE_PATH)
    return (
        _binary_reference(PredictionArtifactKind.ELO_LINEAGE, lineage),
        _binary_reference(PredictionArtifactKind.ELO_STATE, state),
    )


def _binary_reference(
    kind: PredictionArtifactKind,
    source: SourceArtifactReference,
) -> BinaryArtifactReference:
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=source.relative_path,
        state=PredictionArtifactState.PRESENT,
        content_digest=source.content_digest,
        size_bytes=source.size_bytes,
    )


def _required_present_source(
    sources: dict[str, SourceArtifactReference],
    relative_path: str,
) -> SourceArtifactReference:
    try:
        reference = sources[relative_path]
    except KeyError as exc:
        raise ValueError(f"Elo execution source inventory is missing {relative_path}.") from exc
    if (
        reference.state is not PredictionSourceState.PRESENT
        or reference.content_digest is None
        or reference.size_bytes is None
    ):
        raise ValueError(f"Elo execution requires present source artifact {relative_path}.")
    return reference


def _validate_sources(
    references: tuple[SourceArtifactReference, ...],
    *,
    label: str,
) -> dict[str, SourceArtifactReference]:
    if not references:
        raise ValueError(f"{label} source_artifacts must not be empty.")

    paths = tuple(reference.relative_path for reference in references)
    if paths != tuple(sorted(set(paths))):
        raise ValueError(f"{label} source_artifacts must be sorted and unique.")

    return {reference.relative_path: reference for reference in references}


def _validate_revision(
    revision: SourceRevision,
    *,
    label: str,
) -> None:
    if len(revision.commit) != 40 or any(
        character not in "0123456789abcdef" for character in revision.commit
    ):
        raise ValueError(f"{label} source revision must be a lowercase Git commit SHA.")

    if revision.tracked_worktree_clean is not True:
        raise ValueError(f"{label} requires a clean tracked source revision.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real | str):
        raise ValueError(f"{label} must be a positive integer.")

    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc

    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{label} must be a positive integer.")

    return int(numeric)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real | str):
        raise ValueError(f"{label} must be finite numeric data.")

    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be finite numeric data.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric data.")

    return result


def _positive_finite(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def _probability(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0 or result > 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result
