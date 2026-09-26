# src/gridiron_edge/evaluation/comparable_games_corpus.py
"""Immutable historical feature-vector corpus for comparable-game retrieval.

Sibling to ``prediction_input_evidence.py`` and
``logistic_explanation_evidence.py``, not an extension of either: this
corpus is the searchable candidate pool that a comparable-games retrieval
batch (``comparable_games_evidence.py``) is computed against. It never
predicts, scores, or evaluates a model — it is a one-time replay of the
same feature-construction pipeline production prediction already trusts
(``_rebuild_features_with_window`` plus ``FEATURE_SETS["combined"]``),
scaled by one specific fitted estimator's own snapshot.

Every historical game's ``transformed_feature_values`` is built only from
that game's own pre-kickoff rolling and situational state (the same
construction used for live prediction and walk-forward evaluation), so no
row can leak information from after its own kickoff. The corpus is bound to
an exact ``(model_content_digest, scaler_content_digest)`` pair; a caller
comparing a query event against this corpus must verify its own evidence
was produced by the identical snapshot before treating distances as
meaningful (see ``comparable_games_evidence_builder.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from typing import Final

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionFeatureSchema,
    prediction_feature_schema_id,
)

COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION: Final[int] = 1
_DIGEST_PATTERN_LENGTH: Final[int] = 64


@dataclass(frozen=True, slots=True)
class ComparableGamesFrameReference:
    """One content-addressed Parquet frame holding the corpus's rows."""

    artifact: str
    row_count: int
    columns: tuple[str, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class ComparableGamesCorpus:
    """Frozen immutable candidate pool for one model/scaler snapshot."""

    schema_version: int
    corpus_id: str
    model_name: str
    model_type: str
    model_content_digest: str
    scaler_content_digest: str
    feature_schema: PredictionFeatureSchema
    generated_at: datetime
    frame: ComparableGamesFrameReference
    distance_threshold: float
    leave_one_out_median_distance: float
    leave_one_out_percentile: float


def comparable_games_corpus_id(
    *,
    model_name: str,
    model_type: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    frame: ComparableGamesFrameReference,
) -> str:
    """Return the SHA-256 identity of one canonical corpus payload.

    Deliberately excludes ``generated_at`` and the derived threshold
    statistics: identity is determined entirely by the model/scaler
    snapshot and the exact row data, so rebuilding against unchanged inputs
    reproduces the same ``corpus_id`` (an idempotent write-or-replay,
    unlike ``LogisticExplanationBatch``'s timestamp-inclusive identity).
    """
    payload = _identity_payload(
        model_name=model_name,
        model_type=model_type,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        frame=frame,
    )
    return _canonical_digest(payload)


def create_comparable_games_corpus(
    *,
    model_name: str,
    model_type: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    generated_at: datetime,
    frame: ComparableGamesFrameReference,
    distance_threshold: float,
    leave_one_out_median_distance: float,
    leave_one_out_percentile: float,
) -> ComparableGamesCorpus:
    """Create and validate one complete comparable-games corpus."""
    corpus_id = comparable_games_corpus_id(
        model_name=model_name,
        model_type=model_type,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        frame=frame,
    )
    corpus = ComparableGamesCorpus(
        schema_version=COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION,
        corpus_id=corpus_id,
        model_name=model_name,
        model_type=model_type,
        model_content_digest=model_content_digest,
        scaler_content_digest=scaler_content_digest,
        feature_schema=feature_schema,
        generated_at=generated_at,
        frame=frame,
        distance_threshold=distance_threshold,
        leave_one_out_median_distance=leave_one_out_median_distance,
        leave_one_out_percentile=leave_one_out_percentile,
    )
    validate_comparable_games_corpus(corpus)
    return corpus


def validate_comparable_games_corpus(corpus: ComparableGamesCorpus) -> None:
    """Validate one corpus's identity, schema, and invariants."""
    if corpus.schema_version != COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported comparable-games corpus schema_version: {corpus.schema_version}."
        )
    _digest(corpus.corpus_id, "corpus_id")
    _text(corpus.model_name, "model_name")
    _text(corpus.model_type, "model_type")
    _digest(corpus.model_content_digest, "model_content_digest")
    _digest(corpus.scaler_content_digest, "scaler_content_digest")
    _validate_feature_schema(corpus.feature_schema)
    _utc(corpus.generated_at, "generated_at")
    _validate_frame(corpus.frame)
    if corpus.frame.row_count < 2:
        raise ValueError("Comparable-games corpus requires at least two rows.")
    if not (0.0 <= corpus.leave_one_out_percentile <= 100.0):
        raise ValueError("leave_one_out_percentile must be within [0, 100].")
    if corpus.distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    if corpus.leave_one_out_median_distance < 0.0:
        raise ValueError("leave_one_out_median_distance must be non-negative.")

    expected_id = comparable_games_corpus_id(
        model_name=corpus.model_name,
        model_type=corpus.model_type,
        model_content_digest=corpus.model_content_digest,
        scaler_content_digest=corpus.scaler_content_digest,
        feature_schema=corpus.feature_schema,
        frame=corpus.frame,
    )
    if expected_id != corpus.corpus_id:
        raise ValueError("Comparable-games corpus_id does not match its own content.")


def comparable_games_corpus_payload(corpus: ComparableGamesCorpus) -> dict[str, object]:
    """Return the complete stable JSON-compatible corpus manifest."""
    validate_comparable_games_corpus(corpus)
    return {
        "schema_version": corpus.schema_version,
        "corpus_id": corpus.corpus_id,
        "generated_at": corpus.generated_at.isoformat(),
        "distance_threshold": corpus.distance_threshold,
        "leave_one_out_median_distance": corpus.leave_one_out_median_distance,
        "leave_one_out_percentile": corpus.leave_one_out_percentile,
        **_identity_payload(
            model_name=corpus.model_name,
            model_type=corpus.model_type,
            model_content_digest=corpus.model_content_digest,
            scaler_content_digest=corpus.scaler_content_digest,
            feature_schema=corpus.feature_schema,
            frame=corpus.frame,
        ),
    }


def frame_content_digest(columns: tuple[str, ...], row_payloads: list[tuple[object, ...]]) -> str:
    """Return the canonical content digest of one ordered row set.

    Used both when writing the corpus frame (to record its digest) and when
    reading it back (to authenticate the Parquet bytes reproduce the exact
    rows the manifest was built from).
    """
    sanitized_rows = [[_sanitize(item) for item in row] for row in row_payloads]
    encoded = json.dumps(
        {"columns": list(columns), "rows": sanitized_rows},
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode()
    return sha256(encoded).hexdigest()


def _sanitize(value: object) -> object:
    """Map a NaN float (missing market data) to JSON ``null``; pass through otherwise."""
    coerced = value.item() if hasattr(value, "item") else value
    if isinstance(coerced, float) and coerced != coerced:  # noqa: PLR0124 (NaN != NaN)
        return None
    return coerced


def _json_default(value: object) -> object:
    # numpy scalar types (int64/float64) surface from pandas itertuples();
    # coerce to native Python so json.dumps never raises on them.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _validate_frame(frame: ComparableGamesFrameReference) -> None:
    if not isinstance(frame, ComparableGamesFrameReference):
        raise TypeError("frame must be a ComparableGamesFrameReference.")
    _text(frame.artifact, "frame.artifact")
    if frame.row_count < 0:
        raise ValueError("frame.row_count must be non-negative.")
    if not frame.columns:
        raise ValueError("frame.columns must be non-empty.")
    _digest(frame.content_digest, "frame.content_digest")


def _validate_feature_schema(schema: PredictionFeatureSchema) -> None:
    if not isinstance(schema, PredictionFeatureSchema):
        raise TypeError("feature_schema must be a PredictionFeatureSchema.")
    expected_id = prediction_feature_schema_id(
        model_name=schema.model_name,
        model_type=schema.model_type,
        task=schema.task,
        modeling_schema_version=schema.modeling_schema_version,
        epa_window=schema.epa_window,
        feature_set_name=schema.feature_set_name,
        ordered_columns=schema.ordered_columns,
    )
    if expected_id != schema.schema_id:
        raise ValueError("feature_schema.schema_id does not match its own content.")


def _identity_payload(
    *,
    model_name: str,
    model_type: str,
    model_content_digest: str,
    scaler_content_digest: str,
    feature_schema: PredictionFeatureSchema,
    frame: ComparableGamesFrameReference,
) -> dict[str, object]:
    return {
        "model_name": _text(model_name, "model_name"),
        "model_type": _text(model_type, "model_type"),
        "model_content_digest": _digest(model_content_digest, "model_content_digest"),
        "scaler_content_digest": _digest(scaler_content_digest, "scaler_content_digest"),
        "feature_schema": _feature_schema_payload(feature_schema),
        "frame": {
            "artifact": frame.artifact,
            "row_count": frame.row_count,
            "columns": list(frame.columns),
            "content_digest": frame.content_digest,
        },
    }


def _feature_schema_payload(schema: PredictionFeatureSchema) -> dict[str, object]:
    return {
        "schema_id": schema.schema_id,
        "model_name": schema.model_name,
        "model_type": schema.model_type,
        "task": schema.task,
        "modeling_schema_version": schema.modeling_schema_version,
        "epa_window": schema.epa_window,
        "feature_set_name": schema.feature_set_name,
        "ordered_columns": list(schema.ordered_columns),
    }


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != _DIGEST_PATTERN_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC.")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError(f"{label} must use UTC.")
    return value
