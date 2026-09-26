# src/gridiron_edge/evaluation/comparable_games_evidence.py
"""Immutable per-event comparable-games retrieval evidence.

Sibling to ``logistic_explanation_evidence.py``: decomposes one query
game's position in the comparable-games corpus (``comparable_games_corpus.py``)
into its nearest historical neighbors by Euclidean distance in the
champion's own standardized feature space, plus each match's real recorded
outcome. It never predicts or re-scores anything — retrieval, not
inference — and is unaffected by D53's calibration-boundary problem, since
nearest-neighbor distance does not decompose a calibrated decision
function the way per-feature attribution does.

A match's ``distance`` measures feature-space proximity, not similarity of
outcome; two close games can still have opposite results. ``favorite_won``/
``favorite_covered`` describe what actually happened in that historical
game, never a prediction about the query game.
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

COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION: Final[int] = 1
EUCLIDEAN_STANDARDIZED_METRIC: Final[str] = "euclidean_l2_standardized"
EUCLIDEAN_STANDARDIZED_METRIC_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ComparableFeatureContribution:
    """One feature's contribution to the squared distance between two games."""

    feature_name: str
    query_value: float
    candidate_value: float
    squared_difference: float


@dataclass(frozen=True, slots=True)
class ComparableGameMatch:
    """One historical game returned as a comparable, with its real outcome."""

    game_id: str
    rank: int
    distance: float
    season: str
    week: int
    game_date: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    favorite_team: str | None
    spread_magnitude: float | None
    favorite_won: bool | None
    favorite_covered: bool | None
    top_contributing_features: tuple[ComparableFeatureContribution, ...]


@dataclass(frozen=True, slots=True)
class ComparableGamesBatch:
    """Frozen immutable comparable-games retrieval result for one event."""

    schema_version: int
    batch_id: str
    event_id: str
    game_id: str
    corpus_id: str
    model_name: str
    model_type: str
    feature_schema: PredictionFeatureSchema
    metric: str
    metric_version: int
    k_requested: int
    distance_threshold: float
    generated_at: datetime
    matches: tuple[ComparableGameMatch, ...]
    sample_size: int
    favorite_win_rate: float | None
    favorite_cover_rate: float | None


def comparable_games_batch_id(
    *,
    event_id: str,
    game_id: str,
    corpus_id: str,
    model_name: str,
    model_type: str,
    feature_schema: PredictionFeatureSchema,
    metric: str,
    metric_version: int,
    k_requested: int,
    distance_threshold: float,
    matches: tuple[ComparableGameMatch, ...],
    sample_size: int,
    favorite_win_rate: float | None,
    favorite_cover_rate: float | None,
) -> str:
    """Return the SHA-256 identity of one canonical batch payload.

    Deliberately excludes ``generated_at``: identity is determined entirely
    by the query event, the corpus, and the retrieval parameters, so
    re-running ``find-comparables`` for the same event against the same
    corpus reproduces the same ``batch_id`` — an idempotent write-or-replay.
    This is a deliberate improvement over ``LogisticExplanationBatch``,
    whose timestamp-inclusive identity let two runs over the same evidence
    produce two batches claiming the same event (D52).
    """
    payload = _identity_payload(
        event_id=event_id,
        game_id=game_id,
        corpus_id=corpus_id,
        model_name=model_name,
        model_type=model_type,
        feature_schema=feature_schema,
        metric=metric,
        metric_version=metric_version,
        k_requested=k_requested,
        distance_threshold=distance_threshold,
        matches=matches,
        sample_size=sample_size,
        favorite_win_rate=favorite_win_rate,
        favorite_cover_rate=favorite_cover_rate,
    )
    return _canonical_digest(payload)


def create_comparable_games_batch(
    *,
    event_id: str,
    game_id: str,
    corpus_id: str,
    model_name: str,
    model_type: str,
    feature_schema: PredictionFeatureSchema,
    metric: str,
    metric_version: int,
    k_requested: int,
    distance_threshold: float,
    generated_at: datetime,
    matches: tuple[ComparableGameMatch, ...],
    sample_size: int,
    favorite_win_rate: float | None,
    favorite_cover_rate: float | None,
) -> ComparableGamesBatch:
    """Create and validate one complete comparable-games batch."""
    batch_id = comparable_games_batch_id(
        event_id=event_id,
        game_id=game_id,
        corpus_id=corpus_id,
        model_name=model_name,
        model_type=model_type,
        feature_schema=feature_schema,
        metric=metric,
        metric_version=metric_version,
        k_requested=k_requested,
        distance_threshold=distance_threshold,
        matches=matches,
        sample_size=sample_size,
        favorite_win_rate=favorite_win_rate,
        favorite_cover_rate=favorite_cover_rate,
    )
    batch = ComparableGamesBatch(
        schema_version=COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION,
        batch_id=batch_id,
        event_id=event_id,
        game_id=game_id,
        corpus_id=corpus_id,
        model_name=model_name,
        model_type=model_type,
        feature_schema=feature_schema,
        metric=metric,
        metric_version=metric_version,
        k_requested=k_requested,
        distance_threshold=distance_threshold,
        generated_at=generated_at,
        matches=matches,
        sample_size=sample_size,
        favorite_win_rate=favorite_win_rate,
        favorite_cover_rate=favorite_cover_rate,
    )
    validate_comparable_games_batch(batch)
    return batch


def validate_comparable_games_batch(batch: ComparableGamesBatch) -> None:  # noqa: PLR0912
    """Validate one batch's identity, invariants, and internal consistency."""
    if batch.schema_version != COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported comparable-games evidence schema_version: {batch.schema_version}."
        )
    _digest(batch.batch_id, "batch_id")
    _text(batch.event_id, "event_id")
    _text(batch.game_id, "game_id")
    _digest(batch.corpus_id, "corpus_id")
    _text(batch.model_name, "model_name")
    _text(batch.model_type, "model_type")
    _validate_feature_schema(batch.feature_schema)
    _text(batch.metric, "metric")
    if batch.metric_version < 1:
        raise ValueError("metric_version must be at least 1.")
    if batch.k_requested < 1:
        raise ValueError("k_requested must be at least 1.")
    if batch.distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    _utc(batch.generated_at, "generated_at")
    if batch.sample_size != len(batch.matches):
        raise ValueError("sample_size must equal the number of matches.")
    if len(batch.matches) > batch.k_requested:
        raise ValueError("matches cannot exceed k_requested.")
    for previous, current in zip(batch.matches, batch.matches[1:], strict=False):
        if current.rank != previous.rank + 1:
            raise ValueError("matches must be contiguously ranked starting at 1.")
        if current.distance < previous.distance:
            raise ValueError("matches must be sorted by non-decreasing distance.")
    if batch.matches and batch.matches[0].rank != 1:
        raise ValueError("matches must start at rank 1.")
    for match in batch.matches:
        if match.distance > batch.distance_threshold:
            raise ValueError("every match's distance must be within distance_threshold.")
        if match.distance < 0.0:
            raise ValueError("match distance must be non-negative.")
    _validate_rate(batch.favorite_win_rate, "favorite_win_rate")
    _validate_rate(batch.favorite_cover_rate, "favorite_cover_rate")

    expected_id = comparable_games_batch_id(
        event_id=batch.event_id,
        game_id=batch.game_id,
        corpus_id=batch.corpus_id,
        model_name=batch.model_name,
        model_type=batch.model_type,
        feature_schema=batch.feature_schema,
        metric=batch.metric,
        metric_version=batch.metric_version,
        k_requested=batch.k_requested,
        distance_threshold=batch.distance_threshold,
        matches=batch.matches,
        sample_size=batch.sample_size,
        favorite_win_rate=batch.favorite_win_rate,
        favorite_cover_rate=batch.favorite_cover_rate,
    )
    if expected_id != batch.batch_id:
        raise ValueError("Comparable-games batch_id does not match its own content.")


def comparable_games_batch_payload(batch: ComparableGamesBatch) -> dict[str, object]:
    """Return the complete stable JSON-compatible batch representation."""
    validate_comparable_games_batch(batch)
    return {
        "schema_version": batch.schema_version,
        "batch_id": batch.batch_id,
        "generated_at": batch.generated_at.isoformat(),
        **_identity_payload(
            event_id=batch.event_id,
            game_id=batch.game_id,
            corpus_id=batch.corpus_id,
            model_name=batch.model_name,
            model_type=batch.model_type,
            feature_schema=batch.feature_schema,
            metric=batch.metric,
            metric_version=batch.metric_version,
            k_requested=batch.k_requested,
            distance_threshold=batch.distance_threshold,
            matches=batch.matches,
            sample_size=batch.sample_size,
            favorite_win_rate=batch.favorite_win_rate,
            favorite_cover_rate=batch.favorite_cover_rate,
        ),
    }


def _validate_rate(value: float | None, label: str) -> None:
    if value is None:
        return
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be within [0, 1] when present.")


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
    event_id: str,
    game_id: str,
    corpus_id: str,
    model_name: str,
    model_type: str,
    feature_schema: PredictionFeatureSchema,
    metric: str,
    metric_version: int,
    k_requested: int,
    distance_threshold: float,
    matches: tuple[ComparableGameMatch, ...],
    sample_size: int,
    favorite_win_rate: float | None,
    favorite_cover_rate: float | None,
) -> dict[str, object]:
    return {
        "event_id": _text(event_id, "event_id"),
        "game_id": _text(game_id, "game_id"),
        "corpus_id": _digest(corpus_id, "corpus_id"),
        "model_name": _text(model_name, "model_name"),
        "model_type": _text(model_type, "model_type"),
        "feature_schema": _feature_schema_payload(feature_schema),
        "metric": _text(metric, "metric"),
        "metric_version": metric_version,
        "k_requested": k_requested,
        "distance_threshold": distance_threshold,
        "matches": [_match_payload(match) for match in matches],
        "sample_size": sample_size,
        "favorite_win_rate": favorite_win_rate,
        "favorite_cover_rate": favorite_cover_rate,
    }


def _match_payload(match: ComparableGameMatch) -> dict[str, object]:
    return {
        "game_id": match.game_id,
        "rank": match.rank,
        "distance": match.distance,
        "season": match.season,
        "week": match.week,
        "game_date": match.game_date,
        "away_team": match.away_team,
        "home_team": match.home_team,
        "away_score": match.away_score,
        "home_score": match.home_score,
        "favorite_team": match.favorite_team,
        "spread_magnitude": match.spread_magnitude,
        "favorite_won": match.favorite_won,
        "favorite_covered": match.favorite_covered,
        "top_contributing_features": [
            {
                "feature_name": item.feature_name,
                "query_value": item.query_value,
                "candidate_value": item.candidate_value,
                "squared_difference": item.squared_difference,
            }
            for item in match.top_contributing_features
        ],
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
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC.")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError(f"{label} must use UTC.")
    return value
