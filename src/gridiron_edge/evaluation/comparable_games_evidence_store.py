# src/gridiron_edge/evaluation/comparable_games_evidence_store.py
"""Immutable JSON persistence for comparable-games retrieval batches."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.comparable_games_evidence import (
    COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    ComparableGamesBatch,
    comparable_games_batch_payload,
    validate_comparable_games_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import PredictionFeatureSchema

_STORE_DIRECTORY = "data/output/comparable_games"
_STORE_SCHEMA_VERSION = 1


class AmbiguousComparableGamesError(ValueError):
    """More than one persisted batch claims the same event.

    The store is immutable and create-only with no "current" selection
    concept, mirroring ``logistic_explanation_evidence_store.py``. Unlike
    that store, ``batch_id`` here is idempotent per (event, corpus) pair
    (see D-record for this unit), so this specifically indicates the event
    was retrieved against more than one distinct corpus generation (for
    example, before and after a champion retrain) — a genuine staleness
    condition a caller must resolve deliberately, not silently.
    """


def comparable_games_root(repo: Path | None = None) -> Path:
    """Return the canonical comparable-games evidence store root."""
    return (repo or get_settings().repo_root) / _STORE_DIRECTORY


def _embedded_id(path: Path, *, key: str) -> str | None:
    """Return one already-written artifact's embedded identity, or ``None``.

    Malformed JSON or an unexpected shape both resolve to ``None`` (never
    matches a real identity), which the caller treats as a genuine
    conflict rather than letting the decode error propagate uncaught.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return raw.get(key) if isinstance(raw, dict) else None


def comparable_games_batch_path(batch_id: str, *, repo: Path | None = None) -> Path:
    """Return the canonical identity-addressed batch path."""
    identity = _digest(batch_id, "batch_id")
    return (
        comparable_games_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "batches"
        / f"{identity}.json"
    )


def write_comparable_games_batch(batch: ComparableGamesBatch, *, repo: Path | None = None) -> Path:
    """Create one immutable comparable-games batch artifact or accept an idempotent replay.

    Unlike ``write_logistic_explanation_batch``, this does not compare full
    encoded bytes on replay: ``batch_id`` deliberately excludes
    ``generated_at`` (see ``comparable_games_batch_id``), so re-running
    ``find-comparables`` for the same event against the same corpus
    legitimately produces a payload that differs only in that timestamp.
    The batch path is itself content-addressed by ``batch_id``, so an
    existing file at that exact path is definitionally a valid prior write
    of the same content; only its own embedded identity is re-checked.
    """
    validate_comparable_games_batch(batch)
    path = comparable_games_batch_path(batch.batch_id, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _embedded_id(path, key="batch_id") != batch.batch_id:
            raise ValueError(
                "Comparable-games batch identity cannot be reused with different content."
            )
        return path

    encoded = (
        json.dumps(
            {
                "store_schema_version": _STORE_SCHEMA_VERSION,
                "batch_id": batch.batch_id,
                "batch": comparable_games_batch_payload(batch),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _embedded_id(path, key="batch_id") != batch.batch_id:
                raise ValueError(
                    "Comparable-games batch identity cannot be reused with different content."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_comparable_games_batch(path: Path) -> ComparableGamesBatch:
    """Read and strictly validate one exact immutable batch artifact."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Comparable-games batch contains malformed JSON: {path}") from exc
    raw = _object(value, "comparable-games batch artifact")
    _exact_keys(raw, {"store_schema_version", "batch_id", "batch"}, "Artifact")
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != _STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported comparable-games evidence store schema version.")
    embedded_id = _digest(_text(raw["batch_id"], "batch_id"), "batch_id")
    batch = _batch(raw["batch"])
    if embedded_id != batch.batch_id:
        raise ValueError("Stored batch identity does not match batch content.")
    expected = comparable_games_batch_path(embedded_id, repo=_artifact_repo(path))
    if path.resolve() != expected.resolve():
        raise ValueError("Comparable-games batch path and embedded identity disagree.")
    return batch


def list_comparable_games_batches(*, repo: Path | None = None) -> tuple[ComparableGamesBatch, ...]:
    """Scan and deterministically list every persisted comparable-games batch."""
    return _all_batches(repo=repo)


def find_comparable_games_by_event(
    event_id: str,
    *,
    repo: Path | None = None,
) -> ComparableGamesBatch | None:
    """Scan immutable batches and resolve at most one batch for an event."""
    normalized = _text(event_id, "event_id")
    matches = [batch for batch in _all_batches(repo=repo) if batch.event_id == normalized]
    if len(matches) > 1:
        raise AmbiguousComparableGamesError(
            f"Multiple comparable-games batches claim event {normalized!r}."
        )
    return matches[0] if matches else None


def _all_batches(*, repo: Path | None) -> tuple[ComparableGamesBatch, ...]:
    directory = comparable_games_root(repo) / f"schema={_STORE_SCHEMA_VERSION}" / "batches"
    if not directory.exists():
        return ()
    values = tuple(read_comparable_games_batch(path) for path in sorted(directory.glob("*.json")))
    return tuple(sorted(values, key=lambda value: value.batch_id))


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = tuple(Path(_STORE_DIRECTORY).parts)
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Comparable-games batch path is outside the canonical store.")


def _batch(value: object) -> ComparableGamesBatch:
    raw = _object(value, "batch")
    _exact_keys(
        raw,
        {
            "schema_version",
            "batch_id",
            "event_id",
            "game_id",
            "corpus_id",
            "model_name",
            "model_type",
            "feature_schema",
            "metric",
            "metric_version",
            "k_requested",
            "distance_threshold",
            "generated_at",
            "matches",
            "sample_size",
            "favorite_win_rate",
            "favorite_cover_rate",
        },
        "Batch",
    )
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported comparable-games evidence schema version.")
    batch = ComparableGamesBatch(
        schema_version=schema_version,
        batch_id=_digest(_text(raw["batch_id"], "batch_id"), "batch_id"),
        event_id=_text(raw["event_id"], "event_id"),
        game_id=_text(raw["game_id"], "game_id"),
        corpus_id=_digest(_text(raw["corpus_id"], "corpus_id"), "corpus_id"),
        model_name=_text(raw["model_name"], "model_name"),
        model_type=_text(raw["model_type"], "model_type"),
        feature_schema=_feature_schema(raw["feature_schema"]),
        metric=_text(raw["metric"], "metric"),
        metric_version=_integer(raw["metric_version"], "metric_version"),
        k_requested=_integer(raw["k_requested"], "k_requested"),
        distance_threshold=_float(raw["distance_threshold"], "distance_threshold"),
        generated_at=_datetime(raw["generated_at"], "generated_at"),
        matches=tuple(_match(item) for item in _list(raw["matches"], "matches")),
        sample_size=_integer(raw["sample_size"], "sample_size"),
        favorite_win_rate=_optional_float(raw["favorite_win_rate"], "favorite_win_rate"),
        favorite_cover_rate=_optional_float(raw["favorite_cover_rate"], "favorite_cover_rate"),
    )
    validate_comparable_games_batch(batch)
    return batch


def _match(value: object) -> ComparableGameMatch:
    raw = _object(value, "match")
    _exact_keys(
        raw,
        {
            "game_id",
            "rank",
            "distance",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "favorite_team",
            "spread_magnitude",
            "favorite_won",
            "favorite_covered",
            "top_contributing_features",
        },
        "Match",
    )
    return ComparableGameMatch(
        game_id=_text(raw["game_id"], "game_id"),
        rank=_integer(raw["rank"], "rank"),
        distance=_float(raw["distance"], "distance"),
        season=_text(raw["season"], "season"),
        week=_integer(raw["week"], "week"),
        game_date=_text(raw["game_date"], "game_date"),
        away_team=_text(raw["away_team"], "away_team"),
        home_team=_text(raw["home_team"], "home_team"),
        away_score=_integer(raw["away_score"], "away_score"),
        home_score=_integer(raw["home_score"], "home_score"),
        favorite_team=_optional_text(raw["favorite_team"], "favorite_team"),
        spread_magnitude=_optional_float(raw["spread_magnitude"], "spread_magnitude"),
        favorite_won=_optional_bool(raw["favorite_won"], "favorite_won"),
        favorite_covered=_optional_bool(raw["favorite_covered"], "favorite_covered"),
        top_contributing_features=tuple(
            _contribution(item)
            for item in _list(raw["top_contributing_features"], "top_contributing_features")
        ),
    )


def _contribution(value: object) -> ComparableFeatureContribution:
    raw = _object(value, "contribution")
    _exact_keys(
        raw,
        {"feature_name", "query_value", "candidate_value", "squared_difference"},
        "Contribution",
    )
    return ComparableFeatureContribution(
        feature_name=_text(raw["feature_name"], "feature_name"),
        query_value=_float(raw["query_value"], "query_value"),
        candidate_value=_float(raw["candidate_value"], "candidate_value"),
        squared_difference=_float(raw["squared_difference"], "squared_difference"),
    )


def _feature_schema(value: object) -> PredictionFeatureSchema:
    raw = _object(value, "feature_schema")
    _exact_keys(
        raw,
        {
            "schema_id",
            "model_name",
            "model_type",
            "task",
            "modeling_schema_version",
            "epa_window",
            "feature_set_name",
            "ordered_columns",
        },
        "Feature schema",
    )
    return PredictionFeatureSchema(
        schema_id=_digest(_text(raw["schema_id"], "schema_id"), "schema_id"),
        model_name=_text(raw["model_name"], "model_name"),
        model_type=_text(raw["model_type"], "model_type"),
        task=_text(raw["task"], "task"),
        modeling_schema_version=_integer(raw["modeling_schema_version"], "modeling_schema_version"),
        epa_window=_integer(raw["epa_window"], "epa_window"),
        feature_set_name=_text(raw["feature_set_name"], "feature_set_name"),
        ordered_columns=tuple(
            _text(item, "feature column")
            for item in _list(raw["ordered_columns"], "ordered_columns")
        ),
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _exact_keys(raw: dict[str, object], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{label} keys do not match the current schema.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _float(value, label)


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean when present.")
    return value


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp string.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp string.") from exc


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text
