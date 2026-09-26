# src/gridiron_edge/evaluation/comparable_games_corpus_store.py
"""Immutable JSON plus Parquet storage for comparable-games corpora."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.comparable_games_corpus import (
    COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION,
    ComparableGamesCorpus,
    ComparableGamesFrameReference,
    comparable_games_corpus_payload,
    frame_content_digest,
    validate_comparable_games_corpus,
)
from gridiron_edge.evaluation.prediction_input_evidence import PredictionFeatureSchema

_STORE_DIRECTORY = "data/output/comparable_games_corpus"
_STORE_SCHEMA_VERSION = 1


def comparable_games_corpus_root(repo: Path | None = None) -> Path:
    """Return the canonical comparable-games corpus store root."""
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


def comparable_games_corpus_manifest_path(corpus_id: str, *, repo: Path | None = None) -> Path:
    """Return the canonical identity-addressed manifest path."""
    identity = _digest(corpus_id, "corpus_id")
    return (
        comparable_games_corpus_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "corpora"
        / f"{identity}.json"
    )


def comparable_games_corpus_frame_path(corpus_id: str, *, repo: Path | None = None) -> Path:
    """Return the canonical identity-addressed frame path."""
    identity = _digest(corpus_id, "corpus_id")
    return (
        comparable_games_corpus_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "frames"
        / f"{identity}.parquet"
    )


def write_comparable_games_corpus(
    corpus: ComparableGamesCorpus,
    *,
    frame: DataFrame,
    repo: Path | None = None,
) -> Path:
    """Persist one corpus's manifest and row frame, or accept an idempotent replay.

    Unlike ``write_logistic_explanation_batch``, this does not compare full
    encoded bytes on replay: ``corpus_id`` deliberately excludes
    ``generated_at`` (see ``comparable_games_corpus_id``), so a rebuild from
    unchanged model/scaler/frame inputs legitimately produces a manifest
    that differs only in that timestamp. The manifest path is itself
    content-addressed by ``corpus_id``, so an existing file at that exact
    path is definitionally a valid prior write of the same content; only
    its own embedded identity is re-checked, not a byte-for-byte replay of
    the newly-supplied (freshly-timestamped) payload.
    """
    validate_comparable_games_corpus(corpus)
    _validate_frame_matches(corpus.frame, frame)

    manifest_path = comparable_games_corpus_manifest_path(corpus.corpus_id, repo=repo)
    if manifest_path.exists():
        if _embedded_id(manifest_path, key="corpus_id") != corpus.corpus_id:
            raise ValueError(
                "Comparable-games corpus identity cannot be reused with different content."
            )
        return manifest_path

    frame_path = comparable_games_corpus_frame_path(corpus.corpus_id, repo=repo)
    _write_parquet(frame_path, frame)

    encoded = (
        json.dumps(
            {
                "store_schema_version": _STORE_SCHEMA_VERSION,
                "corpus_id": corpus.corpus_id,
                "corpus": comparable_games_corpus_payload(corpus),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        if not manifest_path.exists():
            temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def read_comparable_games_corpus(
    corpus_id: str,
    *,
    repo: Path | None = None,
) -> tuple[ComparableGamesCorpus, DataFrame]:
    """Read and strictly validate one corpus manifest plus its row frame."""
    manifest_path = comparable_games_corpus_manifest_path(corpus_id, repo=repo)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Comparable-games corpus manifest is missing: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Comparable-games corpus manifest contains malformed JSON: {manifest_path}"
        ) from exc
    raw = _object(value, "comparable-games corpus manifest")
    _exact_keys(raw, {"store_schema_version", "corpus_id", "corpus"}, "Artifact")
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != _STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported comparable-games corpus store schema version.")
    embedded_id = _digest(_text(raw["corpus_id"], "corpus_id"), "corpus_id")
    corpus = _corpus(raw["corpus"])
    if embedded_id != corpus.corpus_id:
        raise ValueError("Stored corpus identity does not match corpus content.")

    frame_path = comparable_games_corpus_frame_path(corpus.corpus_id, repo=repo)
    if not frame_path.exists():
        raise FileNotFoundError(f"Comparable-games corpus frame is missing: {frame_path}")
    frame = pd.read_parquet(frame_path)
    _validate_frame_matches(corpus.frame, frame)
    return corpus, frame


def find_latest_comparable_games_corpus(
    *,
    model_name: str,
    model_type: str,
    repo: Path | None = None,
) -> ComparableGamesCorpus | None:
    """Return the most recently generated corpus for one model identity, if any."""
    directory = comparable_games_corpus_root(repo) / f"schema={_STORE_SCHEMA_VERSION}" / "corpora"
    if not directory.exists():
        return None
    matches = [
        _manifest_only(path)
        for path in sorted(directory.glob("*.json"))
        if _manifest_only(path).model_name == model_name
        and _manifest_only(path).model_type == model_type
    ]
    if not matches:
        return None
    return max(matches, key=lambda corpus: corpus.generated_at)


def _manifest_only(path: Path) -> ComparableGamesCorpus:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = _object(value, "comparable-games corpus manifest")
    return _corpus(raw["corpus"])


def _validate_frame_matches(reference: ComparableGamesFrameReference, frame: DataFrame) -> None:
    if len(frame) != reference.row_count:
        raise ValueError("Comparable-games corpus frame row count does not match its manifest.")
    if tuple(frame.columns) != reference.columns:
        raise ValueError("Comparable-games corpus frame columns do not match its manifest.")
    digest = frame_content_digest(
        tuple(frame.columns), list(frame.itertuples(index=False, name=None))
    )
    if digest != reference.content_digest:
        raise ValueError(
            "Comparable-games corpus frame content digest does not match its manifest."
        )


def _write_parquet(path: Path, frame: DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(frame):
            raise ValueError(
                "Comparable-games corpus frame identity cannot be reused with different content."
            )
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        normalized = pd.read_parquet(temporary)
        if not normalized.equals(frame):
            raise ValueError("Serialized comparable-games corpus frame does not replay exactly.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus(value: object) -> ComparableGamesCorpus:
    raw = _object(value, "corpus")
    _exact_keys(
        raw,
        {
            "schema_version",
            "corpus_id",
            "model_name",
            "model_type",
            "model_content_digest",
            "scaler_content_digest",
            "feature_schema",
            "frame",
            "generated_at",
            "distance_threshold",
            "leave_one_out_median_distance",
            "leave_one_out_percentile",
        },
        "Corpus",
    )
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION:
        raise ValueError("Unsupported comparable-games corpus schema version.")
    corpus = ComparableGamesCorpus(
        schema_version=schema_version,
        corpus_id=_digest(_text(raw["corpus_id"], "corpus_id"), "corpus_id"),
        model_name=_text(raw["model_name"], "model_name"),
        model_type=_text(raw["model_type"], "model_type"),
        model_content_digest=_digest(
            _text(raw["model_content_digest"], "model_content_digest"), "model_content_digest"
        ),
        scaler_content_digest=_digest(
            _text(raw["scaler_content_digest"], "scaler_content_digest"), "scaler_content_digest"
        ),
        feature_schema=_feature_schema(raw["feature_schema"]),
        frame=_frame_reference(raw["frame"]),
        generated_at=_datetime(raw["generated_at"], "generated_at"),
        distance_threshold=_float(raw["distance_threshold"], "distance_threshold"),
        leave_one_out_median_distance=_float(
            raw["leave_one_out_median_distance"], "leave_one_out_median_distance"
        ),
        leave_one_out_percentile=_float(
            raw["leave_one_out_percentile"], "leave_one_out_percentile"
        ),
    )
    validate_comparable_games_corpus(corpus)
    return corpus


def _frame_reference(value: object) -> ComparableGamesFrameReference:
    raw = _object(value, "frame")
    _exact_keys(raw, {"artifact", "row_count", "columns", "content_digest"}, "Frame")
    return ComparableGamesFrameReference(
        artifact=_text(raw["artifact"], "artifact"),
        row_count=_integer(raw["row_count"], "row_count"),
        columns=tuple(_text(item, "column") for item in _list(raw["columns"], "columns")),
        content_digest=_digest(_text(raw["content_digest"], "content_digest"), "content_digest"),
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


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


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
