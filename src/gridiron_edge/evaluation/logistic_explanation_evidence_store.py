# src/gridiron_edge/evaluation/logistic_explanation_evidence_store.py
"""Immutable JSON persistence for Logistic explanation evidence batches."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION,
    LogisticExplanationBatch,
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    logistic_explanation_batch_payload,
    validate_logistic_explanation_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import PredictionFeatureSchema

_STORE_DIRECTORY = "data/output/logistic_explanations"
_STORE_SCHEMA_VERSION = 1


class AmbiguousLogisticExplanationError(ValueError):
    """More than one persisted batch claims the same event.

    The store is immutable and create-only with no "current" selection
    concept (see `list_logistic_explanation_batches`). `batch_id` is
    idempotent per (run, evidence) pair as of D56, so this no longer
    happens from re-running `explain-logistic` against unchanged evidence
    — it now specifically indicates a genuine data-provenance conflict
    (for example two distinct evidence generations claiming the same
    event) requiring deliberate resolution. Distinct from a
    malformed-artifact `ValueError`, which callers should treat as store
    corruption.
    """


def logistic_explanation_root(repo: Path | None = None) -> Path:
    """Return the canonical Logistic explanation evidence store root."""
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


def logistic_explanation_batch_path(
    batch_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the canonical identity-addressed batch path."""
    identity = _digest(batch_id, "batch_id")
    return (
        logistic_explanation_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "batches"
        / f"{identity}.json"
    )


def write_logistic_explanation_batch(
    batch: LogisticExplanationBatch,
    *,
    repo: Path | None = None,
) -> Path:
    """Create one immutable explanation batch artifact or accept an idempotent replay.

    ``batch_id`` deliberately excludes ``generated_at`` (D56), so re-running
    `explain-logistic` against unchanged evidence legitimately produces a
    payload that differs only in that timestamp. This does not compare full
    encoded bytes on replay: an existing file at this identity-addressed
    path is definitionally a valid prior write of the same content; only
    its own embedded identity is re-checked.
    """
    validate_logistic_explanation_batch(batch)
    path = logistic_explanation_batch_path(batch.batch_id, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _embedded_id(path, key="batch_id") != batch.batch_id:
            raise ValueError(
                "Logistic explanation batch identity cannot be reused with different content."
            )
        return path

    encoded = (
        json.dumps(
            {
                "store_schema_version": _STORE_SCHEMA_VERSION,
                "batch_id": batch.batch_id,
                "batch": logistic_explanation_batch_payload(batch),
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
                    "Logistic explanation batch identity cannot be reused with different content."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_logistic_explanation_batch(path: Path) -> LogisticExplanationBatch:
    """Read and strictly validate one exact immutable explanation batch artifact."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Logistic explanation batch contains malformed JSON: {path}") from exc
    raw = _object(value, "logistic explanation batch artifact")
    _exact_keys(raw, {"store_schema_version", "batch_id", "batch"}, "Artifact")
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != _STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported logistic explanation evidence store schema version.")
    embedded_id = _digest(_text(raw["batch_id"], "batch_id"), "batch_id")
    batch = _batch(raw["batch"])
    if embedded_id != batch.batch_id:
        raise ValueError("Stored batch identity does not match batch content.")
    expected = logistic_explanation_batch_path(embedded_id, repo=_artifact_repo(path))
    if path.resolve() != expected.resolve():
        raise ValueError("Logistic explanation batch path and embedded identity disagree.")
    return batch


def list_logistic_explanation_batches(
    *,
    repo: Path | None = None,
) -> tuple[LogisticExplanationBatch, ...]:
    """Scan and deterministically list every persisted explanation batch."""
    return _all_batches(repo=repo)


def find_logistic_explanation_by_event(
    event_id: str,
    *,
    repo: Path | None = None,
) -> LogisticExplanationEvent | None:
    """Scan immutable batches and resolve at most one explanation for an event."""
    normalized = _text(event_id, "event_id")
    matches: list[LogisticExplanationEvent] = []
    for batch in _all_batches(repo=repo):
        for event in batch.events:
            if event.event_id == normalized:
                matches.append(event)
    if len(matches) > 1:
        raise AmbiguousLogisticExplanationError(
            f"Multiple logistic explanation batches claim event {normalized!r}."
        )
    return matches[0] if matches else None


def _all_batches(*, repo: Path | None) -> tuple[LogisticExplanationBatch, ...]:
    directory = logistic_explanation_root(repo) / f"schema={_STORE_SCHEMA_VERSION}" / "batches"
    if not directory.exists():
        return ()
    values = tuple(
        read_logistic_explanation_batch(path) for path in sorted(directory.glob("*.json"))
    )
    return tuple(sorted(values, key=lambda value: value.batch_id))


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = tuple(Path(_STORE_DIRECTORY).parts)
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Logistic explanation batch path is outside the canonical store.")


def _batch(value: object) -> LogisticExplanationBatch:
    raw = _object(value, "batch")
    _exact_keys(
        raw,
        {
            "schema_version",
            "batch_id",
            "run_id",
            "evidence_id",
            "model_content_digest",
            "scaler_content_digest",
            "feature_schema",
            "generated_at",
            "events",
        },
        "Batch",
    )
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported logistic explanation evidence schema version.")
    batch = LogisticExplanationBatch(
        schema_version=schema_version,
        batch_id=_digest(_text(raw["batch_id"], "batch_id"), "batch_id"),
        run_id=_text(raw["run_id"], "run_id"),
        evidence_id=_digest(_text(raw["evidence_id"], "evidence_id"), "evidence_id"),
        model_content_digest=_digest(
            _text(raw["model_content_digest"], "model_content_digest"), "model_content_digest"
        ),
        scaler_content_digest=_digest(
            _text(raw["scaler_content_digest"], "scaler_content_digest"), "scaler_content_digest"
        ),
        feature_schema=_feature_schema(raw["feature_schema"]),
        generated_at=_datetime(raw["generated_at"], "generated_at"),
        events=tuple(_event(item) for item in _list(raw["events"], "events")),
    )
    validate_logistic_explanation_batch(batch)
    return batch


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


def _event(value: object) -> LogisticExplanationEvent:
    raw = _object(value, "event")
    _exact_keys(
        raw,
        {
            "event_id",
            "game_id",
            "intercept",
            "contributions",
            "reconstructed_log_odds",
            "reconstructed_probability",
            "raw_estimator_output",
            "tolerance",
        },
        "Event",
    )
    return LogisticExplanationEvent(
        event_id=_text(raw["event_id"], "event_id"),
        game_id=_text(raw["game_id"], "game_id"),
        intercept=_float(raw["intercept"], "intercept"),
        contributions=tuple(
            _contribution(item) for item in _list(raw["contributions"], "contributions")
        ),
        reconstructed_log_odds=_float(raw["reconstructed_log_odds"], "reconstructed_log_odds"),
        reconstructed_probability=_float(
            raw["reconstructed_probability"], "reconstructed_probability"
        ),
        raw_estimator_output=_float(raw["raw_estimator_output"], "raw_estimator_output"),
        tolerance=_float(raw["tolerance"], "tolerance"),
    )


def _contribution(value: object) -> LogisticFeatureContribution:
    raw = _object(value, "contribution")
    _exact_keys(
        raw,
        {"feature_name", "transformed_value", "coefficient", "contribution"},
        "Contribution",
    )
    return LogisticFeatureContribution(
        feature_name=_text(raw["feature_name"], "feature_name"),
        transformed_value=_float(raw["transformed_value"], "transformed_value"),
        coefficient=_float(raw["coefficient"], "coefficient"),
        contribution=_float(raw["contribution"], "contribution"),
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
