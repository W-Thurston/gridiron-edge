# src/gridiron_edge/evaluation/forecast_evidence_disposition_store.py
"""Immutable JSON persistence for forecast-evidence dispositions."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
import json
import os
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.forecast_evidence_disposition import (
    AffectedPredictionComponent,
    ForecastEvidenceDefectReason,
    ForecastEvidenceDisposition,
    ForecastEvidenceDispositionStatus,
    ForecastEvidencePublicationEffect,
    ForecastEvidenceReplacementPolicy,
    validate_forecast_evidence_disposition,
)

FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION: Final[int] = 1
_STORE_DIRECTORY: Final[str] = "data/output/forecast_evidence_dispositions"


def forecast_evidence_disposition_root(
    repo: Path | None = None,
) -> Path:
    """Return the immutable forecast-evidence disposition root."""
    root = repo or get_settings().repo_root
    return root / _STORE_DIRECTORY


def forecast_evidence_disposition_path(
    disposition_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the identity-addressed path for one disposition."""
    identity = _digest(disposition_id, "disposition_id")
    return (
        forecast_evidence_disposition_root(repo)
        / f"schema={FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION}"
        / "dispositions"
        / f"{identity}.json"
    )


def write_forecast_evidence_disposition(
    disposition: ForecastEvidenceDisposition,
    *,
    repo: Path | None = None,
) -> Path:
    """Persist one immutable disposition or accept an exact replay."""
    validate_forecast_evidence_disposition(disposition)
    path = forecast_evidence_disposition_path(disposition.disposition_id, repo=repo)
    encoded = (
        json.dumps(
            _artifact_payload(disposition),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                "Forecast-evidence disposition identity cannot be reused with different content."
            )
        return path

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        temporary.write_text(encoded, encoding="utf-8")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    "Forecast-evidence disposition identity"
                    " cannot be reused with different content."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)

    return path


def read_forecast_evidence_disposition(
    path: Path,
) -> ForecastEvidenceDisposition:
    """Read and strictly validate one exact disposition artifact."""
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Forecast-evidence disposition contains malformed JSON: {path}") from exc

    raw = _object(raw_value, "disposition artifact")
    _exact_keys(
        raw,
        {"store_schema_version", "disposition_id", "disposition"},
        "Disposition artifact",
    )
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported forecast-evidence disposition store schema version.")
    embedded_id = _digest(
        _text(raw["disposition_id"], "disposition_id"),
        "disposition_id",
    )
    disposition = _disposition(raw["disposition"])
    if embedded_id != disposition.disposition_id:
        raise ValueError("Stored disposition identity does not match disposition content.")
    expected = forecast_evidence_disposition_path(
        embedded_id,
        repo=_artifact_repo(path),
    )
    if path.resolve() != expected.resolve():
        raise ValueError("Forecast-evidence disposition path and embedded identity disagree.")
    return disposition


def list_forecast_evidence_dispositions(
    *,
    season: str | None = None,
    week: int | None = None,
    product_id: str | None = None,
    repo: Path | None = None,
) -> tuple[ForecastEvidenceDisposition, ...]:
    """List stored dispositions with optional exact scope and product filters."""
    if season is not None and not season.strip():
        raise ValueError("season must not be empty.")
    if week is not None and (isinstance(week, bool) or not isinstance(week, int) or week < 1):
        raise ValueError("week must be a positive integer.")
    normalized_product_id = None
    if product_id is not None:
        normalized_product_id = _text(product_id, "product_id")

    directory = (
        forecast_evidence_disposition_root(repo)
        / f"schema={FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION}"
        / "dispositions"
    )
    if not directory.exists():
        return ()

    records = tuple(
        read_forecast_evidence_disposition(path) for path in sorted(directory.glob("*.json"))
    )
    filtered = records
    if season is not None:
        filtered = tuple(value for value in filtered if value.season == season)
    if week is not None:
        filtered = tuple(value for value in filtered if value.week == week)
    if normalized_product_id is not None:
        filtered = tuple(
            value for value in filtered if normalized_product_id in value.affected_product_ids
        )
    return tuple(sorted(filtered, key=lambda value: value.disposition_id))


def _artifact_payload(
    disposition: ForecastEvidenceDisposition,
) -> dict[str, object]:
    return {
        "store_schema_version": FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION,
        "disposition_id": disposition.disposition_id,
        "disposition": _disposition_payload(disposition),
    }


def _disposition_payload(
    disposition: ForecastEvidenceDisposition,
) -> dict[str, object]:
    return {
        "schema_version": disposition.schema_version,
        "disposition_id": disposition.disposition_id,
        "recorded_at": disposition.recorded_at.isoformat(),
        "season": disposition.season,
        "week": disposition.week,
        "status": disposition.status.value,
        "reason": disposition.reason.value,
        "publication_effect": disposition.publication_effect.value,
        "replacement_policy": disposition.replacement_policy.value,
        "affected_components": [value.value for value in disposition.affected_components],
        "affected_run_ids": list(disposition.affected_run_ids),
        "affected_event_ids": list(disposition.affected_event_ids),
        "affected_product_ids": list(disposition.affected_product_ids),
        "selected_affected_product_id": disposition.selected_affected_product_id,
        "decision_references": list(disposition.decision_references),
        "evidence_summary": disposition.evidence_summary,
    }


def _disposition(value: object) -> ForecastEvidenceDisposition:
    raw = _object(value, "disposition")
    _exact_keys(
        raw,
        {
            "schema_version",
            "disposition_id",
            "recorded_at",
            "season",
            "week",
            "status",
            "reason",
            "publication_effect",
            "replacement_policy",
            "affected_components",
            "affected_run_ids",
            "affected_event_ids",
            "affected_product_ids",
            "selected_affected_product_id",
            "decision_references",
            "evidence_summary",
        },
        "Disposition",
    )
    disposition = ForecastEvidenceDisposition(
        schema_version=_integer(raw["schema_version"], "schema_version"),
        disposition_id=_digest(
            _text(raw["disposition_id"], "disposition_id"),
            "disposition_id",
        ),
        recorded_at=_datetime(raw["recorded_at"], "recorded_at"),
        season=_text(raw["season"], "season"),
        week=_integer(raw["week"], "week"),
        status=_enum(raw["status"], ForecastEvidenceDispositionStatus, "status"),
        reason=_enum(raw["reason"], ForecastEvidenceDefectReason, "reason"),
        publication_effect=_enum(
            raw["publication_effect"],
            ForecastEvidencePublicationEffect,
            "publication_effect",
        ),
        replacement_policy=_enum(
            raw["replacement_policy"],
            ForecastEvidenceReplacementPolicy,
            "replacement_policy",
        ),
        affected_components=tuple(
            _enum(item, AffectedPredictionComponent, "affected_component")
            for item in _list(raw["affected_components"], "affected_components")
        ),
        affected_run_ids=_text_tuple(raw["affected_run_ids"], "affected_run_ids"),
        affected_event_ids=_text_tuple(raw["affected_event_ids"], "affected_event_ids"),
        affected_product_ids=_text_tuple(raw["affected_product_ids"], "affected_product_ids"),
        selected_affected_product_id=_text(
            raw["selected_affected_product_id"],
            "selected_affected_product_id",
        ),
        decision_references=_text_tuple(raw["decision_references"], "decision_references"),
        evidence_summary=_text(raw["evidence_summary"], "evidence_summary"),
    )
    validate_forecast_evidence_disposition(disposition)
    return disposition


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = tuple(Path(_STORE_DIRECTORY).parts)
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Forecast-evidence disposition path is outside the canonical store.")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, label) for item in _list(value, label))


def _exact_keys(
    raw: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
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


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp string.")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp string.") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return result


def _digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _enum[T: StrEnum](
    value: object,
    enum_type: type[T],
    label: str,
) -> T:
    text = _text(value, label)
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ValueError(f"{label} contains an unsupported value: {text!r}.") from exc
