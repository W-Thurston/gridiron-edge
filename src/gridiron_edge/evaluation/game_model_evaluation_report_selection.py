# src/gridiron_edge/evaluation/game_model_evaluation_report_selection.py

"""Explicit current selection for immutable game-model evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.game_model_evaluation_report import (
    GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION,
)

GAME_MODEL_EVALUATION_REPORT_SELECTION_SCHEMA_VERSION: Final[int] = 1
_CURRENT_FILENAME: Final[str] = "current.json"


@dataclass(frozen=True, slots=True)
class GameModelEvaluationReportSelection:
    """Explicit current report identity and selection timestamp."""

    schema_version: int
    report_id: str
    selected_at: datetime


def game_model_evaluation_report_root(repo: Path | None = None) -> Path:
    """Return the canonical game-model evaluation report root."""
    root = repo or get_settings().repo_root
    return root / "data/output/game_model_evaluation"


def game_model_evaluation_report_path(
    report_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the canonical identity-addressed report manifest path."""
    identity = _digest(report_id, "report_id")
    return (
        game_model_evaluation_report_root(repo)
        / f"schema={GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION}"
        / "reports"
        / f"{identity}.json"
    )


def select_current_game_model_evaluation_report(
    report_id: str,
    *,
    selected_at: datetime,
    repo: Path | None = None,
) -> GameModelEvaluationReportSelection:
    """Explicitly select one existing immutable report as current."""
    selection = GameModelEvaluationReportSelection(
        schema_version=GAME_MODEL_EVALUATION_REPORT_SELECTION_SCHEMA_VERSION,
        report_id=_digest(report_id, "report_id"),
        selected_at=_utc(selected_at, "selected_at"),
    )
    report_path = game_model_evaluation_report_path(selection.report_id, repo=repo)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Game-model evaluation report is not stored: report_id={selection.report_id!r}."
        )
    path = _current_path(repo)
    encoded = (
        json.dumps(
            {
                "schema_version": selection.schema_version,
                "report_id": selection.report_id,
                "selected_at": selection.selected_at.isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return selection


def get_current_game_model_evaluation_report_selection(
    *,
    repo: Path | None = None,
) -> GameModelEvaluationReportSelection:
    """Read and validate the explicitly selected current report."""
    path = _current_path(repo)
    if not path.is_file():
        raise FileNotFoundError("No current game-model evaluation report is selected.")
    raw = _object(json.loads(path.read_text(encoding="utf-8")), "current selection")
    expected = {"schema_version", "report_id", "selected_at"}
    if set(raw) != expected:
        raise ValueError("Game-model evaluation current selection keys do not match schema.")
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != GAME_MODEL_EVALUATION_REPORT_SELECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported game-model evaluation report selection schema version.")
    selection = GameModelEvaluationReportSelection(
        schema_version=schema_version,
        report_id=_digest(_text(raw["report_id"], "report_id"), "report_id"),
        selected_at=_utc_string(raw["selected_at"], "selected_at"),
    )
    report_path = game_model_evaluation_report_path(selection.report_id, repo=repo)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Selected game-model evaluation report is missing: report_id={selection.report_id!r}."
        )
    return selection


def _current_path(repo: Path | None = None) -> Path:
    return game_model_evaluation_report_root(repo) / _CURRENT_FILENAME


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _utc_string(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp string.")
    return _utc(datetime.fromisoformat(value), label)


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value


def _digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value
