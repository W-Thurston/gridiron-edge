# src/gridiron_edge/evaluation/weekly_recap_store.py

"""Immutable JSON plus Parquet storage and explicit selection for weekly and season recaps.

Layout under the registered ``weekly_recaps`` root (``DECISIONS.md`` D86), where N is the weekly
recap schema version::

    schema=N/recaps/{recap_id}.json          weekly manifest
    schema=N/games/{content_digest}.parquet  weekly game frame
    schema=N/seasons/{season_recap_id}.json  season manifest (rolled up from the weekly recaps)
    current.json                             one weekly selection per season and week
    current_seasons.json                     one season selection per season

A rerun on unchanged inputs replays to the same identity and keeps every stored file byte for
byte; a reused identity with different content is rejected. Loading verifies the manifest's
identity and the frame's digest; it never recomputes the recap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.evaluation.game_model_evaluation_report import frame_content_digest
from gridiron_edge.evaluation.weekly_recap import (
    WEEKLY_RECAP_SCHEMA_VERSION,
    BuiltWeeklyRecap,
    WeeklyRecap,
    validate_weekly_recap,
    weekly_recap_from_payload,
    weekly_recap_payload,
)
from gridiron_edge.evaluation.weekly_recap_season import (
    SeasonRecap,
    season_recap_from_payload,
    season_recap_payload,
    validate_season_recap,
)

_CURRENT_FILENAME: Final[str] = "current.json"
_CURRENT_SCHEMA_VERSION: Final[int] = 1
_CURRENT_SEASONS_FILENAME: Final[str] = "current_seasons.json"
_CURRENT_SEASONS_SCHEMA_VERSION: Final[int] = 1


class RecapWriteOutcome(StrEnum):
    """What writing a recap did to the store."""

    WRITTEN = "written"
    REPLAYED = "replayed"


@dataclass(frozen=True, slots=True)
class WeeklyRecapSelection:
    """The explicitly selected recap of one season and week."""

    season: str
    week: int
    recap_id: str
    selected_at: datetime


@dataclass(frozen=True, slots=True)
class SeasonRecapSelection:
    """The explicitly selected season recap of one season."""

    season: str
    season_recap_id: str
    selected_at: datetime


def weekly_recap_root(repo: Path | None = None) -> Path:
    """Return the registered weekly-recap storage root."""
    return dataset_path(repo or get_settings().repo_root, "weekly_recaps")


def weekly_recap_manifest_path(recap_id: str, *, repo: Path | None = None) -> Path:
    """Return the identity-addressed manifest path of one recap."""
    return (
        weekly_recap_root(repo)
        / f"schema={WEEKLY_RECAP_SCHEMA_VERSION}"
        / "recaps"
        / f"{_digest(recap_id)}.json"
    )


def write_weekly_recap(
    built: BuiltWeeklyRecap,
    *,
    repo: Path | None = None,
) -> RecapWriteOutcome:
    """Persist the frame and manifest, or accept an exact replay of a stored recap."""
    recap = built.recap
    validate_weekly_recap(recap)
    _validate_frame(recap, built.games)
    manifest = weekly_recap_manifest_path(recap.recap_id, repo=repo)
    if manifest.exists():
        stored = load_weekly_recap(recap.recap_id, repo=repo)[0]
        if _without_built_at(stored) != _without_built_at(recap):
            raise ValueError("Weekly recap identity cannot be reused with different content.")
        return RecapWriteOutcome.REPLAYED
    _write_parquet(_resolved(weekly_recap_root(repo), recap.games.artifact), built.games)
    encoded = json.dumps(
        weekly_recap_payload(recap), indent=2, sort_keys=True, allow_nan=False
    ).encode()
    _atomic_write(manifest, encoded + b"\n")
    return RecapWriteOutcome.WRITTEN


def load_weekly_recap(
    recap_id: str,
    *,
    repo: Path | None = None,
) -> tuple[WeeklyRecap, DataFrame]:
    """Load one stored recap and its game frame, verifying identity and frame digest."""
    path = weekly_recap_manifest_path(recap_id, repo=repo)
    if not path.is_file():
        raise FileNotFoundError(f"Weekly recap is not stored: recap_id={recap_id!r}.")
    recap = weekly_recap_from_payload(json.loads(path.read_text(encoding="utf-8")))
    if recap.recap_id != recap_id:
        raise ValueError("Weekly recap manifest is stored under another identity.")
    frame_path = _resolved(weekly_recap_root(repo), recap.games.artifact)
    if not frame_path.is_file():
        raise FileNotFoundError(f"Weekly recap game frame is missing: {frame_path}")
    games = pd.read_parquet(frame_path)
    _validate_frame(recap, games)
    return recap, games


def select_current_weekly_recap(
    recap_id: str,
    *,
    selected_at: datetime,
    repo: Path | None = None,
) -> WeeklyRecapSelection:
    """Explicitly select one stored recap for its season and week."""
    _utc(selected_at)
    recap, _ = load_weekly_recap(recap_id, repo=repo)
    current = _read_current(repo)
    current["selections"][_selection_key(recap.season, recap.week)] = {
        "season": recap.season,
        "week": recap.week,
        "recap_id": recap.recap_id,
        "selected_at": selected_at.isoformat(),
    }
    encoded = json.dumps(current, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(_current_path(repo), encoded)
    return WeeklyRecapSelection(recap.season, recap.week, recap.recap_id, selected_at)


def get_current_weekly_recap_selection(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> WeeklyRecapSelection | None:
    """Return the selection of one season and week, or ``None`` when nothing is selected."""
    key = _selection_key(season, week)
    payload = _read_current(repo)["selections"].get(key)
    if payload is None:
        return None
    return _selection_from_payload(key, payload)


def list_current_weekly_recap_selections(
    *,
    repo: Path | None = None,
) -> tuple[WeeklyRecapSelection, ...]:
    """Return every explicit selection, ordered by season and week (never by build time)."""
    selections = [
        _selection_from_payload(key, payload)
        for key, payload in _read_current(repo)["selections"].items()
    ]
    return tuple(sorted(selections, key=lambda selection: (selection.season, selection.week)))


def load_current_weekly_recap(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> tuple[WeeklyRecap, DataFrame]:
    """Load the explicitly selected recap of one season and week."""
    selection = get_current_weekly_recap_selection(season=season, week=week, repo=repo)
    if selection is None:
        raise FileNotFoundError(f"No weekly recap is selected for {season} week {week}.")
    recap, games = load_weekly_recap(selection.recap_id, repo=repo)
    if recap.season != season or recap.week != week:
        raise ValueError("Selected weekly recap covers another season or week.")
    return recap, games


def season_recap_manifest_path(season_recap_id: str, *, repo: Path | None = None) -> Path:
    """Return the identity-addressed manifest path of one season recap."""
    return (
        weekly_recap_root(repo)
        / f"schema={WEEKLY_RECAP_SCHEMA_VERSION}"
        / "seasons"
        / f"{_digest(season_recap_id)}.json"
    )


def write_season_recap(recap: SeasonRecap, *, repo: Path | None = None) -> RecapWriteOutcome:
    """Persist the season manifest, or accept an exact replay of a stored one."""
    validate_season_recap(recap)
    manifest = season_recap_manifest_path(recap.season_recap_id, repo=repo)
    if manifest.exists():
        stored = load_season_recap(recap.season_recap_id, repo=repo)
        if _season_without_built_at(stored) != _season_without_built_at(recap):
            raise ValueError("Season recap identity cannot be reused with different content.")
        return RecapWriteOutcome.REPLAYED
    encoded = json.dumps(
        season_recap_payload(recap), indent=2, sort_keys=True, allow_nan=False
    ).encode()
    _atomic_write(manifest, encoded + b"\n")
    return RecapWriteOutcome.WRITTEN


def load_season_recap(season_recap_id: str, *, repo: Path | None = None) -> SeasonRecap:
    """Load one stored season recap, verifying its identity."""
    path = season_recap_manifest_path(season_recap_id, repo=repo)
    if not path.is_file():
        raise FileNotFoundError(f"Season recap is not stored: season_recap_id={season_recap_id!r}.")
    recap = season_recap_from_payload(json.loads(path.read_text(encoding="utf-8")))
    if recap.season_recap_id != season_recap_id:
        raise ValueError("Season recap manifest is stored under another identity.")
    return recap


def select_current_season_recap(
    season_recap_id: str,
    *,
    selected_at: datetime,
    repo: Path | None = None,
) -> SeasonRecapSelection:
    """Explicitly select one stored season recap for its season."""
    _utc(selected_at)
    recap = load_season_recap(season_recap_id, repo=repo)
    current = _read_current_seasons(repo)
    current["selections"][recap.season] = {
        "season": recap.season,
        "season_recap_id": recap.season_recap_id,
        "selected_at": selected_at.isoformat(),
    }
    encoded = json.dumps(current, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(weekly_recap_root(repo) / _CURRENT_SEASONS_FILENAME, encoded)
    return SeasonRecapSelection(recap.season, recap.season_recap_id, selected_at)


def get_current_season_recap_selection(
    *,
    season: str,
    repo: Path | None = None,
) -> SeasonRecapSelection | None:
    """Return the season recap selection of one season, or ``None`` when nothing is selected."""
    if not season.strip():
        raise ValueError("A season recap selection needs a season.")
    payload = _read_current_seasons(repo)["selections"].get(season)
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "season",
        "season_recap_id",
        "selected_at",
    }:
        raise ValueError("Season recap selection does not match the schema.")
    if payload["season"] != season:
        raise ValueError("Season recap selection is stored under another season.")
    selected_at = datetime.fromisoformat(str(payload["selected_at"]))
    _utc(selected_at)
    return SeasonRecapSelection(season, _digest(str(payload["season_recap_id"])), selected_at)


def load_current_season_recap(*, season: str, repo: Path | None = None) -> SeasonRecap:
    """Load the explicitly selected season recap of one season."""
    selection = get_current_season_recap_selection(season=season, repo=repo)
    if selection is None:
        raise FileNotFoundError(f"No season recap is selected for {season}.")
    recap = load_season_recap(selection.season_recap_id, repo=repo)
    if recap.season != season:
        raise ValueError("Selected season recap covers another season.")
    return recap


def _season_without_built_at(recap: SeasonRecap) -> dict[str, Any]:
    payload = season_recap_payload(recap)
    del payload["built_at"]
    return payload


def _read_current_seasons(repo: Path | None = None) -> dict[str, Any]:
    path = weekly_recap_root(repo) / _CURRENT_SEASONS_FILENAME
    if not path.exists():
        return {"schema_version": _CURRENT_SEASONS_SCHEMA_VERSION, "selections": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != _CURRENT_SEASONS_SCHEMA_VERSION:
        raise ValueError("Unsupported season recap current-selection schema.")
    if not isinstance(raw.get("selections"), dict):
        raise ValueError("Season recap current 'selections' must be an object.")
    return raw


def _without_built_at(recap: WeeklyRecap) -> dict[str, Any]:
    payload = weekly_recap_payload(recap)
    del payload["built_at"]
    return payload


def _validate_frame(recap: WeeklyRecap, frame: DataFrame) -> None:
    reference = recap.games
    if len(frame) != reference.row_count:
        raise ValueError("Weekly recap game frame row count does not match the manifest.")
    if tuple(frame.columns) != reference.columns:
        raise ValueError("Weekly recap game frame columns do not match the manifest.")
    if frame_content_digest(frame) != reference.content_digest:
        raise ValueError("Weekly recap game frame content digest does not match the manifest.")


def _write_parquet(path: Path, frame: DataFrame) -> None:
    """Write a content-addressed frame once; an existing file must hold the same content."""
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise ValueError("Weekly recap game frame identity cannot be reused.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        if not pd.read_parquet(temporary).equals(frame):
            raise ValueError("Serialized weekly recap game frame does not replay exactly.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved(root: Path, artifact: str) -> Path:
    resolved_root = root.resolve()
    path = (root / artifact).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Weekly recap artifact escapes the storage root.") from exc
    return path


def _current_path(repo: Path | None = None) -> Path:
    return weekly_recap_root(repo) / _CURRENT_FILENAME


def _read_current(repo: Path | None = None) -> dict[str, Any]:
    path = _current_path(repo)
    if not path.exists():
        return {"schema_version": _CURRENT_SCHEMA_VERSION, "selections": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != _CURRENT_SCHEMA_VERSION:
        raise ValueError("Unsupported weekly recap current-selection schema.")
    if not isinstance(raw.get("selections"), dict):
        raise ValueError("Weekly recap current 'selections' must be an object.")
    return raw


def _selection_from_payload(key: str, payload: object) -> WeeklyRecapSelection:
    if not isinstance(payload, dict) or set(payload) != {
        "season",
        "week",
        "recap_id",
        "selected_at",
    }:
        raise ValueError("Weekly recap selection does not match the schema.")
    season, week = payload["season"], payload["week"]
    if not isinstance(season, str) or not isinstance(week, int) or isinstance(week, bool):
        raise ValueError("Weekly recap selection needs a string season and an integer week.")
    if _selection_key(season, week) != key:
        raise ValueError("Weekly recap selection is stored under another season or week.")
    selected_at = datetime.fromisoformat(str(payload["selected_at"]))
    _utc(selected_at)
    return WeeklyRecapSelection(season, week, _digest(str(payload["recap_id"])), selected_at)


def _selection_key(season: str, week: int) -> str:
    if not season.strip() or week < 1:
        raise ValueError("A weekly recap selection needs a season and a week of at least 1.")
    return f"{season}_week_{week:02d}"


def _utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("selected_at must be timezone-aware UTC.")


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("A recap id must be a lowercase SHA-256 digest.")
    return value
