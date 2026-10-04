# src/gridiron_edge/weekly_cycle/receipts.py

"""Immutable receipts of every weekly-cycle decision; the receipts are the cycle's state."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

SCHEMA_VERSION: Final[int] = 1

ATTEMPT_CLOSEOUT: Final[str] = "closeout"
ATTEMPT_FORECAST: Final[str] = "forecast"
ATTEMPT_WATCH: Final[str] = "watch"
ATTEMPT_REFRESH: Final[str] = "refresh"
ATTEMPT_EDGES: Final[str] = "edges"

OUTCOME_DONE: Final[str] = "done"
OUTCOME_WAITING: Final[str] = "waiting"
OUTCOME_BLOCKED: Final[str] = "blocked"
OUTCOME_FAILED: Final[str] = "failed"
OUTCOME_UNCHANGED: Final[str] = "unchanged"
OUTCOME_CHANGED: Final[str] = "changed"

#: Outcomes of a step that did not finish; the step is retried after an hour.
UNFINISHED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {OUTCOME_WAITING, OUTCOME_BLOCKED, OUTCOME_FAILED}
)


@dataclass(frozen=True, slots=True)
class CycleReceipt:
    """One weekly-cycle attempt, what it looked at and what came of it.

    ``season`` and ``week`` name the week the attempt acted on: the finished week for a
    closeout, the target week for everything else. ``inputs`` and ``results`` hold the
    attempt-specific evidence (see the watch and the executors).
    """

    decided_at: str
    season: str
    week: int
    attempted: str
    outcome: str
    detail: str
    inputs: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def decided(self) -> datetime:
        """The decision time as an aware UTC datetime."""
        return datetime.fromisoformat(self.decided_at).astimezone(UTC)


def week_directory(repo: Path, *, season: str, week: int) -> Path:
    """Return the receipt directory of one season week."""
    return repo / "data" / "output" / "week_cycle" / f"season={season}" / f"week={week:02d}"


def write_receipt(repo: Path, receipt: CycleReceipt) -> Path:
    """Persist a receipt as a new file; an existing receipt is never replaced."""
    directory = week_directory(repo, season=receipt.season, week=receipt.week)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = receipt.decided.strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}_{receipt.attempted}_{receipt.outcome}.json"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_receipts(repo: Path, *, season: str, week: int) -> list[CycleReceipt]:
    """Return one week's receipts oldest first; an unreadable file is an error, not skipped."""
    directory = week_directory(repo, season=season, week=week)
    if not directory.is_dir():
        return []
    receipts: list[CycleReceipt] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            receipts.append(CycleReceipt(**payload))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unreadable weekly-cycle receipt {path}: {exc}") from exc
    return receipts


def last_receipt(
    receipts: list[CycleReceipt],
    *,
    attempted: Collection[str],
    outcomes: Collection[str] | None = None,
) -> CycleReceipt | None:
    """Return the newest receipt of the given attempts (and outcomes), or ``None``."""
    matching = [
        receipt
        for receipt in receipts
        if receipt.attempted in attempted and (outcomes is None or receipt.outcome in outcomes)
    ]
    return max(matching, key=lambda receipt: receipt.decided, default=None)
