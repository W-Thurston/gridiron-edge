# src/gridiron_edge/deployment/collector_merge.py

"""Merge a staged copy of the worker's quote tree into the local one without losing evidence.

The local ``data/odds`` tree holds evidence the worker never saw (manual ingestion), so a
mirror that replaces differing files destroys it. Every file family is merged by its own
rule (``DECISIONS.md`` D78):

* weekly observation partitions: union of rows, never fewer rows than before;
* ``odds_current.parquet``: the snapshot with the later ``fetched_at`` wins;
* collection plans, run and rollover receipts: immutable, copied only when absent, a
  differing existing file is a conflict and is left untouched;
* ``collection_plans/current.json``: a pointer owned by each machine, never pulled;
* anything else in the staged tree is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

import pandas as pd

from gridiron_edge.ingest.odds.store import (
    merge_observation_partition,
    validate_quote_rows,
    write_current_odds_snapshot,
)

_CURRENT_SNAPSHOT = "odds_current.parquet"
_OBSERVATIONS = "observations.parquet"
_POINTER = "collection_plans/current.json"
_IMMUTABLE_ROOTS = ("collection_plans", "collection_runs", "collection_rollovers")


@dataclass(slots=True)
class MergeReport:
    """What one merge did (or, for a dry run, would do)."""

    copied: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    partitions: dict[str, tuple[int, int]] = field(default_factory=dict)
    current_snapshot: str = "absent"

    @property
    def rows_added(self) -> int:
        """Observation rows added across all partitions."""
        return sum(merged - local for local, merged in self.partitions.values())


def _latest_fetch(rows: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(rows["fetched_at"].max())


def _merge_current_snapshot(staged: Path, local: Path, *, repo: Path, write: bool) -> str:
    pulled = validate_quote_rows(pd.read_parquet(staged))
    if pulled.empty:
        return "pulled snapshot empty, kept local"
    if not local.exists():
        if write:
            write_current_odds_snapshot(pulled, repo=repo)
        return "taken from worker (no local snapshot)"
    existing = validate_quote_rows(pd.read_parquet(local))
    if existing.empty or _latest_fetch(pulled) > _latest_fetch(existing):
        if write:
            write_current_odds_snapshot(pulled, repo=repo)
        return "replaced by the worker's later snapshot"
    return "kept local (not older than the worker's)"


def merge_collector_tree(staged_odds: Path, local_odds: Path, *, write: bool = True) -> MergeReport:
    """Merge ``staged_odds`` into ``local_odds`` by the per-family rules above."""
    report = MergeReport()
    repo = local_odds.parent.parent
    for staged in sorted(path for path in staged_odds.rglob("*") if path.is_file()):
        relative = staged.relative_to(staged_odds)
        key = relative.as_posix()
        target = local_odds / relative
        if key == _POINTER or relative.name.startswith("."):
            continue
        if key == _CURRENT_SNAPSHOT:
            report.current_snapshot = _merge_current_snapshot(
                staged, target, repo=repo, write=write
            )
        elif relative.parts[0] == "history" and relative.name == _OBSERVATIONS:
            report.partitions[key] = merge_observation_partition(
                target, pd.read_parquet(staged), write=write
            )
        elif relative.parts[0] in _IMMUTABLE_ROOTS:
            if not target.exists():
                report.copied.append(key)
                if write:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staged, target)
            elif target.read_bytes() == staged.read_bytes():
                report.unchanged.append(key)
            else:
                report.conflicts.append(key)
    return report
