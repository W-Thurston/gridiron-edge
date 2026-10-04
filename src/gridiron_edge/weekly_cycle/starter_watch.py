# src/gridiron_edge/weekly_cycle/starter_watch.py

"""Compare the starters the resolver projects now with those a weekly product recorded.

The watch reads the cached depth charts and injury reports and the same pure resolver that
feeds the forecast (``DECISIONS.md`` D75, D77), so a difference here is a difference the next
forecast would act on. Games that have kicked off are never compared (``DECISIONS.md`` D82).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.team.projected_starter import latest_snapshot_chart, prepare_snapshots
from gridiron_edge.features.team.qb_profile import resolve_starting_qb_provenance
from gridiron_edge.ingest.nflverse.weekly_availability import load_depth_charts, load_injuries
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    STATUS_COLUMN,
    has_starting_qb_columns,
)
from gridiron_edge.transform.clean.qb_coach_sidecar import build_qb_coach_sidecar_frame

_SIDES = ("away", "home")


@dataclass(frozen=True, slots=True)
class StarterComparison:
    """One side of one unstarted game: what the product assumed and what resolves now."""

    game_id: str
    side: str
    recorded_id: str | None
    recorded_source: str | None
    current_id: str | None
    current_source: str | None
    current_as_of: str | None
    changed: bool


@dataclass(frozen=True, slots=True)
class StarterObservation:
    """What one starter check found for the target week."""

    record_available: bool
    comparisons: tuple[StarterComparison, ...]
    started_game_ids: tuple[str, ...]
    teams: dict[str, dict[str, Any]]

    @property
    def changed(self) -> tuple[StarterComparison, ...]:
        """The compared sides whose starter now resolves differently."""
        return tuple(item for item in self.comparisons if item.changed)

    def as_results(self) -> dict[str, Any]:
        """Return the JSON-ready record stored in a receipt."""
        return {
            "record_available": self.record_available,
            "compared_sides": len(self.comparisons),
            "changed_sides": len(self.changed),
            "changed": [asdict(item) for item in self.changed],
            "comparisons": [asdict(item) for item in self.comparisons],
            "started_game_ids": list(self.started_game_ids),
            "teams": self.teams,
        }


def frame_digest(frame: DataFrame | None) -> str | None:
    """Return a content digest of a cached frame, independent of how it was serialized."""
    if frame is None:
        return None
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode())
    digest.update(str(len(frame)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _clean(value: object) -> str | None:
    # pyrefly: ignore [no-matching-overload]
    return None if pd.isna(value) else str(value)


def unstarted_game_ids(
    schedule: DataFrame, *, season: str, week: int, now: datetime
) -> tuple[list[str], list[str]]:
    """Split one week's game ids into ``(not yet kicked off, kicked off)`` at ``now``."""
    from gridiron_edge.market.collection_plan import _schedule_kickoff_utc

    scoped = schedule.loc[(schedule["season"].astype(str) == season) & (schedule["week"] == week)]
    ahead: list[str] = []
    started: list[str] = []
    for row in scoped.itertuples():
        kickoff = _schedule_kickoff_utc(row.game_date, row.game_time)
        (ahead if kickoff > now else started).append(str(row.game_id))
    return sorted(ahead), sorted(started)


def compare_starters(
    resolved: DataFrame, product: DataFrame, *, game_ids: list[str]
) -> tuple[bool, list[StarterComparison]]:
    """Compare the product's recorded starters with freshly resolved ones for ``game_ids``.

    Returns ``(record_available, comparisons)``; a product without the starting-QB record
    (schema 1) cannot be compared and yields no comparisons.
    """
    if not has_starting_qb_columns(product):
        return False, []
    current = {(str(row.GAME_ID), str(row.SIDE)): row for row in resolved.itertuples(index=False)}
    comparisons: list[StarterComparison] = []
    wanted = set(game_ids)
    for _, game in product.iterrows():
        game_id = str(game["game_id"])
        if game_id not in wanted or game[STATUS_COLUMN] is None:
            continue
        for side in _SIDES:
            recorded = _clean(game[f"{side}_qb_id"])
            row = current.get((game_id, side.upper()))
            now_id = None if row is None else _clean(row.RESOLVED_QB_ID)
            comparisons.append(
                StarterComparison(
                    game_id=game_id,
                    side=side,
                    recorded_id=recorded,
                    recorded_source=_clean(game[f"{side}_qb_source"]),
                    current_id=now_id,
                    current_source=None if row is None else _clean(row.QB_SOURCE),
                    current_as_of=None if row is None else _clean(row.PROJECTION_AS_OF),
                    changed=recorded != now_id,
                )
            )
    return True, comparisons


def team_evidence(repo: Path, *, season: int, week: int, game_ids: list[str]) -> dict[str, Any]:
    """Return, per team of the given games, the chart quarterbacks and their report status."""
    upcoming_path = dataset_path(repo, "schedule_upcoming_raw_nflverse")
    if not game_ids or not upcoming_path.exists():
        return {}
    try:
        snapshots = load_depth_charts(seasons=[season], repo=repo).get(season)
        injuries = load_injuries(seasons=[season], repo=repo).get(season)
    except FileNotFoundError:
        return {}
    raw = pd.read_parquet(upcoming_path)
    games = raw.loc[raw["game_id"].isin(game_ids)]
    teams = sorted({*games["away_team"].astype(str), *games["home_team"].astype(str)})
    statuses: dict[str, dict[str, str | None]] = {}
    if injuries is not None and not injuries.empty:
        reports = injuries.loc[
            (injuries["season"].astype(int) == season)
            & (injuries["week"].astype(int) == week)
            & (injuries["game_type"] == "REG")
        ]
        for team, player, status in zip(
            reports["team"], reports["gsis_id"], reports["report_status"], strict=True
        ):
            if isinstance(player, str):
                statuses.setdefault(str(team), {})[player] = _clean(status)
    by_team = (
        prepare_snapshots(snapshots)
        if snapshots is not None and not snapshots.empty and "dt" in snapshots.columns
        else {}
    )
    evidence: dict[str, Any] = {}
    for team in teams:
        chart: list[tuple[int, str]] = []
        as_of = None
        if team in by_team:
            chart, snapshot_time = latest_snapshot_chart(by_team[team])
            as_of = None if snapshot_time is None else snapshot_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        evidence[team] = {
            "chart": [{"rank": rank, "gsis_id": player} for rank, player in sorted(chart)],
            "chart_as_of": as_of,
            "report_status": statuses.get(team, {}),
        }
    return evidence


def observe_starters(
    repo: Path, *, season: str, week: int, product: DataFrame, schedule: DataFrame, now: datetime
) -> StarterObservation:
    """Resolve the target week's starters from the cached inputs and compare with ``product``."""
    ahead, started = unstarted_game_ids(schedule, season=season, week=week, now=now)
    resolved = resolve_starting_qb_provenance(build_qb_coach_sidecar_frame(repo=repo))
    record_available, comparisons = compare_starters(resolved, product, game_ids=ahead)
    return StarterObservation(
        record_available=record_available,
        comparisons=tuple(comparisons),
        started_game_ids=tuple(started),
        teams=team_evidence(repo, season=int(season.split("-")[0]), week=week, game_ids=ahead),
    )
