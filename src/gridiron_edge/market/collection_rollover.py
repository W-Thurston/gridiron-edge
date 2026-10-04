# src/gridiron_edge/market/collection_rollover.py

"""Automatic weekly rollover of the selected quote-collection plan (``DECISIONS.md`` D79)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Final
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
from gridiron_edge.market.collection_plan import (
    CollectionPlanStatus,
    QuoteCollectionPolicy,
    WeeklyQuoteCollectionPlan,
    _schedule_kickoff_utc,
    build_weekly_quote_collection_plan,
    derive_kickoff_groups,
)
from gridiron_edge.market.collection_plan_store import (
    collection_plan_path,
    read_collection_plan,
    read_current_collection_plan_selection,
    select_current_collection_plan,
    write_collection_plan,
)

RECEIPT_DIRECTORY: Final[str] = "collection_rollovers"
RETRY_AFTER: Final[timedelta] = timedelta(hours=1)

OUTCOME_NOT_DUE: Final[str] = "not_due"
OUTCOME_ROLLED: Final[str] = "rolled"
OUTCOME_SELECTED_EXISTING: Final[str] = "selected_existing"
OUTCOME_SEASON_COMPLETE: Final[str] = "season_complete"
OUTCOME_FAILED: Final[str] = "failed"
OUTCOME_DEFERRED: Final[str] = "deferred"
_RETRIED_OUTCOMES: Final[frozenset[str]] = frozenset({OUTCOME_FAILED, OUTCOME_SEASON_COMPLETE})

ScheduleRefresh = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RolloverReceipt:
    """One rollover decision and what it was based on."""

    decided_at: str
    outcome: str
    detail: str
    current: str | None = None
    target: str | None = None
    schedule_refreshed: bool | None = None
    schedule_refresh_error: str | None = None
    planned_polls: int | None = None
    planned_credit_cost: int | None = None
    kickoff_groups_changed: bool | None = None


def select_target_week(schedule: DataFrame, *, now: datetime) -> tuple[str, int] | None:
    """Return the earliest (season, week) that still has a game kicking off after ``now``."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware UTC.")
    upcoming: list[tuple[str, int]] = []
    for row in schedule.itertuples(index=False):
        if _schedule_kickoff_utc(row.game_date, row.game_time) > now:
            upcoming.append((str(row.season), int(str(row.week))))
    return min(upcoming) if upcoming else None


def nflverse_schedule_refresh(repo: Path, now: datetime) -> ScheduleRefresh:
    """Return the refresh that re-fetches and re-cleans the season's upcoming schedule."""

    def refresh(season_label: str) -> None:
        from gridiron_edge.ingest.nflverse.schedule import fetch_nflverse_upcoming
        from gridiron_edge.transform.clean.schedule_nflverse import clean_nflverse_upcoming

        fetch_nflverse_upcoming(season=int(season_label.split("-")[0]), repo=repo)
        clean_nflverse_upcoming(repo=repo, ingested_at=now)

    return refresh


def _label(season: str, week: int) -> str:
    return f"{season} week {week}"


def _receipts_dir(repo: Path) -> Path:
    return repo / "data" / "odds" / RECEIPT_DIRECTORY


def _latest_receipt(repo: Path) -> dict[str, object] | None:
    directory = _receipts_dir(repo)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.json"))
    if not files:
        return None
    try:
        payload = json.loads(files[-1].read_text())
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_receipt(repo: Path, receipt: RolloverReceipt) -> Path:
    directory = _receipts_dir(repo)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = receipt.decided_at.replace(":", "").replace("-", "")
    path = directory / f"{stamp}_{receipt.outcome}.json"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _current_selection(repo: Path) -> tuple[str, int] | None:
    try:
        selection = read_current_collection_plan_selection(repo=repo)
    except (FileNotFoundError, ValueError):
        return None
    return selection.season, selection.week


def _kickoff_groups_changed(
    schedule: DataFrame, *, season: str, week: int, repo: Path
) -> bool | None:
    try:
        plan = read_collection_plan(season=season, week=week, repo=repo)
    except (FileNotFoundError, ValueError):
        return None
    return derive_kickoff_groups(schedule, season=season, week=week) != plan.kickoff_groups


def _recently_retried(repo: Path, now: datetime) -> dict[str, object] | None:
    latest = _latest_receipt(repo)
    if latest is None or latest.get("outcome") not in _RETRIED_OUTCOMES:
        return None
    decided = pd.Timestamp(str(latest.get("decided_at")))
    if pd.isna(decided) or now - decided.to_pydatetime() >= RETRY_AFTER:
        return None
    return latest


def _build_plan(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    now: datetime,
    policy: QuoteCollectionPolicy,
) -> WeeklyQuoteCollectionPlan:
    return build_weekly_quote_collection_plan(
        schedule,
        season=season,
        week=week,
        plan_start=now,
        created_at=now,
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class _Refreshed:
    schedule: DataFrame | None
    refreshed: bool | None
    error: str | None


def _refresh_schedule(
    schedule: DataFrame | None,
    *,
    repo: Path,
    current: tuple[str, int] | None,
    refresh: ScheduleRefresh | None,
) -> _Refreshed:
    """Refresh the schedule once; a failure keeps the schedule that was already loaded."""
    if refresh is None:
        return _Refreshed(schedule, None, None)
    if current is not None:
        season_label: str | None = current[0]
    elif schedule is not None and not schedule.empty:
        season_label = str(schedule["season"].iloc[0])
    else:
        season_label = None
    if season_label is None:
        return _Refreshed(schedule, False, "no season is known to refresh")
    try:
        refresh(season_label)
        return _Refreshed(load_schedule_upcoming_rich(repo), True, None)
    except Exception as exc:
        return _Refreshed(schedule, False, f"{type(exc).__name__}: {exc}")


def _load_plan_for(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    now: datetime,
    policy: QuoteCollectionPolicy,
    repo: Path,
) -> tuple[WeeklyQuoteCollectionPlan | None, str, str]:
    """Return the plan for the target week (existing or newly built), its outcome and detail."""
    if collection_plan_path(season=season, week=week, repo=repo).is_file():
        try:
            plan = read_collection_plan(season=season, week=week, repo=repo)
        except ValueError as exc:
            return None, OUTCOME_FAILED, f"the existing plan for the new week is invalid: {exc}"
        return plan, OUTCOME_SELECTED_EXISTING, "selected the existing plan for the new week"
    plan = _build_plan(schedule, season=season, week=week, now=now, policy=policy)
    if plan.status is not CollectionPlanStatus.AVAILABLE:
        detail = f"the new week's plan is {plan.status.value}; it was not selected"
        return None, OUTCOME_FAILED, detail
    return plan, OUTCOME_ROLLED, "built and selected a plan for the new week"


def auto_rollover(
    *,
    repo: Path,
    now: datetime,
    refresh: ScheduleRefresh | None,
    policy: QuoteCollectionPolicy | None = None,
    dry_run: bool = False,
) -> RolloverReceipt:
    """Run one rollover decision; an invalid schedule or plan is a recorded failure.

    See :func:`_auto_rollover`. A ``ValueError`` from the schedule or plan (a missing
    kickoff time, an invalid stored plan) becomes a ``failed`` receipt that is debounced
    like any other failure, so a broken input is reported once an hour, not every tick.
    """
    try:
        return _auto_rollover(repo=repo, now=now, refresh=refresh, policy=policy, dry_run=dry_run)
    except ValueError as exc:
        decided_at = now.astimezone(UTC).isoformat()
        previous = None if dry_run else _recently_retried(repo, now)
        if previous is not None:
            return RolloverReceipt(
                decided_at=decided_at,
                outcome=OUTCOME_DEFERRED,
                detail=f"last attempt ({previous.get('outcome')}) is under an hour old",
            )
        current = _current_selection(repo)
        receipt = RolloverReceipt(
            decided_at=decided_at,
            outcome=OUTCOME_FAILED,
            detail=f"the schedule or plan is invalid: {exc}",
            current=None if current is None else _label(*current),
        )
        if not dry_run:
            _write_receipt(repo, receipt)
        return receipt


def _auto_rollover(
    *,
    repo: Path,
    now: datetime,
    refresh: ScheduleRefresh | None,
    policy: QuoteCollectionPolicy | None,
    dry_run: bool,
) -> RolloverReceipt:
    """Select the right week's collection plan, building it when needed.

    Does nothing (and no I/O beyond reading the schedule and selection) while the
    current selection already equals the earliest week with a game still ahead. Once
    that changes it refreshes the schedule through ``refresh`` (a failure keeps the
    existing schedule), then selects an existing plan for the new week or builds one
    from ``now``. A dry run refreshes nothing and writes nothing.
    """
    decided_at = now.astimezone(UTC).isoformat()
    current = _current_selection(repo)
    current_label = None if current is None else _label(*current)
    try:
        schedule: DataFrame | None = load_schedule_upcoming_rich(repo)
    except (FileNotFoundError, OSError):
        schedule = None
    target = None if schedule is None else select_target_week(schedule, now=now)
    if target is not None and current == target:
        return RolloverReceipt(
            decided_at=decided_at,
            outcome=OUTCOME_NOT_DUE,
            detail="the current plan is already the earliest week with a game ahead",
            current=current_label,
            target=_label(*target),
        )
    previous = None if dry_run else _recently_retried(repo, now)
    if previous is not None:
        return RolloverReceipt(
            decided_at=decided_at,
            outcome=OUTCOME_DEFERRED,
            detail=f"last attempt ({previous.get('outcome')}) is under an hour old",
            current=current_label,
        )

    refreshed = (
        _Refreshed(schedule, None, None)
        if dry_run
        else _refresh_schedule(schedule, repo=repo, current=current, refresh=refresh)
    )
    receipt = _decide(
        refreshed,
        repo=repo,
        now=now,
        current=current,
        policy=policy or QuoteCollectionPolicy(),
        dry_run=dry_run,
    )
    if not dry_run and receipt.outcome not in {OUTCOME_NOT_DUE, OUTCOME_DEFERRED}:
        _write_receipt(repo, receipt)
    return receipt


def _decide(
    refreshed: _Refreshed,
    *,
    repo: Path,
    now: datetime,
    current: tuple[str, int] | None,
    policy: QuoteCollectionPolicy,
    dry_run: bool,
) -> RolloverReceipt:
    common = {
        "decided_at": now.astimezone(UTC).isoformat(),
        "current": None if current is None else _label(*current),
        "schedule_refreshed": refreshed.refreshed,
        "schedule_refresh_error": refreshed.error,
    }
    schedule = refreshed.schedule
    if schedule is None:
        return RolloverReceipt(
            outcome=OUTCOME_FAILED, detail="no upcoming schedule is available", **common
        )
    target = select_target_week(schedule, now=now)
    if target is None:
        return RolloverReceipt(
            outcome=OUTCOME_SEASON_COMPLETE,
            detail="no game kicks off after now; the next season rolls over manually",
            **common,
        )
    season, week = target
    if current == target:
        return RolloverReceipt(
            outcome=OUTCOME_NOT_DUE,
            detail="the refreshed schedule leaves the current plan as the target",
            target=_label(*target),
            **common,
        )
    changed = _kickoff_groups_changed(schedule, season=season, week=week, repo=repo)
    plan, outcome, detail = _load_plan_for(
        schedule, season=season, week=week, now=now, policy=policy, repo=repo
    )
    if plan is None:
        return RolloverReceipt(
            outcome=outcome,
            detail=detail,
            target=_label(*target),
            kickoff_groups_changed=changed,
            **common,
        )
    if not dry_run:
        if outcome == OUTCOME_ROLLED:
            write_collection_plan(plan, repo=repo)
        select_current_collection_plan(season=season, week=week, selected_at=now, repo=repo)
    return RolloverReceipt(
        outcome=outcome,
        detail=detail + (" (dry run, nothing written)" if dry_run else ""),
        target=_label(*target),
        planned_polls=plan.planned_poll_count,
        planned_credit_cost=plan.planned_credit_cost,
        kickoff_groups_changed=changed,
        **common,
    )
