# src/gridiron_edge/weekly_cycle/decision.py

"""Decide which weekly-cycle step is due from the schedule and the receipts alone.

Pure: no clock, network or filesystem access (``DECISIONS.md`` D82). The caller supplies the
time, the upcoming schedule, each week's receipts and whether a week has a current product.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from enum import StrEnum

from pandas import DataFrame

from gridiron_edge.market.collection_plan import _schedule_kickoff_utc
from gridiron_edge.market.collection_rollover import select_target_week
from gridiron_edge.weekly_cycle.cadence import (
    RECAP_RETRY_WINDOW,
    RETRY_AFTER,
    closeout_earliest,
    is_due,
    next_due,
    watch_interval,
)
from gridiron_edge.weekly_cycle.receipts import (
    ATTEMPT_CLOSEOUT,
    ATTEMPT_FORECAST,
    ATTEMPT_RECAP,
    ATTEMPT_REFRESH,
    ATTEMPT_WATCH,
    OUTCOME_DONE,
    UNFINISHED_OUTCOMES,
    CycleReceipt,
    last_receipt,
)

ReceiptsFor = Callable[[str, int], list[CycleReceipt]]
HasProduct = Callable[[str, int], bool]

_CHECK_ATTEMPTS = frozenset({ATTEMPT_WATCH, ATTEMPT_FORECAST, ATTEMPT_REFRESH})

#: The closeout stage that builds and selects the finished week's recap and its season's.
RECAP_STAGE = "build-weekly-recap"


class StepKind(StrEnum):
    """What the next weekly-cycle run should do."""

    CLOSEOUT = "closeout"
    FORECAST = "forecast"
    WATCH = "watch"
    IDLE = "idle"
    SEASON_COMPLETE = "season_complete"


@dataclass(frozen=True, slots=True)
class CycleDecision:
    """The step that is due, or why nothing is, and when to look again.

    ``week`` is the week the step acts on (the finished week for a closeout, otherwise the
    target week). ``started`` says whether any game of the target week has kicked off;
    ``hours_to_kickoff`` is the time to its next unstarted kickoff.
    """

    kind: StepKind
    reason: str
    season: str | None = None
    week: int | None = None
    target_week: int | None = None
    started: bool = False
    hours_to_kickoff: float | None = None
    next_check_at: datetime | None = None


def week_kickoffs(schedule: DataFrame, *, season: str, week: int) -> list[datetime]:
    """Return the kickoff times (UTC) of one week's games, earliest first."""
    scoped = schedule.loc[(schedule["season"].astype(str) == season) & (schedule["week"] == week)]
    return sorted(
        _schedule_kickoff_utc(row.game_date, row.game_time) for row in scoped.itertuples()
    )


def _closed(receipts: list[CycleReceipt]) -> bool:
    return last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT}, outcomes={OUTCOME_DONE}) is not None


def _decide_closeout(
    *,
    now: datetime,
    season: str,
    week: int,
    schedule: DataFrame,
    receipts: list[CycleReceipt],
    tz: tzinfo | None,
    target_week: int | None,
) -> CycleDecision:
    base = {"season": season, "week": week, "target_week": target_week}
    earliest = closeout_earliest(week_kickoffs(schedule, season=season, week=week)[-1], tz=tz)
    if now < earliest:
        return CycleDecision(
            StepKind.IDLE,
            f"week {week} closeout waits for {earliest.isoformat(timespec='minutes')}",
            next_check_at=earliest,
            **base,
        )
    previous = last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT})
    if previous is not None and not is_due(previous.decided, RETRY_AFTER, now=now):
        return CycleDecision(
            StepKind.IDLE,
            f"week {week} closeout was {previous.outcome} under an hour ago",
            next_check_at=next_due(previous.decided, RETRY_AFTER, now=now),
            **base,
        )
    return CycleDecision(StepKind.CLOSEOUT, f"week {week} is over and not yet closed", **base)


def _decide_target(
    *,
    now: datetime,
    season: str,
    week: int,
    schedule: DataFrame,
    receipts: list[CycleReceipt],
    has_product: bool,
) -> CycleDecision:
    kickoffs = week_kickoffs(schedule, season=season, week=week)
    ahead = [kickoff for kickoff in kickoffs if kickoff > now]
    hours = (ahead[0] - now).total_seconds() / 3600 if ahead else 0.0
    base = {
        "season": season,
        "week": week,
        "target_week": week,
        "started": len(ahead) < len(kickoffs),
        "hours_to_kickoff": hours,
    }
    previous = last_receipt(
        receipts, attempted={ATTEMPT_FORECAST} if not has_product else _CHECK_ATTEMPTS
    )
    unfinished = previous is not None and previous.outcome in UNFINISHED_OUTCOMES
    if not has_product:
        if (
            unfinished
            and previous is not None
            and not is_due(previous.decided, RETRY_AFTER, now=now)
        ):
            return CycleDecision(
                StepKind.IDLE,
                f"week {week} forecast was {previous.outcome} under an hour ago",
                next_check_at=next_due(previous.decided, RETRY_AFTER, now=now),
                **base,
            )
        return CycleDecision(StepKind.FORECAST, f"week {week} has no current forecast", **base)

    interval = RETRY_AFTER if unfinished else watch_interval(hours)
    last = None if previous is None else previous.decided
    if is_due(last, interval, now=now):
        return CycleDecision(
            StepKind.WATCH,
            f"starter check due ({hours:.1f} h to the next kickoff)",
            **base,
        )
    return CycleDecision(
        StepKind.IDLE,
        f"next starter check is not due ({hours:.1f} h to the next kickoff)",
        next_check_at=next_due(last, interval, now=now),
        **base,
    )


def decide(
    *,
    now: datetime,
    schedule: DataFrame,
    receipts_for: ReceiptsFor,
    has_product: HasProduct,
    tz: tzinfo | None = None,
    target: tuple[str, int] | None = None,
) -> CycleDecision:
    """Return the weekly-cycle step that is due at ``now``.

    The target week is the earliest week with a game still ahead (``select_target_week``).
    Its predecessor is closed first when it has no closeout receipt, no forecast exists for
    the target week yet, and the closeout window has opened; then the target week is
    forecast when it has no current product; otherwise the starter watch runs on its
    cadence. ``target`` overrides the derived week. When no game is left in the
    schedule, only the final week's closeout remains.
    """
    target = target or select_target_week(schedule, now=now)
    if target is None:
        if schedule.empty:
            return CycleDecision(StepKind.IDLE, "the upcoming schedule is empty")
        season = str(schedule["season"].astype(str).max())
        final = int(schedule.loc[schedule["season"].astype(str) == season, "week"].max())
        receipts = receipts_for(season, final)
        if _closed(receipts):
            return CycleDecision(
                StepKind.SEASON_COMPLETE,
                f"{season} week {final} is closed and no game is left",
                season=season,
                week=final,
            )
        return _decide_closeout(
            now=now,
            season=season,
            week=final,
            schedule=schedule,
            receipts=receipts,
            tz=tz,
            target_week=None,
        )

    season, week = target
    product_exists = has_product(season, week)
    previous_week = week - 1
    if (
        previous_week >= 1
        and not product_exists
        and week_kickoffs(schedule, season=season, week=previous_week)
    ):
        previous_receipts = receipts_for(season, previous_week)
        if not _closed(previous_receipts):
            return _decide_closeout(
                now=now,
                season=season,
                week=previous_week,
                schedule=schedule,
                receipts=previous_receipts,
                tz=tz,
                target_week=week,
            )
    return _decide_target(
        now=now,
        season=season,
        week=week,
        schedule=schedule,
        receipts=receipts_for(season, week),
        has_product=product_exists,
    )


def refresh_allowed(receipts: list[CycleReceipt], *, now: datetime) -> bool:
    """Return whether a forecast refresh may start (at most one attempt an hour)."""
    previous = last_receipt(receipts, attempted={ATTEMPT_REFRESH, ATTEMPT_FORECAST})
    return previous is None or is_due(previous.decided, timedelta(hours=1), now=now)


@dataclass(frozen=True, slots=True)
class RecapSelectionState:
    """What the recap store selects for one finished week and its season.

    ``week_selected``: the week has a selected recap stored at the current recap schema.
    ``season_reads_week``: the selected season recap reads that recap. ``stale_weeks``: the
    season's other weeks whose selected recap is not stored at the current schema, which the
    season roll-up refuses until they are rebuilt.
    """

    week_selected: bool
    season_reads_week: bool
    stale_weeks: tuple[int, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether the week's recap and the season recap reading it are both selected."""
        return self.week_selected and self.season_reads_week


class RecapRebuildStatus(StrEnum):
    """Where a pending recap rebuild stands."""

    DUE = "due"
    WAITING = "waiting"
    HELD = "held"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RecapRebuild:
    """A finished week whose closeout could not build its recap, and what the cycle does next."""

    season: str
    week: int
    status: RecapRebuildStatus
    reason: str
    last_outcome: str
    next_check_at: datetime | None = None


def soft_failed_recap_closeout(receipts: list[CycleReceipt]) -> CycleReceipt | None:
    """Return the week's latest done closeout when it soft-failed the recap stage."""
    closeout = last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT}, outcomes={OUTCOME_DONE})
    if closeout is None or closeout.results.get("stages", {}).get(RECAP_STAGE) != "soft-failed":
        return None
    return closeout


def _recap_failure(closeout: CycleReceipt) -> str:
    prefix = f"{RECAP_STAGE}: "
    for warning in closeout.results.get("warnings", []):
        if isinstance(warning, str) and warning.startswith(prefix):
            return warning.removeprefix(prefix)
    return "soft-failed"


def decide_recap_rebuild(
    *,
    now: datetime,
    season: str,
    week: int,
    closeout: CycleReceipt,
    receipts: list[CycleReceipt],
    state: RecapSelectionState,
) -> RecapRebuild | None:
    """Return the pending recap rebuild of a week whose closeout soft-failed it, or ``None``.

    Nothing is pending once the store selects the week's recap and a season recap reading it,
    whoever rebuilt them. Otherwise the stage is retried hourly for ``RECAP_RETRY_WINDOW``
    after the closeout (``DECISIONS.md`` D91), unless another week's selected recap is not
    stored at the current schema: the season roll-up refuses until that week is rebuilt, so the
    retry is held rather than repeated.
    """
    if state.complete:
        return None
    attempts = [
        receipt
        for receipt in receipts
        if receipt.attempted == ATTEMPT_RECAP and receipt.decided >= closeout.decided
    ]
    latest = max(attempts, key=lambda receipt: receipt.decided, default=None)
    last_outcome = (
        f"closeout soft-failed it: {_recap_failure(closeout)}"
        if latest is None
        else f"rebuild {latest.outcome}: {latest.detail}"
    )
    base = {"season": season, "week": week, "last_outcome": last_outcome}
    manual = f"post-week --only build-weekly-recap --week {{week}} --season {season}"
    if now >= closeout.decided + RECAP_RETRY_WINDOW:
        return RecapRebuild(
            status=RecapRebuildStatus.STOPPED,
            reason=f"retries stopped {RECAP_RETRY_WINDOW.days} days after the closeout; "
            f"rebuild by hand: {manual.format(week=week)}",
            **base,
        )
    if state.stale_weeks:
        weeks = ", ".join(str(number) for number in state.stale_weeks)
        return RecapRebuild(
            status=RecapRebuildStatus.HELD,
            reason=f"the season recap waits on week(s) {weeks}, whose selected recap is not "
            f"stored at the current schema; rebuild each: {manual.format(week='W')}",
            **base,
        )
    last = closeout.decided if latest is None else latest.decided
    if not is_due(last, RETRY_AFTER, now=now):
        return RecapRebuild(
            status=RecapRebuildStatus.WAITING,
            reason="last attempt under an hour ago",
            next_check_at=next_due(last, RETRY_AFTER, now=now),
            **base,
        )
    return RecapRebuild(
        status=RecapRebuildStatus.DUE,
        reason="the last attempt was an hour or more ago",
        **base,
    )
