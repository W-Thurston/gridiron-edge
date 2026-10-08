# src/gridiron_edge/weekly_cycle/runner.py

"""Run the weekly cycle: decide what is due, do it through ports, and leave a receipt.

Every effect (stage runs, network refreshes, product reads, the clock) goes through
:class:`CyclePorts`, so the state machine is tested against stubs; the CLI supplies the real
ports (``DECISIONS.md`` D82).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

from pandas import DataFrame

from gridiron_edge.weekly_cycle.cadence import watch_interval
from gridiron_edge.weekly_cycle.decision import (
    CycleDecision,
    RecapRebuild,
    RecapRebuildStatus,
    RecapSelectionState,
    StepKind,
    decide,
    decide_recap_rebuild,
    refresh_allowed,
    soft_failed_recap_closeout,
)
from gridiron_edge.weekly_cycle.receipts import (
    ATTEMPT_CLOSEOUT,
    ATTEMPT_EDGES,
    ATTEMPT_FORECAST,
    ATTEMPT_RECAP,
    ATTEMPT_REFRESH,
    ATTEMPT_WATCH,
    OUTCOME_BLOCKED,
    OUTCOME_CHANGED,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    OUTCOME_UNCHANGED,
    OUTCOME_WAITING,
    CycleReceipt,
    read_receipts,
    write_receipt,
)
from gridiron_edge.weekly_cycle.starter_watch import StarterObservation

#: The stages whose failure means "upstream data for the finished week is not all in yet".
RESULTS_GATE_STAGES = frozenset({"verify-week-complete", "verify-week-data-current"})
_MAX_STEPS = 3


@dataclass(frozen=True, slots=True)
class StageRun:
    """The outcome of a composite stage run, reduced to what a receipt records."""

    success: bool
    detail: str
    failed_stage: str | None = None
    stages: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AvailabilityRefresh:
    """Whether the depth chart and injury caches were refreshed from upstream."""

    depth_charts_ok: bool
    injuries_ok: bool


@dataclass(frozen=True, slots=True)
class CyclePorts:
    """Every effect the cycle performs, injectable for tests."""

    clock: Callable[[], datetime]
    load_schedule: Callable[[], DataFrame]
    product_id: Callable[[str, int], str | None]
    load_product: Callable[[str, int], DataFrame]
    dirty_files: Callable[[], list[str]]
    run_closeout: Callable[[str, int, bool], StageRun]
    run_forecast: Callable[[str, int], StageRun]
    run_edges: Callable[[str, int], StageRun]
    refresh_availability: Callable[[int], AvailabilityRefresh]
    digests: Callable[[int], dict[str, str | None]]
    observe: Callable[[str, int, DataFrame, DataFrame, datetime], StarterObservation]
    edge_inputs: Callable[[str, int], dict[str, Any]]
    recap_state: Callable[[str, int], RecapSelectionState]
    run_recap: Callable[[str, int], StageRun]


@dataclass(slots=True)
class CycleOutcome:
    """What a run did, in order, and whether anything failed."""

    lines: list[str] = field(default_factory=list)
    failed: bool = False
    decision: CycleDecision | None = None


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _last_with(receipts: list[CycleReceipt], key: str) -> dict[str, Any] | None:
    for receipt in reversed(receipts):
        value = receipt.results.get(key)
        if isinstance(value, dict):
            return value
    return None


def _dirty_summary(files: list[str]) -> str:
    shown = ", ".join(files[:5]) + (f" and {len(files) - 5} more" if len(files) > 5 else "")
    return f"tracked worktree has uncommitted changes: {shown}"


class _Cycle:
    def __init__(
        self,
        *,
        repo: Path,
        ports: CyclePorts,
        tz: tzinfo | None,
        target: tuple[str, int] | None,
        allow_incomplete: bool,
    ) -> None:
        self.repo = repo
        self.ports = ports
        self.tz = tz
        self.target = target
        self.allow_incomplete = allow_incomplete
        self.out = CycleOutcome()

    # -- helpers ---------------------------------------------------------

    def receipts(self, season: str, week: int) -> list[CycleReceipt]:
        return read_receipts(self.repo, season=season, week=week)

    def decide(self) -> CycleDecision:
        now = self.ports.clock()
        return decide(
            now=now,
            schedule=self.ports.load_schedule(),
            receipts_for=self.receipts,
            has_product=lambda season, week: self.ports.product_id(season, week) is not None,
            tz=self.tz,
            target=self.target,
        )

    def record(
        self,
        decision: CycleDecision,
        *,
        attempted: str,
        outcome: str,
        detail: str,
        inputs: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        assert decision.season is not None and decision.week is not None
        write_receipt(
            self.repo,
            CycleReceipt(
                decided_at=_iso(self.ports.clock()),
                season=decision.season,
                week=decision.week,
                attempted=attempted,
                outcome=outcome,
                detail=detail,
                inputs=inputs or {},
                results=results or {},
            ),
        )
        self.say(f"{attempted}: {outcome} - {detail}")

    def say(self, line: str) -> None:
        self.out.lines.append(line)

    def guarded(self, decision: CycleDecision, attempted: str, action: Callable[[], bool]) -> bool:
        """Run one step, turning an unexpected exception into a failed receipt."""
        try:
            return action()
        except Exception as exc:
            self.out.failed = True
            self.record(
                decision,
                attempted=attempted,
                outcome=OUTCOME_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return False

    # -- steps -----------------------------------------------------------

    def closeout(self, decision: CycleDecision) -> bool:
        assert decision.season is not None and decision.week is not None
        run = self.ports.run_closeout(decision.season, decision.week, self.allow_incomplete)
        inputs = {"allow_incomplete": self.allow_incomplete}
        results = {"stages": run.stages, "warnings": run.warnings, "report": run.report}
        if run.success:
            self.record(
                decision,
                attempted=ATTEMPT_CLOSEOUT,
                outcome=OUTCOME_DONE,
                detail=run.detail,
                inputs=inputs,
                results=results,
            )
            return True
        waiting = run.failed_stage in RESULTS_GATE_STAGES
        self.out.failed = self.out.failed or not waiting
        self.record(
            decision,
            attempted=ATTEMPT_CLOSEOUT,
            outcome=OUTCOME_WAITING if waiting else OUTCOME_FAILED,
            detail=run.detail,
            inputs=inputs,
            results=results,
        )
        return False

    def forecast(self, decision: CycleDecision, *, attempted: str, inputs: dict[str, Any]) -> bool:
        assert decision.season is not None and decision.week is not None
        season, week = decision.season, decision.week
        dirty = self.ports.dirty_files()
        if dirty:
            self.record(
                decision,
                attempted=attempted,
                outcome=OUTCOME_BLOCKED,
                detail=_dirty_summary(dirty),
                inputs=inputs,
                results={"dirty_files": dirty},
            )
            return False
        before = self.ports.product_id(season, week)
        run = self.ports.run_forecast(season, week)
        after = self.ports.product_id(season, week)
        results: dict[str, Any] = {
            "product_before": before,
            "product_after": after,
            "stages": run.stages,
            "warnings": run.warnings,
            "report": run.report,
        }
        if run.success:
            results["edge_inputs"] = self.ports.edge_inputs(season, week)
            results["digests"] = self.ports.digests(int(season.split("-")[0]))
        else:
            self.out.failed = True
        self.record(
            decision,
            attempted=attempted,
            outcome=OUTCOME_DONE if run.success else OUTCOME_FAILED,
            detail=run.detail,
            inputs=inputs,
            results=results,
        )
        return run.success

    def watch(self, decision: CycleDecision) -> bool:
        assert decision.season is not None and decision.week is not None
        season, week = decision.season, decision.week
        now = self.ports.clock()
        hours = decision.hours_to_kickoff or 0.0
        inputs: dict[str, Any] = {
            "hours_to_kickoff": round(hours, 2),
            "interval_hours": watch_interval(hours).total_seconds() / 3600,
            "game_started": decision.started,
        }
        refreshed = self.ports.refresh_availability(int(season.split("-")[0]))
        inputs["depth_charts_refreshed"] = refreshed.depth_charts_ok
        inputs["injuries_refreshed"] = refreshed.injuries_ok
        previous = self.receipts(season, week)
        if not (refreshed.depth_charts_ok and refreshed.injuries_ok):
            self.out.failed = True
            self.record(
                decision,
                attempted=ATTEMPT_WATCH,
                outcome=OUTCOME_FAILED,
                detail="could not refresh the depth charts or injury reports from upstream",
                inputs=inputs,
            )
            return False

        digests = self.ports.digests(int(season.split("-")[0]))
        earlier = _last_with(previous, "digests") or {}
        product = self.ports.load_product(season, week)
        observation = self.ports.observe(season, week, product, self.ports.load_schedule(), now)
        results: dict[str, Any] = observation.as_results()
        results["digests"] = digests
        results["upstream_changed"] = {
            name: (None if name not in earlier else earlier[name] != digest)
            for name, digest in digests.items()
        }
        changed = observation.changed
        if not observation.record_available:
            results["disposition"] = "no_starter_record"
            self.record(
                decision,
                attempted=ATTEMPT_WATCH,
                outcome=OUTCOME_UNCHANGED,
                detail="the current product records no starters, so nothing can be compared",
                inputs=inputs,
                results=results,
            )
            return True
        if not changed:
            results["disposition"] = "no_change"
            self.record(
                decision,
                attempted=ATTEMPT_WATCH,
                outcome=OUTCOME_UNCHANGED,
                detail=f"{len(observation.comparisons)} starters compared, none changed",
                inputs=inputs,
                results=results,
            )
            return True

        names = ", ".join(f"{item.game_id} {item.side}" for item in changed)
        if not refresh_allowed(previous, now=now):
            results["disposition"] = "refresh_deferred"
            detail = f"starter changed, refresh deferred: one attempt an hour ({names})"
        else:
            results["disposition"] = "refresh"
            detail = f"starter changed, refreshing the forecast ({names})"
        self.record(
            decision,
            attempted=ATTEMPT_WATCH,
            outcome=OUTCOME_CHANGED,
            detail=detail,
            inputs=inputs,
            results=results,
        )
        if results["disposition"] == "refresh":
            return self.forecast(decision, attempted=ATTEMPT_REFRESH, inputs=inputs)
        return True

    def edges(self, season: str, week: int) -> bool:
        """Rebuild the edge report when a newer quote snapshot or product has arrived."""
        if self.ports.product_id(season, week) is None:
            return True
        current = self.ports.edge_inputs(season, week)
        if current == (_last_with(self.receipts(season, week), "edge_inputs") or {}):
            return True
        decision = CycleDecision(StepKind.IDLE, "edges", season=season, week=week)
        run = self.ports.run_edges(season, week)
        # A stale market blocks edges by design; record it so the same snapshot is not retried.
        self.record(
            decision,
            attempted=ATTEMPT_EDGES,
            outcome=OUTCOME_DONE if run.success else OUTCOME_BLOCKED,
            detail=run.detail,
            inputs={"edge_inputs": current},
            results={"edge_inputs": current, "stages": run.stages},
        )
        return True

    def recap_rebuilds(self, season: str, through_week: int) -> list[RecapRebuild]:
        """Return the pending recap rebuild of every week of the season up to ``through_week``."""
        now = self.ports.clock()
        pending: list[RecapRebuild] = []
        for week in range(1, through_week + 1):
            receipts = self.receipts(season, week)
            closeout = soft_failed_recap_closeout(receipts)
            if closeout is None:
                continue
            rebuild = decide_recap_rebuild(
                now=now,
                season=season,
                week=week,
                closeout=closeout,
                receipts=receipts,
                state=self.ports.recap_state(season, week),
            )
            if rebuild is not None:
                pending.append(rebuild)
        return pending

    def rebuild_recap(self, rebuild: RecapRebuild) -> bool:
        """Rerun the closeout's recap stage for one week; a failure is retried, never fatal."""
        run = self.ports.run_recap(rebuild.season, rebuild.week)
        self.record(
            CycleDecision(StepKind.IDLE, "recap", season=rebuild.season, week=rebuild.week),
            attempted=ATTEMPT_RECAP,
            outcome=OUTCOME_DONE if run.success else OUTCOME_FAILED,
            detail=run.detail,
            results={"stages": run.stages, "warnings": run.warnings, "report": run.report},
        )
        return run.success

    def recaps(self, decision: CycleDecision, *, dry_run: bool) -> bool:
        """Report each pending recap rebuild and, unless a dry run, run those that are due."""
        assert decision.season is not None
        through = max(week for week in (decision.week, decision.target_week) if week is not None)
        for rebuild in self.recap_rebuilds(decision.season, through):
            if dry_run or rebuild.status is not RecapRebuildStatus.DUE:
                self.say(_describe_recap(rebuild, tz=self.tz))
                continue
            week_decision = CycleDecision(
                StepKind.IDLE, "recap", season=rebuild.season, week=rebuild.week
            )
            self.guarded(week_decision, ATTEMPT_RECAP, lambda r=rebuild: self.rebuild_recap(r))
        return True


def run_cycle(
    *,
    repo: Path,
    ports: CyclePorts,
    tz: tzinfo | None = None,
    target: tuple[str, int] | None = None,
    allow_incomplete: bool = False,
    dry_run: bool = False,
) -> CycleOutcome:
    """Do whatever is due, up to a closeout and the forecast that follows it.

    A dry run reports the decision and writes nothing. Otherwise each step leaves a receipt;
    after the steps the edge report is rebuilt when its inputs changed, and a recap the closeout
    soft-failed is rebuilt when due (``DECISIONS.md`` D91). A failed step is reported in the
    outcome (``failed``) and retried after an hour by the decision rules.
    """
    cycle = _Cycle(repo=repo, ports=ports, tz=tz, target=target, allow_incomplete=allow_incomplete)
    decision = cycle.decide()
    for _ in range(_MAX_STEPS):
        cycle.out.decision = decision
        cycle.say(_describe(decision, tz=tz))
        if dry_run or decision.kind not in {StepKind.CLOSEOUT, StepKind.FORECAST, StepKind.WATCH}:
            break
        if decision.kind is StepKind.CLOSEOUT:
            ok = cycle.guarded(decision, ATTEMPT_CLOSEOUT, lambda d=decision: cycle.closeout(d))
        elif decision.kind is StepKind.FORECAST:
            ok = cycle.guarded(
                decision,
                ATTEMPT_FORECAST,
                lambda d=decision: cycle.forecast(d, attempted=ATTEMPT_FORECAST, inputs={}),
            )
        else:
            ok = cycle.guarded(decision, ATTEMPT_WATCH, lambda d=decision: cycle.watch(d))
        if not ok or decision.kind is StepKind.WATCH:
            break
        decision = cycle.decide()
    if not dry_run and decision.season is not None and decision.target_week is not None:
        cycle.guarded(
            decision,
            ATTEMPT_EDGES,
            lambda: cycle.edges(decision.season or "", decision.target_week or 0),
        )
    if decision.season is not None and decision.week is not None:
        if dry_run:  # an error surfaces as it is; a dry run records nothing
            cycle.recaps(decision, dry_run=True)
        else:
            cycle.guarded(decision, ATTEMPT_RECAP, lambda: cycle.recaps(decision, dry_run=False))
    return cycle.out


def _describe(decision: CycleDecision, *, tz: tzinfo | None) -> str:
    target = (
        f"{decision.season} week {decision.target_week}"
        if decision.target_week is not None
        else (f"{decision.season} (no game left)" if decision.season else "no schedule")
    )
    line = f"target {target}: {decision.kind.value} - {decision.reason}"
    if decision.next_check_at is not None:
        line += f"; next check {decision.next_check_at.astimezone(tz):%Y-%m-%d %H:%M %Z}"
    return line


def _describe_recap(rebuild: RecapRebuild, *, tz: tzinfo | None) -> str:
    line = (
        f"week {rebuild.week} recap rebuild {rebuild.status.value}: {rebuild.reason}; "
        f"last: {rebuild.last_outcome}"
    )
    if rebuild.next_check_at is not None:
        line += f"; next check {rebuild.next_check_at.astimezone(tz):%Y-%m-%d %H:%M %Z}"
    return line
