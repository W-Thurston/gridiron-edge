# src/gridiron_edge/weekly_cycle/alerts.py

"""Decide when the weekly cycle needs a person, and which alert notices that calls for.

The conditions and the notice plan are pure: the caller supplies the time, each week's receipts,
the closeout windows, the pending recap rebuilds and the alert history (``DECISIONS.md`` D92).
Every notice is an immutable alert receipt; the open alerts are the keys whose latest notice is
not ``resolved``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
from typing import Final

from gridiron_edge.weekly_cycle.cadence import (
    ALERT_REMINDER_INTERVAL,
    ALERT_WAITING_AFTER,
    is_due,
)
from gridiron_edge.weekly_cycle.decision import RecapRebuild, RecapRebuildStatus
from gridiron_edge.weekly_cycle.receipts import (
    ATTEMPT_CLOSEOUT,
    ATTEMPT_EDGES,
    ATTEMPT_FORECAST,
    ATTEMPT_RECAP,
    ATTEMPT_REFRESH,
    ATTEMPT_WATCH,
    OUTCOME_BLOCKED,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    OUTCOME_WAITING,
    CycleReceipt,
    last_receipt,
    receipt_stamp,
    write_new_json,
)

SCHEMA_VERSION: Final[int] = 1

#: The closeout's informational stage that checks the quote collector rolled to the next week.
COLLECTOR_STAGE: Final[str] = "check-collector"
#: The forecast stage, and the edges hook, that build the week's edge report.
EDGES_STAGE: Final[str] = "generate-edges"
#: Steps that act on the target week; their alerts end once the week is no longer the target.
TARGET_WEEK_STEPS: Final[frozenset[str]] = frozenset(
    {ATTEMPT_FORECAST, ATTEMPT_REFRESH, ATTEMPT_WATCH, ATTEMPT_EDGES}
)
_DETAIL_LIMIT = 200


class Notice(StrEnum):
    """What an alert receipt announces."""

    OPENED = "opened"
    REMINDER = "reminder"
    RESOLVED = "resolved"


class Delivery(StrEnum):
    """Where a notice went: a channel, nowhere but the run output, or a failed send."""

    SENT = "sent"
    LOCAL = "local"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, order=True)
class AlertKey:
    """One thing the cycle needs a person for: a step of a season week in one condition."""

    season: str
    week: int
    step: str
    condition: str

    def describe(self) -> str:
        """Name the alert: season, week, step and condition."""
        return f"{self.season} week {self.week} {self.step} {self.condition}"


@dataclass(frozen=True, slots=True)
class AlertReceipt:
    """One alert notice and how it was delivered."""

    decided_at: str
    season: str
    week: int
    step: str
    condition: str
    notice: str
    detail: str
    delivery: str
    delivery_detail: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def key(self) -> AlertKey:
        """The alert this notice belongs to."""
        return AlertKey(self.season, self.week, self.step, self.condition)

    @property
    def decided(self) -> datetime:
        """The notice time as an aware UTC datetime."""
        return datetime.fromisoformat(self.decided_at).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class OpenAlert:
    """An alert whose latest notice is not ``resolved``."""

    key: AlertKey
    opened_at: datetime
    last: AlertReceipt


@dataclass(frozen=True, slots=True)
class PlannedNotice:
    """A notice to send now, with its one-line reason."""

    key: AlertKey
    notice: Notice
    detail: str

    def title(self) -> str:
        """The notice's headline: season, week, step and condition only."""
        if self.notice is Notice.RESOLVED:
            return f"{self.key.describe()} resolved"
        if self.notice is Notice.REMINDER:
            return f"{self.key.describe()} (still open)"
        return self.key.describe()


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _DETAIL_LIMIT else text[: _DETAIL_LIMIT - 3] + "..."


def _stage_warning(receipt: CycleReceipt, stage: str) -> str:
    prefix = f"{stage}: "
    for warning in receipt.results.get("warnings", []):
        if isinstance(warning, str) and warning.startswith(prefix):
            return warning.removeprefix(prefix)
    return receipt.detail


def _week_conditions(
    *,
    now: datetime,
    season: str,
    week: int,
    receipts: list[CycleReceipt],
    is_target: bool,
    closeout_window: datetime | None,
    collector_rolled: Callable[[str, int], bool],
) -> dict[AlertKey, str]:
    found: dict[AlertKey, str] = {}

    def add(step: str, condition: str, detail: str) -> None:
        found[AlertKey(season, week, step, condition)] = _short(detail)

    closeout = last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT})
    if closeout is not None:
        if closeout.outcome == OUTCOME_FAILED:
            add(ATTEMPT_CLOSEOUT, OUTCOME_FAILED, closeout.detail)
        elif (
            closeout.outcome == OUTCOME_WAITING
            and closeout_window is not None
            and now >= closeout_window + ALERT_WAITING_AFTER
        ):
            add(ATTEMPT_CLOSEOUT, OUTCOME_WAITING, closeout.detail)
        elif (
            closeout.outcome == OUTCOME_DONE
            and closeout.results.get("stages", {}).get(COLLECTOR_STAGE) == "soft-failed"
            and not collector_rolled(season, week + 1)
        ):
            add(COLLECTOR_STAGE, "soft-failed", _stage_warning(closeout, COLLECTOR_STAGE))

    # A failed retry is still being retried (D91); only a hook that raised needs a person here.
    recap = last_receipt(receipts, attempted={ATTEMPT_RECAP})
    if recap is not None and recap.outcome == OUTCOME_FAILED and "exception" in recap.results:
        add(ATTEMPT_RECAP, OUTCOME_FAILED, recap.detail)

    if not is_target:
        return found
    for step, conditions in (
        (ATTEMPT_FORECAST, {OUTCOME_FAILED, OUTCOME_BLOCKED}),
        (ATTEMPT_REFRESH, {OUTCOME_FAILED, OUTCOME_BLOCKED}),
        (ATTEMPT_WATCH, {OUTCOME_FAILED}),
    ):
        latest = last_receipt(receipts, attempted={step})
        if latest is not None and latest.outcome in conditions:
            add(step, latest.outcome, latest.detail)

    hook = last_receipt(receipts, attempted={ATTEMPT_EDGES})
    if hook is not None and hook.outcome == OUTCOME_FAILED:
        add(ATTEMPT_EDGES, OUTCOME_FAILED, hook.detail)
    built = [receipt for receipt in receipts if EDGES_STAGE in receipt.results.get("stages", {})]
    edges = max(built, key=lambda receipt: receipt.decided, default=None)
    if edges is not None and edges.results["stages"][EDGES_STAGE] != "ok":
        detail = (
            edges.detail if edges.attempted == ATTEMPT_EDGES else _stage_warning(edges, EDGES_STAGE)
        )
        add(ATTEMPT_EDGES, OUTCOME_BLOCKED, detail)
    return found


def open_conditions(
    *,
    now: datetime,
    season: str,
    receipts_by_week: Mapping[int, list[CycleReceipt]],
    target: tuple[str, int] | None,
    closeout_windows: Mapping[int, datetime],
    recap_rebuilds: list[RecapRebuild],
    collector_rolled: Callable[[str, int], bool],
) -> dict[AlertKey, str]:
    """Return every condition in which the cycle needs a person, with its one-line reason.

    Target-week steps (forecast, refresh, watch, edges) are judged on the target week only.
    A closeout counts once it failed, has waited ``ALERT_WAITING_AFTER`` past its window, or
    soft-failed the collector check while the collector has not rolled to the next week. A
    recap rebuild counts once it is held or stopped (``DECISIONS.md`` D91, D92).
    """
    found: dict[AlertKey, str] = {}
    for week, receipts in sorted(receipts_by_week.items()):
        found.update(
            _week_conditions(
                now=now,
                season=season,
                week=week,
                receipts=receipts,
                is_target=target == (season, week),
                closeout_window=closeout_windows.get(week),
                collector_rolled=collector_rolled,
            )
        )
    for rebuild in recap_rebuilds:
        if rebuild.status in {RecapRebuildStatus.HELD, RecapRebuildStatus.STOPPED}:
            key = AlertKey(rebuild.season, rebuild.week, ATTEMPT_RECAP, rebuild.status.value)
            found[key] = _short(rebuild.reason)
    return found


def open_alerts(history: list[AlertReceipt]) -> dict[AlertKey, OpenAlert]:
    """Return the alerts whose latest notice is not ``resolved``, from oldest-first history."""
    opened: dict[AlertKey, datetime] = {}
    latest: dict[AlertKey, AlertReceipt] = {}
    for receipt in sorted(history, key=lambda item: item.decided):
        if receipt.notice == Notice.OPENED:
            opened[receipt.key] = receipt.decided
        latest[receipt.key] = receipt
    return {
        key: OpenAlert(key=key, opened_at=opened.get(key, last.decided), last=last)
        for key, last in latest.items()
        if last.notice != Notice.RESOLVED
    }


def plan_notices(
    *,
    now: datetime,
    conditions: Mapping[AlertKey, str],
    history: list[AlertReceipt],
    target: tuple[str, int] | None,
) -> list[PlannedNotice]:
    """Return the notices due now: newly open, reminded, or resolved, in key order.

    An open condition is announced once, then reminded every ``ALERT_REMINDER_INTERVAL`` from
    its latest notice; an open alert whose condition is gone is resolved once.
    """
    current = open_alerts(history)
    planned: list[PlannedNotice] = []
    for key in sorted(set(conditions) | set(current)):
        alert = current.get(key)
        if key in conditions and alert is None:
            planned.append(PlannedNotice(key, Notice.OPENED, conditions[key]))
        elif key in conditions and alert is not None:
            if is_due(alert.last.decided, ALERT_REMINDER_INTERVAL, now=now):
                planned.append(PlannedNotice(key, Notice.REMINDER, conditions[key]))
        elif key.step in TARGET_WEEK_STEPS and target != (key.season, key.week):
            detail = f"week {key.week} is no longer the target week"
            planned.append(PlannedNotice(key, Notice.RESOLVED, detail))
        else:
            planned.append(PlannedNotice(key, Notice.RESOLVED, "the condition has cleared"))
    return planned


# ---------------------------------------------------------------------------
# Alert receipts
# ---------------------------------------------------------------------------


def alert_directory(repo: Path) -> Path:
    """Return the directory of every alert receipt."""
    return repo / "data" / "output" / "week_cycle" / "alerts"


def write_alert_receipt(repo: Path, receipt: AlertReceipt) -> Path:
    """Persist one notice as a new file; an existing receipt is never replaced."""
    name = (
        f"{receipt_stamp(receipt.decided)}_{receipt.season}_week{receipt.week:02d}_"
        f"{receipt.step}_{receipt.condition}_{receipt.notice}.json"
    )
    return write_new_json(alert_directory(repo) / name, asdict(receipt))


def read_alert_receipts(repo: Path) -> list[AlertReceipt]:
    """Return every alert receipt oldest first; an unreadable file is an error, not skipped."""
    directory = alert_directory(repo)
    if not directory.is_dir():
        return []
    receipts: list[AlertReceipt] = []
    for path in sorted(directory.glob("*.json")):
        try:
            receipts.append(AlertReceipt(**json.loads(path.read_text())))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unreadable weekly-cycle alert receipt {path}: {exc}") from exc
    return receipts
