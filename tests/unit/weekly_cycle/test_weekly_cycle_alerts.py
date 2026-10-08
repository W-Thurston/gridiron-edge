# tests/unit/weekly_cycle/test_weekly_cycle_alerts.py

"""Tests for the conditions that need a person and the alert notices they call for (D92)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gridiron_edge.weekly_cycle.alerts import (
    AlertKey,
    AlertReceipt,
    Notice,
    open_alerts,
    open_conditions,
    plan_notices,
    read_alert_receipts,
    write_alert_receipt,
)
from gridiron_edge.weekly_cycle.decision import RecapRebuild, RecapRebuildStatus
from gridiron_edge.weekly_cycle.receipts import CycleReceipt

pytestmark = pytest.mark.unit

SEASON = "2026-2027"
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def _receipt(
    attempted: str,
    outcome: str,
    *,
    week: int = 5,
    hours_ago: float = 1.0,
    detail: str = "detail",
    results: dict[str, Any] | None = None,
) -> CycleReceipt:
    return CycleReceipt(
        decided_at=(NOW - timedelta(hours=hours_ago)).isoformat(),
        season=SEASON,
        week=week,
        attempted=attempted,
        outcome=outcome,
        detail=detail,
        results=results or {},
    )


def _conditions(
    receipts: list[CycleReceipt],
    *,
    week: int = 5,
    target: tuple[str, int] | None = (SEASON, 5),
    window: datetime | None = None,
    rebuilds: list[RecapRebuild] | None = None,
    rolled: bool = True,
) -> dict[AlertKey, str]:
    return open_conditions(
        now=NOW,
        season=SEASON,
        receipts_by_week={week: receipts},
        target=target,
        closeout_windows={} if window is None else {week: window},
        recap_rebuilds=rebuilds or [],
        collector_rolled=lambda season, next_week: rolled,
    )


def _key(step: str, condition: str, week: int = 5) -> AlertKey:
    return AlertKey(SEASON, week, step, condition)


def test_a_failed_attempt_needs_a_person_until_the_next_attempt_succeeds() -> None:
    failed = _receipt("watch", "failed", hours_ago=2, detail="could not refresh")

    assert _conditions([failed]) == {_key("watch", "failed"): "could not refresh"}
    assert _conditions([failed, _receipt("watch", "unchanged")]) == {}


def test_a_blocked_forecast_or_refresh_needs_a_person() -> None:
    assert set(_conditions([_receipt("forecast", "blocked")])) == {_key("forecast", "blocked")}
    assert set(_conditions([_receipt("refresh", "blocked")])) == {_key("refresh", "blocked")}


def test_target_week_steps_count_only_on_the_target_week() -> None:
    assert _conditions([_receipt("watch", "failed")], target=(SEASON, 6)) == {}


def test_a_closeout_alerts_when_it_fails_but_waits_twelve_hours_before_alerting() -> None:
    window = NOW - timedelta(hours=11, minutes=59)
    waiting = _receipt("closeout", "waiting", week=4)

    assert _conditions([_receipt("closeout", "failed", week=4)], week=4, target=None) == {
        _key("closeout", "failed", 4): "detail"
    }
    assert _conditions([waiting], week=4, target=None, window=window) == {}
    assert set(_conditions([waiting], week=4, target=None, window=NOW - timedelta(hours=12))) == {
        _key("closeout", "waiting", 4)
    }


def test_a_soft_failed_collector_check_is_open_until_the_collector_rolls() -> None:
    closeout = _receipt(
        "closeout",
        "done",
        week=4,
        results={
            "stages": {"check-collector": "soft-failed"},
            "warnings": ["check-collector: no rollover receipt yet"],
        },
    )

    assert _conditions([closeout], week=4, target=(SEASON, 5), rolled=False) == {
        _key("check-collector", "soft-failed", 4): "no rollover receipt yet"
    }
    assert _conditions([closeout], week=4, target=(SEASON, 5), rolled=True) == {}


def test_edges_not_built_is_one_condition_from_the_forecast_or_the_edges_hook() -> None:
    forecast = _receipt(
        "forecast",
        "done",
        hours_ago=3,
        results={
            "stages": {"generate-edges": "soft-failed"},
            "warnings": ["generate-edges: edge calculation blocked: no_market_data"],
        },
    )
    hook = _receipt(
        "edges",
        "blocked",
        hours_ago=2,
        detail="edge calculation blocked: model_edges_unvalidated",
        results={"stages": {"generate-edges": "blocked"}},
    )
    built = _receipt("edges", "done", results={"stages": {"generate-edges": "ok"}})

    assert _conditions([forecast]) == {
        _key("edges", "blocked"): "edge calculation blocked: no_market_data"
    }
    assert _conditions([forecast, hook]) == {
        _key("edges", "blocked"): "edge calculation blocked: model_edges_unvalidated"
    }
    assert _conditions([forecast, hook, built]) == {}


def test_a_failed_recap_retry_is_still_retried_but_a_hook_that_raised_needs_a_person() -> None:
    retry = _receipt("recap", "failed", week=4, results={"stages": {"build-weekly-recap": "x"}})
    raised = _receipt(
        "recap", "failed", week=4, detail="OSError: disk", results={"exception": "OSError"}
    )

    assert _conditions([retry], week=4) == {}
    assert _conditions([raised], week=4) == {_key("recap", "failed", 4): "OSError: disk"}


@pytest.mark.parametrize(
    ("status", "alerts"),
    [
        (RecapRebuildStatus.DUE, False),
        (RecapRebuildStatus.WAITING, False),
        (RecapRebuildStatus.HELD, True),
        (RecapRebuildStatus.STOPPED, True),
    ],
)
def test_a_recap_rebuild_alerts_once_it_is_held_or_stopped(
    status: RecapRebuildStatus, alerts: bool
) -> None:
    rebuild = RecapRebuild(SEASON, 4, status, reason="why", last_outcome="last")

    found = _conditions([], week=4, rebuilds=[rebuild])

    assert found == ({_key("recap", status.value, 4): "why"} if alerts else {})


def test_a_long_reason_is_cut_to_one_short_line() -> None:
    found = _conditions([_receipt("watch", "failed", detail="x" * 300 + "\nmore")])

    assert len(found[_key("watch", "failed")]) == 200


def _notice(key: AlertKey, notice: Notice, *, hours_ago: float) -> AlertReceipt:
    return AlertReceipt(
        decided_at=(NOW - timedelta(hours=hours_ago)).isoformat(),
        season=key.season,
        week=key.week,
        step=key.step,
        condition=key.condition,
        notice=notice.value,
        detail="detail",
        delivery="sent",
    )


def test_a_new_condition_opens_once_and_is_reminded_daily() -> None:
    key = _key("watch", "failed")
    target = (SEASON, 5)

    opened = plan_notices(now=NOW, conditions={key: "why"}, history=[], target=target)
    quiet = plan_notices(
        now=NOW,
        conditions={key: "why"},
        history=[_notice(key, Notice.OPENED, hours_ago=23)],
        target=target,
    )
    reminded = plan_notices(
        now=NOW,
        conditions={key: "why"},
        history=[
            _notice(key, Notice.OPENED, hours_ago=48),
            _notice(key, Notice.REMINDER, hours_ago=24),
        ],
        target=target,
    )

    assert [(item.notice, item.title()) for item in opened] == [
        (Notice.OPENED, "2026-2027 week 5 watch failed")
    ]
    assert quiet == []
    assert [(item.notice, item.title()) for item in reminded] == [
        (Notice.REMINDER, "2026-2027 week 5 watch failed (still open)")
    ]


def test_a_cleared_condition_resolves_once_with_its_reason() -> None:
    watch = _key("watch", "failed")
    closeout = _key("closeout", "failed", 4)
    history = [
        _notice(watch, Notice.OPENED, hours_ago=3),
        _notice(closeout, Notice.OPENED, hours_ago=3),
    ]

    moved_on = plan_notices(now=NOW, conditions={}, history=history, target=(SEASON, 6))
    after = [*history, _notice(watch, Notice.RESOLVED, hours_ago=1)]

    assert [(item.key, item.notice, item.detail) for item in moved_on] == [
        (closeout, Notice.RESOLVED, "the condition has cleared"),
        (watch, Notice.RESOLVED, "week 5 is no longer the target week"),
    ]
    assert moved_on[1].title() == "2026-2027 week 5 watch failed resolved"
    assert [
        item.key for item in plan_notices(now=NOW, conditions={}, history=after, target=None)
    ] == [closeout]


def test_a_condition_that_returns_after_resolving_opens_again() -> None:
    key = _key("forecast", "blocked")
    history = [
        _notice(key, Notice.OPENED, hours_ago=5),
        _notice(key, Notice.RESOLVED, hours_ago=3),
    ]

    planned = plan_notices(now=NOW, conditions={key: "dirty"}, history=history, target=(SEASON, 5))

    assert [item.notice for item in planned] == [Notice.OPENED]


def test_alert_receipts_round_trip_and_the_latest_notice_decides_what_is_open(
    tmp_path: Path,
) -> None:
    key = _key("watch", "failed")
    other = _key("edges", "blocked")
    for receipt in (
        _notice(key, Notice.OPENED, hours_ago=30),
        _notice(key, Notice.REMINDER, hours_ago=6),
        _notice(other, Notice.OPENED, hours_ago=5),
        _notice(other, Notice.RESOLVED, hours_ago=1),
    ):
        write_alert_receipt(tmp_path, receipt)

    history = read_alert_receipts(tmp_path)
    current = open_alerts(history)

    assert len(history) == 4
    assert list(current) == [key]
    assert current[key].opened_at == NOW - timedelta(hours=30)
    assert current[key].last.notice == "reminder"


def test_an_unreadable_alert_receipt_is_an_error(tmp_path: Path) -> None:
    write_alert_receipt(tmp_path, _notice(_key("watch", "failed"), Notice.OPENED, hours_ago=1))
    directory = tmp_path / "data" / "output" / "week_cycle" / "alerts"
    (directory / "broken.json").write_text("{")

    with pytest.raises(ValueError, match="Unreadable weekly-cycle alert receipt"):
        read_alert_receipts(tmp_path)
