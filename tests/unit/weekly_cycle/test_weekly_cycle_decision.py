# tests/unit/weekly_cycle/test_weekly_cycle_decision.py

"""Tests for the weekly-cycle cadence and the state-driven decision (``DECISIONS.md`` D82)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pandas import DataFrame
import pytest

from gridiron_edge.weekly_cycle.cadence import closeout_earliest, is_due, watch_interval
from gridiron_edge.weekly_cycle.decision import (
    CycleDecision,
    StepKind,
    decide,
    refresh_allowed,
)
from gridiron_edge.weekly_cycle.receipts import (
    ATTEMPT_CLOSEOUT,
    ATTEMPT_FORECAST,
    ATTEMPT_REFRESH,
    ATTEMPT_WATCH,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    OUTCOME_UNCHANGED,
    OUTCOME_WAITING,
    CycleReceipt,
)

pytestmark = pytest.mark.unit

SEASON = "2026-2027"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _game(week: int, game_id: str, date: str, time: str) -> dict[str, object]:
    return {
        "season": SEASON,
        "week": week,
        "game_id": game_id,
        "game_date": date,
        "game_time": time,
    }


def _schedule() -> DataFrame:
    # Kickoffs are Eastern wall times (EDT, UTC-4 in October).
    return DataFrame(
        [
            _game(4, "w4-sun", "2026-10-04", "13:00:00"),
            _game(4, "w4-mon", "2026-10-05", "20:15:00"),  # 2026-10-06T00:15Z
            _game(5, "w5-thu", "2026-10-08", "20:15:00"),  # 2026-10-09T00:15Z
            _game(5, "w5-sun", "2026-10-11", "13:00:00"),
        ]
    )


def _receipt(week: int, attempted: str, outcome: str, at: str) -> CycleReceipt:
    return CycleReceipt(
        decided_at=_utc(at).isoformat(),
        season=SEASON,
        week=week,
        attempted=attempted,
        outcome=outcome,
        detail="",
    )


def _decide(
    now: str,
    *,
    receipts: list[CycleReceipt] | None = None,
    products: set[int] | None = None,
    schedule: DataFrame | None = None,
    target: tuple[str, int] | None = None,
) -> CycleDecision:
    stored = receipts or []
    return decide(
        now=_utc(now),
        schedule=_schedule() if schedule is None else schedule,
        receipts_for=lambda season, week: [
            r for r in stored if r.season == season and r.week == week
        ],
        has_product=lambda season, week: week in (products or set()),
        tz=UTC,
        target=target,
    )


@pytest.mark.parametrize(
    ("hours", "interval"),
    [
        (200.0, 24),
        (72.01, 24),
        (72.0, 12),
        (60.0, 12),
        (48.0, 6),
        (30.0, 6),
        (24.0, 3),
        (7.0, 3),
        (6.0, 1),
        (0.5, 1),
        (0.0, 1),
    ],
)
def test_watch_interval_follows_the_tiers_with_inclusive_upper_bounds(
    hours: float, interval: int
) -> None:
    assert watch_interval(hours) == timedelta(hours=interval)


def test_a_step_is_due_a_tick_slack_before_its_interval() -> None:
    now = _utc("2026-10-06T15:00:00")
    assert is_due(None, timedelta(hours=1), now=now)
    assert is_due(now - timedelta(minutes=56), timedelta(hours=1), now=now)
    assert not is_due(now - timedelta(minutes=50), timedelta(hours=1), now=now)


def test_closeout_waits_for_six_local_after_the_last_game_is_over() -> None:
    assert closeout_earliest(_utc("2026-10-06T00:15:00"), tz=UTC) == _utc("2026-10-06T06:00:00")
    assert closeout_earliest(_utc("2026-10-04T17:00:00"), tz=UTC) == _utc("2026-10-05T06:00:00")
    # A game that ends exactly at 06:00 may close at once.
    assert closeout_earliest(_utc("2026-10-06T02:00:00"), tz=UTC) == _utc("2026-10-06T06:00:00")


def test_while_a_game_is_ahead_the_target_is_that_week_and_the_watch_runs() -> None:
    decision = _decide("2026-10-05T23:00:00", products={4})

    assert (decision.kind, decision.week, decision.target_week) == (StepKind.WATCH, 4, 4)
    assert decision.started is True
    assert decision.hours_to_kickoff == pytest.approx(1.25)


def test_the_finished_week_is_not_closed_before_the_window_opens() -> None:
    early = _decide("2026-10-06T01:00:00")

    assert early.kind is StepKind.IDLE
    assert early.week == 4
    assert early.next_check_at == _utc("2026-10-06T06:00:00")


def test_the_finished_week_is_closed_once_the_window_opens() -> None:
    decision = _decide("2026-10-06T06:30:00")

    assert (decision.kind, decision.season, decision.week, decision.target_week) == (
        StepKind.CLOSEOUT,
        SEASON,
        4,
        5,
    )


def test_a_closeout_still_waiting_is_retried_hourly_with_tick_slack() -> None:
    waiting = [_receipt(4, ATTEMPT_CLOSEOUT, OUTCOME_WAITING, "2026-10-06T06:10:00")]

    soon = _decide("2026-10-06T06:40:00", receipts=waiting)
    later = _decide("2026-10-06T07:07:00", receipts=waiting)

    assert soon.kind is StepKind.IDLE
    assert soon.next_check_at == _utc("2026-10-06T07:05:00")
    assert later.kind is StepKind.CLOSEOUT


def test_a_closed_week_leads_to_the_forecast_of_the_target_week() -> None:
    closed = [_receipt(4, ATTEMPT_CLOSEOUT, OUTCOME_DONE, "2026-10-06T06:30:00")]

    decision = _decide("2026-10-06T06:45:00", receipts=closed)

    assert (decision.kind, decision.week) == (StepKind.FORECAST, 5)


def test_a_failed_forecast_is_retried_after_an_hour() -> None:
    receipts = [
        _receipt(4, ATTEMPT_CLOSEOUT, OUTCOME_DONE, "2026-10-06T06:30:00"),
        _receipt(5, ATTEMPT_FORECAST, OUTCOME_FAILED, "2026-10-06T06:50:00"),
    ]

    assert _decide("2026-10-06T07:10:00", receipts=receipts).kind is StepKind.IDLE
    assert _decide("2026-10-06T07:47:00", receipts=receipts).kind is StepKind.FORECAST


def test_an_existing_forecast_means_the_previous_week_is_not_closed_again() -> None:
    # No closeout receipt, but week 5 already has a product (closed by hand before the cycle).
    decision = _decide("2026-10-06T08:00:00", products={5})

    assert (decision.kind, decision.week) == (StepKind.WATCH, 5)


@pytest.mark.parametrize(
    ("now", "last_check", "due"),
    [
        ("2026-10-06T08:00:00", "2026-10-05T20:00:00", True),  # 64 h out: 12 h, last 12 h ago
        ("2026-10-06T08:00:00", "2026-10-05T23:00:00", False),  # last only 9 h ago
        ("2026-10-08T12:00:00", "2026-10-08T05:30:00", True),  # 12 h out: 3 h tier
        ("2026-10-08T12:00:00", "2026-10-08T10:00:00", False),
        ("2026-10-08T22:00:00", "2026-10-08T20:55:00", True),  # 2 h out: hourly
    ],
)
def test_the_watch_runs_at_the_cadence_of_the_hours_to_kickoff(
    now: str, last_check: str, due: bool
) -> None:
    receipts = [_receipt(5, ATTEMPT_WATCH, OUTCOME_UNCHANGED, last_check)]

    decision = _decide(now, receipts=receipts, products={5})

    assert (decision.kind is StepKind.WATCH) is due
    assert decision.kind in {StepKind.WATCH, StepKind.IDLE}
    if not due:
        assert decision.next_check_at is not None
        assert decision.next_check_at > _utc(now)


def test_a_forecast_or_refresh_counts_as_a_starter_check() -> None:
    forecast = [_receipt(5, ATTEMPT_FORECAST, OUTCOME_DONE, "2026-10-06T07:50:00")]
    refresh = [_receipt(5, ATTEMPT_REFRESH, OUTCOME_DONE, "2026-10-06T07:50:00")]

    assert _decide("2026-10-06T08:00:00", receipts=forecast, products={5}).kind is StepKind.IDLE
    assert _decide("2026-10-06T08:00:00", receipts=refresh, products={5}).kind is StepKind.IDLE


def test_a_failed_check_is_retried_hourly_not_at_the_tier() -> None:
    failed = [_receipt(5, ATTEMPT_WATCH, OUTCOME_FAILED, "2026-10-06T08:00:00")]

    assert _decide("2026-10-06T08:30:00", receipts=failed, products={5}).kind is StepKind.IDLE
    assert _decide("2026-10-06T09:02:00", receipts=failed, products={5}).kind is StepKind.WATCH


def test_after_the_first_kickoff_of_the_target_week_the_decision_says_started() -> None:
    before = _decide("2026-10-08T12:00:00", products={5})
    after = _decide("2026-10-09T06:00:00", products={5})

    assert before.started is False
    assert after.started is True
    assert after.hours_to_kickoff == pytest.approx(
        (_utc("2026-10-11T17:00:00") - _utc("2026-10-09T06:00:00")).total_seconds() / 3600
    )


def test_week_one_is_forecast_without_a_closeout() -> None:
    schedule = DataFrame(
        [_game(1, "w1", "2026-09-10", "20:20:00"), _game(2, "w2", "2026-09-17", "20:15:00")]
    )

    decision = _decide("2026-08-20T12:00:00", schedule=schedule)

    assert (decision.kind, decision.week) == (StepKind.FORECAST, 1)


def test_the_last_week_is_closed_and_then_the_season_is_complete() -> None:
    schedule = DataFrame([_game(17, "w17", "2027-01-03", "13:00:00")])

    open_week = _decide("2027-01-05T08:00:00", schedule=schedule)
    closed = _decide(
        "2027-01-05T08:00:00",
        schedule=schedule,
        receipts=[_receipt(17, ATTEMPT_CLOSEOUT, OUTCOME_DONE, "2027-01-05T07:00:00")],
    )

    assert (open_week.kind, open_week.week, open_week.target_week) == (StepKind.CLOSEOUT, 17, None)
    assert closed.kind is StepKind.SEASON_COMPLETE


def test_an_empty_schedule_is_idle_and_a_target_override_wins() -> None:
    empty = _decide("2026-10-06T08:00:00", schedule=DataFrame(columns=["season", "week"]))
    override = _decide("2026-10-06T08:00:00", products={4}, target=(SEASON, 4))

    assert empty.kind is StepKind.IDLE
    assert (override.kind, override.week) == (StepKind.WATCH, 4)
    assert override.started is True


def test_a_refresh_is_allowed_once_an_hour() -> None:
    now = _utc("2026-10-06T09:00:00")
    recent = [_receipt(5, ATTEMPT_REFRESH, OUTCOME_FAILED, "2026-10-06T08:30:00")]
    old = [_receipt(5, ATTEMPT_FORECAST, OUTCOME_DONE, "2026-10-06T07:30:00")]

    assert refresh_allowed([], now=now)
    assert not refresh_allowed(recent, now=now)
    assert refresh_allowed(old, now=now)
