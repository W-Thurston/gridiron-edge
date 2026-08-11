"""Tests for scheduler-neutral weekly quote collection planning."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.collection_plan import (
    CollectionPlanStatus,
    CollectionReason,
    QuoteCollectionPolicy,
    build_weekly_quote_collection_plan,
    derive_kickoff_groups,
    validate_weekly_quote_collection_plan,
)


def _schedule() -> DataFrame:
    return DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g-thu",
                "game_date": "2026-09-10",
                "game_time": "20:20:00",
            },
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g-sun-a",
                "game_date": "2026-09-13",
                "game_time": "13:00:00",
            },
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g-sun-b",
                "game_date": "2026-09-13",
                "game_time": "13:00:00",
            },
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g-mon",
                "game_date": "2026-09-14",
                "game_time": "20:15:00",
            },
        ]
    )


def _plan(*, limit: int = 34):
    return build_weekly_quote_collection_plan(
        _schedule(),
        season="2026-2027",
        week=1,
        plan_start=datetime(2026, 9, 8, 12, tzinfo=UTC),
        created_at=datetime(2026, 8, 11, 14, tzinfo=UTC),
        policy=QuoteCollectionPolicy(weekly_poll_limit=limit),
    )


def test_kickoffs_use_eastern_named_timezone_and_group_games() -> None:
    groups = derive_kickoff_groups(_schedule(), season="2026-2027", week=1)
    assert groups[0].commence_time == datetime(2026, 9, 11, 0, 20, tzinfo=UTC)
    assert groups[1].game_ids == ("g-sun-a", "g-sun-b")
    assert groups[1].commence_time == datetime(2026, 9, 13, 17, 0, tzinfo=UTC)


def test_standard_time_conversion_uses_named_timezone() -> None:
    schedule = DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 12,
                "game_id": "g",
                "game_date": "2026-11-22",
                "game_time": "13:00:00",
            }
        ]
    )
    groups = derive_kickoff_groups(schedule, season="2026-2027", week=12)
    assert groups[0].commence_time == datetime(2026, 11, 22, 18, 0, tzinfo=UTC)


def test_default_plan_is_bounded_explained_and_ordered() -> None:
    plan = _plan()
    assert plan.status is CollectionPlanStatus.AVAILABLE
    assert 0 < plan.planned_poll_count <= 34
    assert plan.planned_credit_cost == plan.planned_poll_count * 3
    assert plan.remaining_poll_capacity == 34 - plan.planned_poll_count
    assert tuple(poll.scheduled_at for poll in plan.polls) == tuple(
        sorted(poll.scheduled_at for poll in plan.polls)
    )
    assert len({poll.scheduled_at for poll in plan.polls}) == len(plan.polls)
    assert {poll.reason for poll in plan.polls} <= set(CollectionReason)
    assert all(poll.scheduled_at < poll.next_kickoff for poll in plan.polls)


def test_budget_pressure_is_explicit_and_deterministic() -> None:
    first = _plan(limit=5)
    second = _plan(limit=5)
    assert first == second
    assert first.planned_poll_count == 5
    assert first.omitted_candidate_count > 0


def test_empty_scope_is_unavailable() -> None:
    plan = build_weekly_quote_collection_plan(
        _schedule(),
        season="2026-2027",
        week=2,
        plan_start=datetime(2026, 9, 8, 12, tzinfo=UTC),
        created_at=datetime(2026, 8, 11, 14, tzinfo=UTC),
    )
    assert plan.status is CollectionPlanStatus.SCHEDULE_UNAVAILABLE
    assert plan.polls == ()


@pytest.mark.parametrize("column", ["game_id", "game_date", "game_time"])
def test_invalid_schedule_values_are_rejected(column: str) -> None:
    schedule = _schedule()
    schedule.loc[0, column] = None
    with pytest.raises(ValueError):
        derive_kickoff_groups(schedule, season="2026-2027", week=1)


def test_duplicate_game_ids_are_rejected() -> None:
    schedule = _schedule()
    schedule.loc[1, "game_id"] = "g-thu"
    with pytest.raises(ValueError, match="unique"):
        derive_kickoff_groups(schedule, season="2026-2027", week=1)


def test_schedule_input_is_not_mutated() -> None:
    schedule = _schedule()
    expected = schedule.copy(deep=True)
    build_weekly_quote_collection_plan(
        schedule,
        season="2026-2027",
        week=1,
        plan_start=datetime(2026, 9, 8, 12, tzinfo=UTC),
        created_at=datetime(2026, 8, 11, 14, tzinfo=UTC),
    )
    pd.testing.assert_frame_equal(schedule, expected)


def test_semantic_validation_rejects_over_budget_plan() -> None:
    plan = _plan(limit=5)
    broken = replace(plan, planned_poll_count=6)
    with pytest.raises(ValueError):
        validate_weekly_quote_collection_plan(broken)
