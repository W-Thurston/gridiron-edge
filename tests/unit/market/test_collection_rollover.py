# tests/unit/market/test_collection_rollover.py

"""Tests for the automatic weekly collection-plan rollover."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.collection_plan_store import (
    read_current_collection_plan_selection,
    select_current_collection_plan,
)
from gridiron_edge.market.collection_rollover import (
    RECEIPT_DIRECTORY,
    auto_rollover,
    select_target_week,
)

pytestmark = pytest.mark.unit

SEASON = "2026-2027"


def _game(week: int, game_id: str, date: str, time: str) -> dict[str, object]:
    return {
        "season": SEASON,
        "week": week,
        "game_id": game_id,
        "game_date": date,
        "game_time": time,
    }


def _schedule() -> DataFrame:
    return DataFrame(
        [
            _game(4, "w4-sun", "2026-10-04", "13:00:00"),
            _game(4, "w4-mon", "2026-10-05", "20:15:00"),
            _game(5, "w5-thu", "2026-10-08", "20:15:00"),
            _game(5, "w5-sun", "2026-10-11", "13:00:00"),
        ]
    )


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def _repo(tmp_path: Path, schedule: DataFrame | None = None) -> Path:
    path = tmp_path / "data" / "cleaned"
    path.mkdir(parents=True)
    (schedule if schedule is not None else _schedule()).to_parquet(
        path / "NFL_upcoming_schedule_rich.parquet"
    )
    return tmp_path


def _select(repo: Path, week: int, *, now: str = "2026-10-03T12:00:00+00:00") -> None:
    from gridiron_edge.market.collection_plan import build_weekly_quote_collection_plan
    from gridiron_edge.market.collection_plan_store import write_collection_plan

    when = _utc(now)
    plan = build_weekly_quote_collection_plan(
        pd.read_parquet(repo / "data/cleaned/NFL_upcoming_schedule_rich.parquet"),
        season=SEASON,
        week=week,
        plan_start=when,
        created_at=when,
    )
    write_collection_plan(plan, repo=repo)
    select_current_collection_plan(season=SEASON, week=week, selected_at=when, repo=repo)


def _receipts(repo: Path) -> list[dict[str, object]]:
    directory = repo / "data" / "odds" / RECEIPT_DIRECTORY
    if not directory.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def test_target_is_the_earliest_week_with_a_game_still_ahead() -> None:
    schedule = _schedule()

    assert select_target_week(schedule, now=_utc("2026-10-03T12:00:00+00:00")) == (SEASON, 4)
    # Monday night's game (20:15 ET = 00:15Z) is still ahead: week 4 stays the target.
    assert select_target_week(schedule, now=_utc("2026-10-06T00:00:00+00:00")) == (SEASON, 4)
    assert select_target_week(schedule, now=_utc("2026-10-06T00:30:00+00:00")) == (SEASON, 5)
    assert select_target_week(schedule, now=_utc("2026-10-12T00:00:00+00:00")) is None


def test_target_requires_a_timezone_aware_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        select_target_week(_schedule(), now=datetime(2026, 10, 3))


def test_not_due_touches_nothing_and_never_refreshes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)
    calls: list[str] = []

    receipt = auto_rollover(repo=repo, now=_utc("2026-10-04T12:00:00+00:00"), refresh=calls.append)

    assert receipt.outcome == "not_due"
    assert calls == []
    assert _receipts(repo) == []


def test_due_rollover_refreshes_builds_selects_and_writes_a_receipt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)
    calls: list[str] = []
    now = _utc("2026-10-06T12:00:00+00:00")

    receipt = auto_rollover(repo=repo, now=now, refresh=calls.append)

    assert receipt.outcome == "rolled"
    assert receipt.target == f"{SEASON} week 5"
    assert receipt.schedule_refreshed is True
    assert receipt.planned_polls and receipt.planned_polls > 0
    assert calls == [SEASON]
    selection = read_current_collection_plan_selection(repo=repo)
    assert (selection.season, selection.week, selection.selected_at) == (SEASON, 5, now)
    [stored] = _receipts(repo)
    assert stored["outcome"] == "rolled"
    assert stored["planned_credit_cost"] == receipt.planned_credit_cost


def test_a_second_run_is_not_due(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)
    now = _utc("2026-10-06T12:00:00+00:00")
    auto_rollover(repo=repo, now=now, refresh=None)

    again = auto_rollover(repo=repo, now=now + timedelta(minutes=5), refresh=None)

    assert again.outcome == "not_due"
    assert len(_receipts(repo)) == 1


def test_a_missing_selection_rolls_to_the_target(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    receipt = auto_rollover(repo=repo, now=_utc("2026-10-03T12:00:00+00:00"), refresh=None)

    assert receipt.outcome == "rolled"
    assert receipt.current is None
    assert receipt.target == f"{SEASON} week 4"


def test_an_existing_plan_is_selected_without_being_rebuilt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 5, now="2026-10-01T12:00:00+00:00")
    plan_file = repo / f"data/odds/collection_plans/season={SEASON}/week=05.json"
    original = plan_file.read_bytes()
    _select_other = _utc("2026-10-02T12:00:00+00:00")
    select_current_collection_plan(season=SEASON, week=5, selected_at=_select_other, repo=repo)
    _select(repo, 4, now="2026-10-02T13:00:00+00:00")

    receipt = auto_rollover(repo=repo, now=_utc("2026-10-06T12:00:00+00:00"), refresh=None)

    assert receipt.outcome == "selected_existing"
    assert plan_file.read_bytes() == original


def test_a_failed_refresh_keeps_the_existing_schedule_and_is_recorded(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)

    def broken(_: str) -> None:
        raise ConnectionError("nflverse unreachable")

    receipt = auto_rollover(repo=repo, now=_utc("2026-10-06T12:00:00+00:00"), refresh=broken)

    assert receipt.outcome == "rolled"
    assert receipt.schedule_refreshed is False
    assert "nflverse unreachable" in (receipt.schedule_refresh_error or "")
    assert read_current_collection_plan_selection(repo=repo).week == 5


def test_a_refresh_that_brings_new_kickoffs_is_used_and_reported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)
    _select(repo, 5, now="2026-10-03T12:00:00+00:00")
    select_current_collection_plan(
        season=SEASON, week=4, selected_at=_utc("2026-10-03T13:00:00+00:00"), repo=repo
    )
    flexed = _schedule()
    flexed.loc[flexed["game_id"] == "w5-sun", "game_time"] = "16:25:00"

    def refresh(_: str) -> None:
        flexed.to_parquet(repo / "data/cleaned/NFL_upcoming_schedule_rich.parquet")

    receipt = auto_rollover(repo=repo, now=_utc("2026-10-06T12:00:00+00:00"), refresh=refresh)

    assert receipt.outcome == "selected_existing"
    assert receipt.kickoff_groups_changed is True


def test_the_season_complete_state_rolls_nothing_and_is_debounced(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 5)
    calls: list[str] = []
    now = _utc("2026-10-12T12:00:00+00:00")

    first = auto_rollover(repo=repo, now=now, refresh=calls.append)
    second = auto_rollover(repo=repo, now=now + timedelta(minutes=5), refresh=calls.append)
    later = auto_rollover(repo=repo, now=now + timedelta(hours=2), refresh=calls.append)

    assert first.outcome == "season_complete"
    assert second.outcome == "deferred"
    assert later.outcome == "season_complete"
    assert len(calls) == 2
    assert read_current_collection_plan_selection(repo=repo).week == 5
    assert len(_receipts(repo)) == 2


def test_a_missing_schedule_fails_closed_and_is_debounced(tmp_path: Path) -> None:
    receipt = auto_rollover(repo=tmp_path, now=_utc("2026-10-06T12:00:00+00:00"), refresh=None)
    again = auto_rollover(repo=tmp_path, now=_utc("2026-10-06T12:05:00+00:00"), refresh=None)

    assert receipt.outcome == "failed"
    assert again.outcome == "deferred"
    assert not (tmp_path / "data/odds/collection_plans/current.json").exists()


def test_a_week_with_no_usable_plan_is_not_selected(tmp_path: Path) -> None:
    schedule = _schedule()
    schedule.loc[schedule["week"] == 5, "game_time"] = None
    repo = _repo(tmp_path, schedule)
    _select(repo, 4)

    now = _utc("2026-10-06T12:00:00+00:00")
    first = auto_rollover(repo=repo, now=now, refresh=None)
    again = auto_rollover(repo=repo, now=now + timedelta(minutes=5), refresh=None)

    assert first.outcome == "failed"
    assert "kickoff" in first.detail
    assert again.outcome == "deferred"
    assert read_current_collection_plan_selection(repo=repo).week == 4
    assert len(_receipts(repo)) == 1


def test_dry_run_reports_without_refreshing_or_writing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _select(repo, 4)
    calls: list[str] = []
    before = (repo / "data/odds/collection_plans/current.json").read_bytes()

    receipt = auto_rollover(
        repo=repo, now=_utc("2026-10-06T12:00:00+00:00"), refresh=calls.append, dry_run=True
    )

    assert receipt.outcome == "rolled"
    assert "dry run" in receipt.detail
    assert calls == []
    assert (repo / "data/odds/collection_plans/current.json").read_bytes() == before
    assert not (repo / f"data/odds/collection_plans/season={SEASON}/week=05.json").exists()
    assert _receipts(repo) == []
