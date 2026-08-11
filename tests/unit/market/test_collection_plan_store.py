"""Tests for collection-plan JSON persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

from pandas import DataFrame
import pytest

from gridiron_edge.market.collection_plan import build_weekly_quote_collection_plan
from gridiron_edge.market.collection_plan_store import (
    CURRENT_COLLECTION_PLAN_SCHEMA_VERSION,
    collection_plan_path,
    current_collection_plan_path,
    load_current_collection_plan,
    read_collection_plan,
    read_current_collection_plan_selection,
    select_current_collection_plan,
    write_collection_plan,
)


def _plan():
    schedule = DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g",
                "game_date": "2026-09-10",
                "game_time": "20:20:00",
            }
        ]
    )
    return build_weekly_quote_collection_plan(
        schedule,
        season="2026-2027",
        week=1,
        plan_start=datetime(2026, 9, 8, 12, tzinfo=UTC),
        created_at=datetime(2026, 8, 11, 14, tzinfo=UTC),
    )


def test_plan_path_is_deterministic(tmp_path: Path) -> None:
    assert collection_plan_path(season="2026-2027", week=1, repo=tmp_path) == (
        tmp_path / "data" / "odds" / "collection_plans" / "season=2026-2027" / "week=01.json"
    )


def test_round_trip_is_exact(tmp_path: Path) -> None:
    plan = _plan()
    path = write_collection_plan(plan, repo=tmp_path)
    assert read_collection_plan(season="2026-2027", week=1, repo=tmp_path) == plan
    assert json.loads(path.read_text())["schema_version"] == 1


def test_write_rejects_invalid_plan_without_replacing_prior(tmp_path: Path) -> None:
    plan = _plan()
    path = write_collection_plan(plan, repo=tmp_path)
    before = path.read_bytes()
    broken = replace(plan, planned_credit_cost=999)
    with pytest.raises(ValueError):
        write_collection_plan(broken, repo=tmp_path)
    assert path.read_bytes() == before


def test_read_rejects_unknown_schema(tmp_path: Path) -> None:
    path = write_collection_plan(_plan(), repo=tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        read_collection_plan(season="2026-2027", week=1, repo=tmp_path)


def test_current_selection_path_is_deterministic(tmp_path: Path) -> None:
    assert current_collection_plan_path(repo=tmp_path) == (
        tmp_path / "data" / "odds" / "collection_plans" / "current.json"
    )


def test_writing_plan_does_not_select_it(tmp_path: Path) -> None:
    write_collection_plan(_plan(), repo=tmp_path)
    assert not current_collection_plan_path(repo=tmp_path).exists()


def test_selects_and_loads_existing_plan(tmp_path: Path) -> None:
    plan = _plan()
    write_collection_plan(plan, repo=tmp_path)
    selection = select_current_collection_plan(
        season=plan.season,
        week=plan.week,
        selected_at=datetime(2026, 8, 11, 18, tzinfo=UTC),
        repo=tmp_path,
    )
    assert selection.schema_version == CURRENT_COLLECTION_PLAN_SCHEMA_VERSION
    assert read_current_collection_plan_selection(repo=tmp_path) == selection
    assert load_current_collection_plan(repo=tmp_path) == plan


def test_selection_requires_existing_plan(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_current_collection_plan(
            season="2026-2027",
            week=1,
            selected_at=datetime(2026, 8, 11, 18, tzinfo=UTC),
            repo=tmp_path,
        )


def test_missing_current_selection_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_current_collection_plan_selection(repo=tmp_path)


def test_read_rejects_unknown_selection_schema(tmp_path: Path) -> None:
    path = current_collection_plan_path(repo=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "season": "2026-2027",
                "week": 1,
                "selected_at": "2026-08-11T18:00:00Z",
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_current_collection_plan_selection(repo=tmp_path)


def test_failed_selection_preserves_previous_selection(tmp_path: Path) -> None:
    plan = _plan()
    write_collection_plan(plan, repo=tmp_path)
    select_current_collection_plan(
        season=plan.season,
        week=plan.week,
        selected_at=datetime(2026, 8, 11, 18, tzinfo=UTC),
        repo=tmp_path,
    )
    path = current_collection_plan_path(repo=tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileNotFoundError):
        select_current_collection_plan(
            season="2026-2027",
            week=2,
            selected_at=datetime(2026, 8, 11, 19, tzinfo=UTC),
            repo=tmp_path,
        )
    assert path.read_bytes() == before
