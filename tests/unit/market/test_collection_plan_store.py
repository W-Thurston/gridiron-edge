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
    collection_plan_path,
    read_collection_plan,
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
