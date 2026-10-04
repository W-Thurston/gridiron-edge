# tests/unit/weekly_cycle/test_weekly_cycle_receipts_and_watch.py

"""Tests for weekly-cycle receipts and the starter comparison (``DECISIONS.md`` D82)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.models.game_prediction.weekly_starting_qb import STARTING_QB_COLUMNS
from gridiron_edge.weekly_cycle.receipts import (
    ATTEMPT_CLOSEOUT,
    ATTEMPT_WATCH,
    OUTCOME_DONE,
    OUTCOME_FAILED,
    CycleReceipt,
    last_receipt,
    read_receipts,
    week_directory,
    write_receipt,
)
from gridiron_edge.weekly_cycle.starter_watch import (
    compare_starters,
    frame_digest,
    observe_starters,
    team_evidence,
    unstarted_game_ids,
)

pytestmark = pytest.mark.unit

SEASON = "2026-2027"


def _receipt(
    at: str, attempted: str = ATTEMPT_CLOSEOUT, outcome: str = OUTCOME_DONE
) -> CycleReceipt:
    return CycleReceipt(
        decided_at=datetime.fromisoformat(at).replace(tzinfo=UTC).isoformat(),
        season=SEASON,
        week=4,
        attempted=attempted,
        outcome=outcome,
        detail="d",
        inputs={"a": 1},
        results={"b": [2]},
    )


def test_receipts_round_trip_oldest_first_under_the_week_directory(tmp_path: Path) -> None:
    later = _receipt("2026-10-06T07:00:00", ATTEMPT_WATCH)
    earlier = _receipt("2026-10-06T06:00:00")

    path = write_receipt(tmp_path, later)
    write_receipt(tmp_path, earlier)

    assert path.parent == week_directory(tmp_path, season=SEASON, week=4)
    assert path.parent == tmp_path / "data/output/week_cycle/season=2026-2027/week=04"
    assert read_receipts(tmp_path, season=SEASON, week=4) == [earlier, later]
    assert read_receipts(tmp_path, season=SEASON, week=9) == []
    assert json.loads(path.read_text())["schema_version"] == 1


def test_a_receipt_is_never_replaced(tmp_path: Path) -> None:
    receipt = _receipt("2026-10-06T06:00:00")
    path = write_receipt(tmp_path, receipt)
    before = path.read_text()

    with pytest.raises(FileExistsError):
        write_receipt(tmp_path, replace(receipt, detail="changed"))

    assert path.read_text() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_an_unreadable_receipt_is_an_error_not_a_skip(tmp_path: Path) -> None:
    path = write_receipt(tmp_path, _receipt("2026-10-06T06:00:00"))
    path.write_text("{not json")

    with pytest.raises(ValueError, match="Unreadable weekly-cycle receipt"):
        read_receipts(tmp_path, season=SEASON, week=4)


def test_last_receipt_filters_by_attempt_and_outcome() -> None:
    receipts = [
        _receipt("2026-10-06T06:00:00", ATTEMPT_CLOSEOUT, OUTCOME_FAILED),
        _receipt("2026-10-06T07:00:00", ATTEMPT_CLOSEOUT, OUTCOME_DONE),
        _receipt("2026-10-06T08:00:00", ATTEMPT_WATCH, OUTCOME_DONE),
    ]

    assert last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT}) == receipts[1]
    assert (
        last_receipt(receipts, attempted={ATTEMPT_CLOSEOUT}, outcomes={OUTCOME_FAILED})
        == receipts[0]
    )
    assert last_receipt([], attempted={ATTEMPT_WATCH}) is None


def _product(**overrides: list[object]) -> DataFrame:
    base: dict[str, list[object]] = {
        "game_id": ["g1", "g2"],
        "starting_qb_status": ["available", "available"],
    }
    for side in ("away", "home"):
        for field in (
            "qb_id",
            "qb_name",
            "qb_source",
            "qb_projection_as_of",
            "qb_nflverse_id",
            "qb_carried_forward_id",
            "qb_differs_from_carry_forward",
        ):
            base[f"{side}_{field}"] = [None, None]
    base["away_qb_id"] = ["a1", "a2"]
    base["home_qb_id"] = ["h1", None]
    base["away_qb_source"] = ["depth_chart_report", "carried_forward"]
    base["home_qb_source"] = ["nflverse_schedule", None]
    base.update(overrides)
    assert set(STARTING_QB_COLUMNS) <= set(base)
    return DataFrame(base)


def _resolved(rows: list[tuple[str, str, str | None]]) -> DataFrame:
    return DataFrame(
        {
            "GAME_ID": [row[0] for row in rows],
            "SIDE": [row[1] for row in rows],
            "RESOLVED_QB_ID": [row[2] for row in rows],
            "QB_SOURCE": "depth_chart_report",
            "PROJECTION_AS_OF": "2026-10-06T08:00:00Z",
        }
    )


def test_unchanged_starters_compare_equal_and_a_new_id_is_a_change() -> None:
    resolved = _resolved(
        [("g1", "AWAY", "a1"), ("g1", "HOME", "h1"), ("g2", "AWAY", "a9"), ("g2", "HOME", None)]
    )

    available, comparisons = compare_starters(resolved, _product(), game_ids=["g1", "g2"])

    assert available is True
    assert [(c.game_id, c.side, c.changed) for c in comparisons] == [
        ("g1", "away", False),
        ("g1", "home", False),
        ("g2", "away", True),
        ("g2", "home", False),  # recorded none, resolves none
    ]
    changed = comparisons[2]
    assert (changed.recorded_id, changed.current_id) == ("a2", "a9")
    assert changed.current_as_of == "2026-10-06T08:00:00Z"


def test_a_starter_that_newly_resolves_or_stops_resolving_is_a_change() -> None:
    resolved = _resolved([("g2", "HOME", "h2"), ("g1", "HOME", None), ("g1", "AWAY", "a1")])

    _, comparisons = compare_starters(resolved, _product(), game_ids=["g1", "g2"])
    flagged = {(c.game_id, c.side) for c in comparisons if c.changed}

    assert flagged == {("g2", "home"), ("g1", "home"), ("g2", "away")}


def test_games_that_have_started_are_not_compared() -> None:
    resolved = _resolved([("g1", "AWAY", "zzz"), ("g1", "HOME", "zzz")])

    _, comparisons = compare_starters(resolved, _product(), game_ids=["g2"])

    assert {c.game_id for c in comparisons} == {"g2"}


def test_a_product_without_the_starter_record_cannot_be_compared() -> None:
    available, comparisons = compare_starters(
        _resolved([("g1", "AWAY", "a1")]), DataFrame({"game_id": ["g1"]}), game_ids=["g1"]
    )

    assert (available, comparisons) == (False, [])


def test_the_digest_follows_content_not_serialization() -> None:
    frame = DataFrame({"a": [1, 2], "b": ["x", "y"]})

    assert frame_digest(frame) == frame_digest(frame.copy())
    assert frame_digest(frame) != frame_digest(DataFrame({"a": [1, 3], "b": ["x", "y"]}))
    assert frame_digest(frame) != frame_digest(frame.rename(columns={"b": "c"}))
    assert frame_digest(None) is None


def _schedule() -> DataFrame:
    return DataFrame(
        {
            "season": [SEASON] * 3,
            "week": [5, 5, 6],
            "game_id": ["g1", "g2", "g3"],
            "game_date": ["2026-10-08", "2026-10-11", "2026-10-18"],
            "game_time": ["20:15:00", "13:00:00", "13:00:00"],
        }
    )


def test_games_split_at_kickoff_for_the_requested_week_only() -> None:
    now = datetime(2026, 10, 9, 6, 0, tzinfo=UTC)  # after Thursday night's 00:15Z kickoff

    assert unstarted_game_ids(_schedule(), season=SEASON, week=5, now=now) == (["g2"], ["g1"])


def _write_caches(repo: Path) -> None:
    raw = repo / "data" / "raw"
    (raw / "depth_charts").mkdir(parents=True)
    (raw / "injuries").mkdir(parents=True)
    DataFrame(
        {
            "dt": ["2026-10-07T12:00:00Z"] * 2 + ["2026-10-08T12:00:00Z"] * 2,
            "team": ["IND", "IND", "IND", "IND"],
            "pos_abb": ["QB"] * 4,
            "pos_rank": [1, 2, 1, 2],
            "gsis_id": ["old1", "old2", "new1", "new2"],
        }
    ).to_parquet(raw / "depth_charts" / "depth_charts_2026.parquet")
    DataFrame(
        {
            "season": [2026],
            "week": [5],
            "game_type": ["REG"],
            "team": ["IND"],
            "gsis_id": ["new1"],
            "report_status": ["Out"],
        }
    ).to_parquet(raw / "injuries" / "injuries_2026.parquet")
    (repo / "data" / "raw" / "NFL_upcoming_schedule_nflverse.parquet").parent.mkdir(exist_ok=True)
    DataFrame(
        {"game_id": ["g1", "g2"], "away_team": ["IND", "BUF"], "home_team": ["HOU", "NYJ"]}
    ).to_parquet(repo / "data" / "raw" / "NFL_upcoming_schedule_nflverse.parquet")


def test_team_evidence_records_the_latest_chart_and_report_status(tmp_path: Path) -> None:
    _write_caches(tmp_path)

    evidence = team_evidence(tmp_path, season=2026, week=5, game_ids=["g1"])

    assert set(evidence) == {"HOU", "IND"}
    assert evidence["IND"] == {
        "chart": [{"rank": 1, "gsis_id": "new1"}, {"rank": 2, "gsis_id": "new2"}],
        "chart_as_of": "2026-10-08T12:00:00Z",
        "report_status": {"new1": "Out"},
    }
    assert evidence["HOU"]["chart"] == []
    assert team_evidence(tmp_path, season=2026, week=5, game_ids=[]) == {}


def test_team_evidence_is_empty_without_the_caches(tmp_path: Path) -> None:
    assert team_evidence(tmp_path, season=2026, week=5, game_ids=["g1"]) == {}


def _sidecar() -> DataFrame:
    columns = ["GAME_ID", "YEAR", "WEEK_NUM", "AWAY_TEAM", "HOME_TEAM", "AWAY_QB_ID", "HOME_QB_ID"]
    return DataFrame(
        [
            ["g0", SEASON, 4, "A", "B", "a1", "h1"],
            ["g1", SEASON, 5, "A", "B", None, None],
            ["g2", SEASON, 5, "C", "D", "c1", None],
        ],
        columns=columns,
    )


def test_observing_resolves_the_unstarted_games_and_skips_started_ones(tmp_path: Path) -> None:
    product = _product(
        away_qb_id=["a1", "c1"],
        home_qb_id=["h1", "d9"],
    )
    now = datetime(2026, 10, 9, 6, 0, tzinfo=UTC)
    schedule = _schedule()

    with patch(
        "gridiron_edge.weekly_cycle.starter_watch.build_qb_coach_sidecar_frame",
        return_value=_sidecar(),
    ):
        observation = observe_starters(
            tmp_path, season=SEASON, week=5, product=product, schedule=schedule, now=now
        )

    assert observation.record_available is True
    assert observation.started_game_ids == ("g1",)
    assert {c.game_id for c in observation.comparisons} == {"g2"}
    # g2 home: nothing is published and the carry-forward has no earlier starter for team D.
    assert [(c.side, c.recorded_id, c.current_id) for c in observation.comparisons] == [
        ("away", "c1", "c1"),
        ("home", "d9", None),
    ]
    assert [c.side for c in observation.changed] == ["home"]
    results = observation.as_results()
    assert results["changed_sides"] == 1
    assert results["started_game_ids"] == ["g1"]
    assert pd.api.types.is_dict_like(results["teams"])
