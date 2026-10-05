# tests/unit/evaluation/test_weekly_recap_store.py

"""Tests for weekly-recap storage, replay and explicit selection (``DECISIONS.md`` D86)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.weekly_recap import BUILT_AT, GAME_C, SEASON, WEEK, recap_week

from gridiron_edge.evaluation.weekly_recap import BuiltWeeklyRecap, build_weekly_recap
from gridiron_edge.evaluation.weekly_recap_store import (
    RecapWriteOutcome,
    get_current_weekly_recap_selection,
    list_current_weekly_recap_selections,
    load_current_weekly_recap,
    load_weekly_recap,
    select_current_weekly_recap,
    weekly_recap_manifest_path,
    weekly_recap_root,
    write_weekly_recap,
)

pytestmark = pytest.mark.unit

SELECTED_AT = datetime(2026, 10, 7, 13, 0, tzinfo=UTC)


def _built(*, built_at: datetime = BUILT_AT, closing_line_c: float = -1.0) -> BuiltWeeklyRecap:
    week = recap_week()
    games = week.games.copy()
    games.loc[games["GAME_ID"] == GAME_C, "VEGAS_LINE"] = closing_line_c
    return build_weekly_recap(
        season=SEASON,
        week=WEEK,
        schedule=week.schedule,
        product=week.product,
        forecast_events=week.events,
        games=games,
        built_at=built_at,
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_registered_root_is_under_data_output(tmp_path: Path) -> None:
    assert weekly_recap_root(tmp_path) == tmp_path / "data/output/weekly_recaps"


def test_a_written_recap_loads_back_exactly(tmp_path: Path) -> None:
    built = _built()

    assert write_weekly_recap(built, repo=tmp_path) is RecapWriteOutcome.WRITTEN
    recap, games = load_weekly_recap(built.recap.recap_id, repo=tmp_path)

    assert recap == built.recap
    pd.testing.assert_frame_equal(games, built.games)
    assert weekly_recap_manifest_path(recap.recap_id, repo=tmp_path).is_file()
    assert (weekly_recap_root(tmp_path) / recap.games.artifact).is_file()


def test_a_rebuild_on_unchanged_inputs_replays_and_keeps_every_file(tmp_path: Path) -> None:
    write_weekly_recap(_built(), repo=tmp_path)
    before = _files(tmp_path)

    later = _built(built_at=BUILT_AT + timedelta(days=2))
    assert write_weekly_recap(later, repo=tmp_path) is RecapWriteOutcome.REPLAYED
    assert _files(tmp_path) == before
    assert load_weekly_recap(later.recap.recap_id, repo=tmp_path)[0].built_at == BUILT_AT


def test_changed_inputs_give_a_new_recap_and_keep_the_old_one(tmp_path: Path) -> None:
    first = _built()
    second = _built(closing_line_c=-1.5)
    write_weekly_recap(first, repo=tmp_path)
    write_weekly_recap(second, repo=tmp_path)

    assert second.recap.recap_id != first.recap.recap_id
    assert load_weekly_recap(first.recap.recap_id, repo=tmp_path)[0] == first.recap
    assert load_weekly_recap(second.recap.recap_id, repo=tmp_path)[0] == second.recap


def test_a_stored_manifest_with_other_content_under_the_identity_is_rejected(
    tmp_path: Path,
) -> None:
    built = _built()
    write_weekly_recap(built, repo=tmp_path)
    path = weekly_recap_manifest_path(built.recap.recap_id, repo=tmp_path)
    payload = json.loads(path.read_text())
    payload["product_id"] = "another_product"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match its content"):
        write_weekly_recap(built, repo=tmp_path)


def test_a_tampered_game_frame_is_rejected_on_load(tmp_path: Path) -> None:
    built = _built()
    write_weekly_recap(built, repo=tmp_path)
    frame_path = weekly_recap_root(tmp_path) / built.recap.games.artifact
    tampered = built.games.copy()
    tampered.loc[0, "home_win_prob"] = 0.99
    tampered.to_parquet(frame_path, index=False)

    with pytest.raises(ValueError, match="content digest"):
        load_weekly_recap(built.recap.recap_id, repo=tmp_path)


def test_a_frame_that_does_not_match_its_manifest_is_not_written(tmp_path: Path) -> None:
    built = _built()
    mismatched = BuiltWeeklyRecap(recap=built.recap, games=built.games.iloc[:2])

    with pytest.raises(ValueError, match="row count"):
        write_weekly_recap(mismatched, repo=tmp_path)
    assert not weekly_recap_root(tmp_path).exists()


def test_selection_round_trips_and_moves_to_a_newer_recap(tmp_path: Path) -> None:
    first = _built()
    second = _built(closing_line_c=-1.5)
    write_weekly_recap(first, repo=tmp_path)
    write_weekly_recap(second, repo=tmp_path)

    assert get_current_weekly_recap_selection(season=SEASON, week=WEEK, repo=tmp_path) is None
    select_current_weekly_recap(first.recap.recap_id, selected_at=SELECTED_AT, repo=tmp_path)
    selection = get_current_weekly_recap_selection(season=SEASON, week=WEEK, repo=tmp_path)
    assert selection is not None
    assert selection.recap_id == first.recap.recap_id
    assert selection.selected_at == SELECTED_AT

    select_current_weekly_recap(second.recap.recap_id, selected_at=SELECTED_AT, repo=tmp_path)
    recap, games = load_current_weekly_recap(season=SEASON, week=WEEK, repo=tmp_path)
    assert recap == second.recap
    pd.testing.assert_frame_equal(games, second.games)


def test_selecting_an_unstored_recap_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not stored"):
        select_current_weekly_recap("0" * 64, selected_at=SELECTED_AT, repo=tmp_path)


def test_loading_an_unselected_week_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No weekly recap is selected"):
        load_current_weekly_recap(season=SEASON, week=WEEK, repo=tmp_path)


def test_selection_requires_utc(tmp_path: Path) -> None:
    built = _built()
    write_weekly_recap(built, repo=tmp_path)

    with pytest.raises(ValueError, match="UTC"):
        select_current_weekly_recap(
            built.recap.recap_id, selected_at=datetime(2026, 10, 7), repo=tmp_path
        )


def _write_current(tmp_path: Path, selections: dict[str, dict[str, object]]) -> None:
    path = weekly_recap_root(tmp_path) / "current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "selections": selections}))


def _selection_payload(season: str, week: int, selected_at: datetime) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "recap_id": f"{week:064x}",
        "selected_at": selected_at.isoformat(),
    }


def test_no_selections_list_as_empty(tmp_path: Path) -> None:
    assert list_current_weekly_recap_selections(repo=tmp_path) == ()


def test_selections_list_by_season_and_week_not_by_selection_time(tmp_path: Path) -> None:
    _write_current(
        tmp_path,
        {
            "2026-2027_week_10": _selection_payload("2026-2027", 10, SELECTED_AT),
            "2025-2026_week_18": _selection_payload(
                "2025-2026", 18, SELECTED_AT + timedelta(days=3)
            ),
            "2026-2027_week_02": _selection_payload(
                "2026-2027", 2, SELECTED_AT + timedelta(days=1)
            ),
        },
    )

    listed = list_current_weekly_recap_selections(repo=tmp_path)

    assert [(selection.season, selection.week) for selection in listed] == [
        ("2025-2026", 18),
        ("2026-2027", 2),
        ("2026-2027", 10),
    ]
    assert listed[1].recap_id == f"{2:064x}"
    assert listed[1].selected_at == SELECTED_AT + timedelta(days=1)


def test_a_selection_stored_under_another_week_key_is_rejected(tmp_path: Path) -> None:
    _write_current(tmp_path, {"2026-2027_week_03": _selection_payload("2026-2027", 4, SELECTED_AT)})

    with pytest.raises(ValueError, match="another season or week"):
        list_current_weekly_recap_selections(repo=tmp_path)
