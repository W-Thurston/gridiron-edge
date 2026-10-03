# tests/unit/transform/clean/test_qb_coach_sidecar_projection.py

"""Tests for the projected starting-QB columns of the QB-and-coach sidecar (D75)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.loaders import load_qb_coach_sidecar
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.features.team.qb_profile import resolve_starting_qb_history
from gridiron_edge.transform.clean.qb_coach_sidecar import (
    PROJECTION_COLUMNS,
    build_qb_coach_sidecar,
    project_unplayed_starters,
)

pytestmark = pytest.mark.unit


def _upcoming() -> DataFrame:
    return DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [3, 3, 4],
            "game_type": ["REG", "REG", "REG"],
            "game_id": ["2025_03_KC_DEN", "2025_03_BUF_MIA", "2025_04_KC_BUF"],
            "away_team": ["KC", "BUF", "KC"],
            "home_team": ["DEN", "MIA", "BUF"],
            "result": [None, None, None],
            "away_qb_id": [None, None, None],
            "away_qb_name": [None, None, None],
            "home_qb_id": [None, None, None],
            "home_qb_name": [None, None, None],
            "away_coach": ["c1", "c2", "c1"],
            "home_coach": ["c3", "c4", "c2"],
        }
    )


def _snapshots() -> DataFrame:
    def rows(when: str, team: str, ids: list[str]) -> list[dict[str, object]]:
        return [
            {"dt": when, "team": team, "pos_abb": "QB", "pos_rank": rank, "gsis_id": player}
            for rank, player in enumerate(ids, start=1)
        ]

    return DataFrame(
        rows("2025-09-18T08:00:00Z", "KC", ["00-A", "00-B"])
        + rows("2025-09-19T09:30:15Z", "KC", ["00-B", "00-A"])
        + rows("2025-09-19T09:30:15Z", "DEN", ["00-X", "00-Y"])
        + rows("2025-09-19T09:30:15Z", "BUF", ["00-L", "00-M"])
    )


def _injuries(*, week: int = 3) -> DataFrame:
    return DataFrame(
        {
            "season": [2025, 2025],
            "week": [week, week],
            "game_type": ["REG", "REG"],
            "team": ["DEN", "BUF"],
            "gsis_id": ["00-X", "00-L"],
            "report_status": ["Out", "Questionable"],
        }
    )


def _projection(snapshots: DataFrame | None, injuries: DataFrame | None) -> DataFrame:
    return project_unplayed_starters(_upcoming(), snapshots, injuries).set_index("game_id")


def test_projects_the_earliest_unplayed_week_from_the_latest_chart_and_report() -> None:
    result = _projection(_snapshots(), _injuries())

    kc = result.loc["2025_03_KC_DEN"]
    assert kc["AWAY_QB_ID_PROJECTED"] == "00-B"  # latest snapshot ranks 00-B first
    assert kc["HOME_QB_ID_PROJECTED"] == "00-Y"  # 00-X is reported Out
    assert kc["AWAY_QB_PROJECTION_SOURCE"] == "depth_chart_report"
    assert kc["AWAY_QB_PROJECTION_AS_OF"] == "2025-09-19T09:30:15Z"
    buf = result.loc["2025_03_BUF_MIA"]
    assert buf["AWAY_QB_ID_PROJECTED"] == "00-L"  # Questionable stays eligible
    assert pd.isna(buf["HOME_QB_ID_PROJECTED"])  # MIA has no chart


def test_later_weeks_are_not_projected() -> None:
    result = _projection(_snapshots(), _injuries())

    assert result.loc["2025_04_KC_BUF", list(PROJECTION_COLUMNS)].isna().all()


@pytest.mark.parametrize(
    "case",
    ["no_snapshots", "no_injuries", "old_chart_format", "report_for_another_week"],
)
def test_nothing_is_projected_without_both_inputs(case: str) -> None:
    snapshots: DataFrame | None = _snapshots()
    injuries: DataFrame | None = _injuries()
    if case == "no_snapshots":
        snapshots = None
    elif case == "no_injuries":
        injuries = None
    elif case == "old_chart_format":
        snapshots = DataFrame({"season": [2024], "club_code": ["KC"], "depth_position": ["QB"]})
    else:
        injuries = _injuries(week=2)

    result = _projection(snapshots, injuries)

    assert result[list(PROJECTION_COLUMNS)].isna().all().all()
    assert len(result) == 3


def _raw_historical() -> DataFrame:
    return DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "game_id": ["2025_01_KC_LAC", "2025_02_KC_BUF"],
            "away_team": ["KC", "KC"],
            "home_team": ["LAC", "BUF"],
            "result": [7.0, -3.0],
            "away_qb_id": ["00-A", "00-A"],
            "away_qb_name": ["A", "A"],
            "home_qb_id": ["00-C", "00-L"],
            "home_qb_name": ["C", "L"],
            "away_coach": ["c1", "c1"],
            "home_coach": ["c5", "c2"],
        }
    )


def _write_caches(repo: Path, *, snapshots: DataFrame, injuries: DataFrame) -> None:
    for name, frame in (("depth_charts", snapshots), ("injuries", injuries)):
        directory = repo / "data" / "raw" / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / f"{name}_2025.parquet", index=False)


def test_build_writes_projections_for_unplayed_games_only(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", _upcoming())
    _write_caches(tmp_path, snapshots=_snapshots(), injuries=_injuries())

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path).set_index("GAME_ID")

    assert sidecar.loc["2025_03_KC_DEN", "HOME_QB_ID_PROJECTED"] == "00-Y"
    played = sidecar.loc[["2025_01_KC_LAC", "2025_02_KC_BUF"], list(PROJECTION_COLUMNS)]
    assert played.isna().all().all()


def test_build_without_caches_leaves_every_projection_null(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", _upcoming())

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    assert sidecar[list(PROJECTION_COLUMNS)].isna().all().all()


def _history_sidecar(*, projected: bool) -> DataFrame:
    """Two played games, then three unplayed ones for the same team (KC away)."""
    rows = []
    for week, actual, projection in (
        (1, "00-A", None),
        (2, "00-A", None),
        (3, "00-B", None),
        (4, "00-B", "00-C" if projected else None),
        (5, None, None),
    ):
        rows.append(
            {
                "GAME_ID": f"2025_{week:02d}_KC_X{week}",
                "YEAR": "2025-2026",
                "WEEK_NUM": week,
                "AWAY_TEAM": "Kansas City Chiefs",
                "HOME_TEAM": f"Other {week}",
                "AWAY_QB_ID": actual,
                "HOME_QB_ID": f"00-H{week}",
                "AWAY_QB_ID_PROJECTED": projection,
                "HOME_QB_ID_PROJECTED": None,
            }
        )
    return DataFrame(rows).astype(
        {"AWAY_QB_ID_PROJECTED": "string", "HOME_QB_ID_PROJECTED": "string"}
    )


def _kc(history: DataFrame) -> DataFrame:
    return history[(history["TEAM"] == "Kansas City Chiefs")].set_index("WEEK_NUM")


def test_projected_id_replaces_the_resolved_id_for_its_own_game_only() -> None:
    history = _kc(resolve_starting_qb_history(_history_sidecar(projected=True)))

    assert history.loc[4, "RESOLVED_QB_ID"] == "00-C"  # projected (nflverse value was 00-B)
    assert history.loc[5, "RESOLVED_QB_ID"] == "00-B"  # later weeks anchor to the raw id
    assert list(history.loc[[1, 2, 3], "RESOLVED_QB_ID"]) == ["00-A", "00-A", "00-B"]


def test_change_flag_follows_the_resolved_ids() -> None:
    history = _kc(resolve_starting_qb_history(_history_sidecar(projected=True)))

    assert history.loc[4, "QB_CHANGE_FLAG"] == 1  # 00-B -> projected 00-C
    assert history.loc[5, "QB_CHANGE_FLAG"] == 1  # 00-C -> back to 00-B
    assert history.loc[3, "QB_CHANGE_FLAG"] == 1  # an ordinary played-game change


def test_without_projections_the_history_is_unchanged() -> None:
    with_columns = resolve_starting_qb_history(_history_sidecar(projected=False))
    without_columns = resolve_starting_qb_history(
        _history_sidecar(projected=False).drop(
            columns=["AWAY_QB_ID_PROJECTED", "HOME_QB_ID_PROJECTED"]
        )
    )

    pd.testing.assert_frame_equal(with_columns, without_columns)
    assert _kc(with_columns).loc[4, "RESOLVED_QB_ID"] == "00-B"


def test_played_games_never_change_with_a_projection_present() -> None:
    plain = _kc(resolve_starting_qb_history(_history_sidecar(projected=False)))
    projected = _kc(resolve_starting_qb_history(_history_sidecar(projected=True)))

    assert list(plain.loc[[1, 2, 3], "RESOLVED_QB_ID"]) == list(
        projected.loc[[1, 2, 3], "RESOLVED_QB_ID"]
    )
