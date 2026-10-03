# tests/unit/features/test_projected_starter.py

"""Tests for the pure projected-starting-QB resolver."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.team.projected_starter import (
    UNAVAILABLE_PRIMARY,
    UNAVAILABLE_VARIANT,
    latest_snapshot_chart,
    ordered_candidates,
    prepare_snapshots,
    project_starter,
)

pytestmark = pytest.mark.unit


def _starter(chart: list[tuple[int, str]], statuses: dict[str, str | None]) -> str | None:
    return project_starter(ordered_candidates(chart, statuses, UNAVAILABLE_PRIMARY))


def test_takes_the_top_chart_quarterback_when_everyone_is_available() -> None:
    assert _starter([(2, "B"), (1, "A"), (3, "C")], {}) == "A"


def test_skips_out_and_doubtful_but_not_questionable() -> None:
    chart = [(1, "A"), (2, "B"), (3, "C")]

    assert _starter(chart, {"A": "Out"}) == "B"
    assert _starter(chart, {"A": "Doubtful", "B": "Out"}) == "C"
    assert _starter(chart, {"A": "Questionable"}) == "A"
    skipped = ordered_candidates(chart, {"A": "Questionable"}, UNAVAILABLE_VARIANT)
    assert project_starter(skipped) == "B"


def test_names_the_top_of_the_chart_when_everyone_is_unavailable() -> None:
    assert _starter([(1, "A"), (2, "B")], {"A": "Out", "B": "Out"}) == "A"


def test_empty_chart_has_no_starter() -> None:
    assert _starter([], {}) is None


def test_candidates_record_the_inputs_and_drop_repeated_players() -> None:
    candidates = ordered_candidates(
        [(1, "A"), (1, "A"), (2, "B")], {"A": "Out"}, UNAVAILABLE_PRIMARY
    )

    assert [(c.gsis_id, c.chart_rank, c.status, c.available) for c in candidates] == [
        ("A", 1, "Out", False),
        ("B", 2, None, True),
    ]


def test_tied_rank_one_keeps_input_order() -> None:
    assert _starter([(1, "A"), (1, "B")], {}) == "A"


def _snapshots() -> DataFrame:
    def rows(when: str, team: str, ids: list[str]) -> list[dict[str, object]]:
        return [
            {"dt": when, "team": team, "pos_abb": "QB", "pos_rank": rank, "gsis_id": player}
            for rank, player in enumerate(ids, start=1)
        ]

    return DataFrame(
        rows("2025-09-01T08:00:00Z", "KC", ["A", "B"])
        + rows("2025-09-04T08:00:00Z", "KC", ["B", "A", "C"])
        + rows("2025-09-04T08:00:00Z", "DEN", ["X"])
        + [
            {
                "dt": "2025-09-04T08:00:00Z",
                "team": "KC",
                "pos_abb": "RB",
                "pos_rank": 1,
                "gsis_id": "R",
            }
        ]
    )


def test_snapshots_are_indexed_by_team_and_keep_only_quarterbacks() -> None:
    by_team = prepare_snapshots(_snapshots())

    assert sorted(by_team) == ["DEN", "KC"]
    assert set(by_team["KC"]["gsis_id"]) == {"A", "B", "C"}


def test_latest_chart_returns_the_newest_snapshot_and_its_time() -> None:
    chart, as_of = latest_snapshot_chart(prepare_snapshots(_snapshots())["KC"])

    assert chart == [(1, "B"), (2, "A"), (3, "C")]
    assert as_of == pd.Timestamp("2025-09-04T08:00:00Z")


def test_latest_chart_respects_the_cutoff() -> None:
    team = prepare_snapshots(_snapshots())["KC"]

    chart, as_of = latest_snapshot_chart(team, cutoff=pd.Timestamp("2025-09-02T00:00:00Z"))
    assert chart == [(1, "A"), (2, "B")]
    assert as_of == pd.Timestamp("2025-09-01T08:00:00Z")

    assert latest_snapshot_chart(team, cutoff=pd.Timestamp("2025-08-01T00:00:00Z")) == ([], None)
