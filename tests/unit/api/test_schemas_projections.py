# tests/unit/api/test_schemas_projections.py

"""Unit tests for projections schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.projections import (
    ProjectionsList,
    TeamProjectionRow,
)


class TestTeamProjectionRow:
    def test_minimum(self) -> None:
        row = TeamProjectionRow(abbr="SEA", name="Seattle Seahawks")
        assert row.abbr == "SEA"
        assert row.make_playoffs is None

    def test_populated(self) -> None:
        row = TeamProjectionRow(
            abbr="SEA",
            name="Seattle Seahawks",
            avg_wins=10.87,
            make_playoffs=0.7762,
            reach_div=0.5497,
            win_sb=0.1038,
        )
        assert row.avg_wins == 10.87
        assert row.win_sb == 0.1038

    def test_frozen(self) -> None:
        row = TeamProjectionRow(abbr="SEA", name="Seattle Seahawks")
        with pytest.raises(ValidationError):
            row.abbr = "BUF"

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            TeamProjectionRow(abbr="SEA", name="Seahawks", foo="bar")


class TestProjectionsList:
    def test_empty(self) -> None:
        pl = ProjectionsList()
        assert pl.items == []
        assert pl.season is None
        assert pl.computed_at is None

    def test_populated(self) -> None:
        pl = ProjectionsList(
            season="2025-2026",
            computed_at="2025-11-24T18:30:00Z",
            items=[
                TeamProjectionRow(
                    abbr="SEA",
                    name="Seattle Seahawks",
                    win_sb=0.1038,
                ),
            ],
            total=1,
        )
        assert pl.season == "2025-2026"
        assert pl.items[0].abbr == "SEA"
