# tests/unit/api/test_schemas_teams.py

"""Unit tests for teams schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.teams import (
    RatingHistoryPoint,
    RecentResult,
    TeamProfile,
    TeamRankingRow,
    TeamRankingsList,
    TeamRecord,
)


class TestTeamRecord:
    def test_default(self) -> None:
        r = TeamRecord()
        assert r.wins == 0 and r.losses == 0 and r.ties == 0

    def test_populated(self) -> None:
        r = TeamRecord(wins=10, losses=2, ties=0)
        assert r.wins == 10


class TestTeamRankingRow:
    def test_minimum(self) -> None:
        row = TeamRankingRow(abbr="BAL", name="Baltimore Ravens")
        assert row.abbr == "BAL"
        assert row.rating is None

    def test_populated(self) -> None:
        row = TeamRankingRow(
            abbr="BAL",
            name="Baltimore Ravens",
            rating=1642.3,
            rank=1,
            record=TeamRecord(wins=10, losses=2),
        )
        assert row.rank == 1
        assert row.record.wins == 10


class TestTeamRankingsList:
    def test_empty(self) -> None:
        rl = TeamRankingsList()
        assert rl.items == []
        assert rl.season is None

    def test_populated(self) -> None:
        rl = TeamRankingsList(
            season="2025-2026",
            as_of_week=12,
            items=[
                TeamRankingRow(abbr="BAL", name="Baltimore Ravens", rating=1642.3, rank=1),
            ],
            total=1,
        )
        assert rl.season == "2025-2026"


class TestTeamProfile:
    def test_minimum(self) -> None:
        p = TeamProfile(abbr="BAL", name="Baltimore Ravens")
        assert p.rating is None
        assert p.rating_history is None

    def test_populated(self) -> None:
        p = TeamProfile(
            abbr="BAL",
            name="Baltimore Ravens",
            season="2025-2026",
            as_of_week=12,
            rating=1642.3,
            rank=1,
            record=TeamRecord(wins=10, losses=2),
            rating_history=[RatingHistoryPoint(week=1, rating=1600.0)],
            recent_results=[
                RecentResult(
                    week=12,
                    opponent="CLE",
                    is_home=True,
                    result="W",
                    score_for=31,
                    score_against=14,
                ),
            ],
        )
        assert p.rating == 1642.3
        assert p.recent_results[0].opponent == "CLE"


class TestRecentResult:
    def test_default(self) -> None:
        # Test that a fully null RecentResult can construct.
        # Requires only week; other fields default None.
        r = RecentResult(week=1)
        assert r.result is None

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            RecentResult(week=1, foo="bar")
