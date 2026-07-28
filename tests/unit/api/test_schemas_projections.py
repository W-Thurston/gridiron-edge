# tests/unit/api/test_schemas_projections.py

"""Unit tests for projections schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.projections import (
    ProjectionGridResponse,
    ProjectionGridTeam,
    ProjectionGridWeek,
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


class TestProjectionGridWeek:
    def test_projected_game(self) -> None:
        week = ProjectionGridWeek(
            week=1,
            state="projected",
            opponent="NE",
            is_home=True,
            game_id="2026_01_NE_SEA",
            game_date="2026-09-09",
            game_time="20:20:00",
            win_probability=0.64,
        )

        assert week.week == 1
        assert week.state == "projected"
        assert week.opponent == "NE"
        assert week.win_probability == 0.64
        assert week.actual_result is None

    def test_played_game(self) -> None:
        week = ProjectionGridWeek(
            week=5,
            state="played",
            opponent="SF",
            is_home=False,
            game_id="2026_05_SEA_SF",
            game_date="2026-10-11",
            game_time="16:25:00",
            win_probability=1.0,
            actual_result="W",
        )

        assert week.state == "played"
        assert week.actual_result == "W"

    def test_bye(self) -> None:
        week = ProjectionGridWeek(
            week=7,
            state="bye",
        )

        assert week.opponent is None
        assert week.win_probability is None

    def test_tie_result(self) -> None:
        week = ProjectionGridWeek(
            week=8,
            state="played",
            opponent="LAR",
            actual_result="T",
            win_probability=0.0,
        )

        assert week.actual_result == "T"

    def test_rejects_invalid_state(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionGridWeek(
                week=1,
                state="unknown",
            )

    def test_rejects_invalid_week(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionGridWeek(
                week=19,
                state="projected",
            )

    def test_rejects_out_of_range_probability(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionGridWeek(
                week=1,
                state="projected",
                win_probability=1.01,
            )

    def test_frozen(self) -> None:
        week = ProjectionGridWeek(
            week=1,
            state="projected",
        )

        with pytest.raises(ValidationError):
            week.state = "bye"


class TestProjectionGridTeam:
    def test_populated(self) -> None:
        team = ProjectionGridTeam(
            abbr="SEA",
            name="Seattle Seahawks",
            weeks=[
                ProjectionGridWeek(
                    week=1,
                    state="projected",
                    opponent="NE",
                    win_probability=0.64,
                ),
                ProjectionGridWeek(
                    week=2,
                    state="bye",
                ),
            ],
        )

        assert team.abbr == "SEA"
        assert len(team.weeks) == 2
        assert team.weeks[1].state == "bye"

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionGridTeam(
                abbr="SEA",
                name="Seattle Seahawks",
                weeks=[],
                foo="bar",
            )


class TestProjectionGridResponse:
    def test_defaults(self) -> None:
        response = ProjectionGridResponse()

        assert response.items == []
        assert response.completed_through_week == 0
        assert response.regular_season_weeks == 18

    def test_populated(self) -> None:
        response = ProjectionGridResponse(
            season="2026-2027",
            completed_through_week=0,
            regular_season_weeks=18,
            items=[
                ProjectionGridTeam(
                    abbr="SEA",
                    name="Seattle Seahawks",
                    weeks=[
                        ProjectionGridWeek(
                            week=1,
                            state="projected",
                            opponent="NE",
                            is_home=True,
                            win_probability=0.64,
                        ),
                    ],
                ),
            ],
            total=1,
        )

        assert response.season == "2026-2027"
        assert response.total == 1
        assert response.items[0].weeks[0].opponent == "NE"

    def test_rejects_invalid_completed_week(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionGridResponse(
                completed_through_week=19,
            )
