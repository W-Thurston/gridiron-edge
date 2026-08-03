# tests/unit/api/test_serializers_projections.py

"""Unit tests for projections serializer."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.loaders import ProjectionGridData
from gridiron_edge.api.schemas.projections import (
    ProjectionsList,
    TeamProjectionRow,
)
from gridiron_edge.api.serializers.projections import (
    serialize_projection_grid,
    serialize_projections,
)

LONG_TO_SHORT = {
    "Seattle Seahawks": "SEA",
    "Buffalo Bills": "BUF",
    "Baltimore Ravens": "BAL",
}


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TEAM": "SEA",
                "AVG_WINS": 10.87025,
                "P_MAKE_PLAYOFFS": 0.7762,
                "P_REACH_DIV": 0.5497,
                "P_REACH_CONF": 0.3274,
                "P_REACH_SB": 0.1866,
                "P_WIN_SB": 0.1038,
                "elo_delta": 12.0,
            },
            {
                "TEAM": "BUF",
                "AVG_WINS": 10.3247,
                "P_MAKE_PLAYOFFS": 0.728,
                "P_REACH_DIV": 0.4856,
                "P_REACH_CONF": 0.2747,
                "P_REACH_SB": 0.1582,
                "P_WIN_SB": 0.0875,
                "elo_delta": -4.0,
            },
        ],
    )


class TestSerializeProjections:
    def test_empty_df_marks_items_unavailable(self) -> None:
        result = serialize_projections(
            pd.DataFrame(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        assert result.items == []
        assert result.total == 0
        assert result.response_meta is not None
        assert "items" in result.response_meta.field_status

    def test_populated(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            "2025-11-24T18:30:00Z",
            n_simulations=None,
        )
        assert result.total == 2
        assert result.season == "2025-2026"
        assert result.computed_at == "2025-11-24T18:30:00Z"
        # Sort by P_WIN_SB descending, so SEA first
        assert result.items[0].abbr == "SEA"
        assert result.items[0].name == "Seattle Seahawks"
        assert result.items[0].win_sb == 0.1038
        assert result.items[0].make_playoffs == 0.7762
        assert result.items[0].elo_delta == 12.0

    def test_marks_pending_status_fields(self) -> None:
        result: ProjectionsList = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        fs = result.response_meta.field_status
        assert "items.clinched" in fs
        assert "items.eliminated" in fs

    def test_unknown_abbr_falls_back_to_abbr_as_name(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "TEAM": "XXX",
                    "AVG_WINS": 5.0,
                    "P_MAKE_PLAYOFFS": 0.1,
                    "P_REACH_DIV": 0.0,
                    "P_REACH_CONF": 0.0,
                    "P_REACH_SB": 0.0,
                    "P_WIN_SB": 0.0,
                },
            ],
        )
        result = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        # Unknown abbreviation should default name to the abbr
        assert result.items[0].abbr == "XXX"
        assert result.items[0].name == "XXX"


class TestNSimulations:
    def test_populates_n_simulations(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )
        assert result.n_simulations == 10000

    def test_null_n_simulations_stays_null(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        assert result.n_simulations is None

    def test_no_longer_marks_n_simulations_pending(self) -> None:
        """The field_status marker on n_simulations should be gone
        now that we populate the field."""
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )
        fs = result.response_meta.field_status
        assert "n_simulations" not in fs

    def test_populated_elo_delta_has_no_unavailable_status(self) -> None:
        result: ProjectionsList = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        assert "items.elo_delta" not in result.response_meta.field_status

    def test_all_null_elo_deltas_mark_no_prior_snapshot(self) -> None:
        df: DataFrame = _make_df()
        df["elo_delta"] = None

        result = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        status = result.response_meta.field_status["items.elo_delta"]
        assert status.status == "blocked"
        assert status.blocker == "no_prior_snapshot"
        assert status.roadmap == "data"
        assert all(item.elo_delta is None for item in result.items)

    def test_partial_null_elo_deltas_do_not_mark_entire_field_unavailable(self) -> None:
        df: DataFrame = _make_df()
        df.loc[df["TEAM"] == "BUF", "elo_delta"] = None

        result: ProjectionsList = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        assert "items.elo_delta" not in result.response_meta.field_status
        by_team: dict[str, TeamProjectionRow] = {item.abbr: item for item in result.items}
        assert by_team["SEA"].elo_delta == 12.0
        assert by_team["BUF"].elo_delta is None


class TestSerializeProjectionGrid:
    def _probabilities(self) -> pd.DataFrame:
        rows = [
            {
                "TEAM": "SEA",
                "W01_WIN_P": 1.0,
                "W02_WIN_P": 0.64,
                "W03_WIN_P": 0.0,
                "W04_WIN_P": 0.0,
            },
            {
                "TEAM": "BUF",
                "W01_WIN_P": 0.0,
                "W02_WIN_P": 0.36,
                "W03_WIN_P": 0.0,
                "W04_WIN_P": 0.0,
            },
        ]

        for row in rows:
            for week in range(5, 19):
                row[f"W{week:02d}_WIN_P"] = 0.5

        return pd.DataFrame(rows)

    def _schedule(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "week": 1,
                    "game_date": "2026-09-13",
                    "game_time": "13:00:00",
                    "away_team": "Buffalo Bills",
                    "home_team": "Seattle Seahawks",
                    "season": "2026-2027",
                    "game_id": "2026_01_BUF_SEA",
                },
                {
                    "week": 2,
                    "game_date": "2026-09-20",
                    "game_time": "16:25:00",
                    "away_team": "Seattle Seahawks",
                    "home_team": "Buffalo Bills",
                    "season": "2026-2027",
                    "game_id": "2026_02_SEA_BUF",
                },
                {
                    "week": 4,
                    "game_date": "2026-10-04",
                    "game_time": "13:00:00",
                    "away_team": "Seattle Seahawks",
                    "home_team": "Buffalo Bills",
                    "season": "2026-2027",
                    "game_id": "2026_04_SEA_BUF",
                },
            ]
        )

    def _data(
        self,
        *,
        probabilities: pd.DataFrame | None = None,
        schedule: pd.DataFrame | None = None,
        games: pd.DataFrame | None = None,
        schedule_available: bool = True,
        completed_through_week: int = 1,
    ) -> ProjectionGridData:
        return ProjectionGridData(
            probabilities=(self._probabilities() if probabilities is None else probabilities),
            schedule=(self._schedule() if schedule is None else schedule),
            games=(
                pd.DataFrame(
                    [
                        {
                            "GAME_ID": "2026_01_BUF_SEA",
                            "AWAY_TEAM": "Buffalo Bills",
                            "HOME_TEAM": "Seattle Seahawks",
                            "AWAY_SCORE": 20,
                            "HOME_SCORE": 27,
                        }
                    ]
                )
                if games is None
                else games
            ),
            long_to_short={
                "Seattle Seahawks": "SEA",
                "Buffalo Bills": "BUF",
            },
            season="2026-2027",
            completed_through_week=completed_through_week,
            schedule_available=schedule_available,
        )

    def test_serializes_played_projected_and_bye_states(self) -> None:
        result = serialize_projection_grid(self._data())

        assert result.total == 2
        assert result.completed_through_week == 1

        by_team = {row.abbr: row for row in result.items}
        sea = by_team["SEA"]

        assert sea.weeks[0].state == "played"
        assert sea.weeks[0].actual_result == "W"
        assert sea.weeks[0].win_probability == 1.0
        assert sea.weeks[0].opponent == "BUF"
        assert sea.weeks[0].is_home is True

        assert sea.weeks[1].state == "projected"
        assert sea.weeks[1].win_probability == 0.64
        assert sea.weeks[1].opponent == "BUF"
        assert sea.weeks[1].is_home is False

        # No Week 3 schedule row confirms a bye, despite the artifact's 0.0.
        assert sea.weeks[2].state == "bye"
        assert sea.weeks[2].win_probability is None

    def test_serializes_played_loss(self) -> None:
        result = serialize_projection_grid(self._data())

        by_team = {row.abbr: row for row in result.items}
        buf = by_team["BUF"]

        assert buf.weeks[0].state == "played"
        assert buf.weeks[0].actual_result == "L"
        assert buf.weeks[0].win_probability == 0.0

    def test_serializes_played_tie_for_both_teams(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "GAME_ID": "2026_01_BUF_SEA",
                    "AWAY_TEAM": "Buffalo Bills",
                    "HOME_TEAM": "Seattle Seahawks",
                    "AWAY_SCORE": 21,
                    "HOME_SCORE": 21,
                }
            ]
        )

        result = serialize_projection_grid(self._data(games=games))

        assert {row.weeks[0].actual_result for row in result.items} == {"T"}

    def test_unplayed_game_is_not_marked_played(
        self,
    ) -> None:
        games = pd.DataFrame(
            [
                {
                    "GAME_ID": "2026_01_BUF_SEA",
                    "AWAY_TEAM": "Buffalo Bills",
                    "HOME_TEAM": "Seattle Seahawks",
                    "AWAY_SCORE": None,
                    "HOME_SCORE": None,
                }
            ]
        )

        result = serialize_projection_grid(self._data(games=games))

        by_team = {row.abbr: row for row in result.items}

        assert by_team["SEA"].weeks[0].state == ("projected")
        assert by_team["BUF"].weeks[0].state == ("projected")
        assert by_team["SEA"].weeks[0].actual_result is None
        assert by_team["BUF"].weeks[0].actual_result is None

    def test_unmapped_completed_game_is_not_played(
        self,
    ) -> None:
        games = pd.DataFrame(
            [
                {
                    "GAME_ID": "2026_01_BUF_SEA",
                    "AWAY_TEAM": "Unknown Away",
                    "HOME_TEAM": "Unknown Home",
                    "AWAY_SCORE": 20,
                    "HOME_SCORE": 27,
                }
            ]
        )

        result = serialize_projection_grid(self._data(games=games))

        assert {row.weeks[0].state for row in result.items} == {"projected"}

    def test_scheduled_week_without_probability_is_unavailable(self) -> None:
        probabilities = self._probabilities().drop(columns=["W04_WIN_P"])

        result = serialize_projection_grid(self._data(probabilities=probabilities))

        sea = next(row for row in result.items if row.abbr == "SEA")

        assert sea.weeks[3].state == "unavailable"
        assert sea.weeks[3].opponent == "BUF"
        assert sea.weeks[3].win_probability is None

    def test_missing_schedule_does_not_create_byes(self) -> None:
        result = serialize_projection_grid(
            self._data(
                schedule=pd.DataFrame(),
                schedule_available=False,
            )
        )

        assert all(week.state == "unavailable" for row in result.items for week in row.weeks)

        status = result.response_meta.field_status["items.weeks"]
        assert status.status == "blocked"
        assert status.blocker == "no_schedule_data"
        assert status.roadmap == "data"

    def test_empty_probabilities_marks_items_unavailable(self) -> None:
        result = serialize_projection_grid(self._data(probabilities=pd.DataFrame()))

        assert result.items == []
        assert result.total == 0

        status = result.response_meta.field_status["items"]
        assert status.status == "blocked"
        assert status.blocker == "no_projections_data"

    def test_unknown_team_name_falls_back_to_abbreviation(self) -> None:
        probabilities = self._probabilities().iloc[[0]].copy()
        probabilities.loc[:, "TEAM"] = "XXX"

        result = serialize_projection_grid(
            self._data(
                probabilities=probabilities,
                schedule=pd.DataFrame(),
                schedule_available=False,
            )
        )

        assert result.items[0].abbr == "XXX"
        assert result.items[0].name == "XXX"
