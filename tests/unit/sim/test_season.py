# tests/unit/sim/test_season.py

"""Tests for canonical season simulation input assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.sim._types import (
    AWAY_WIN,
    HOME_WIN,
    ROUND_WC,
    TIE,
    UNPLAYED,
    TeamIndex,
)
from gridiron_edge.sim.season import (
    build_schedule_arrays,
    extract_fixed_playoff_winners,
)


@pytest.fixture
def team_index() -> TeamIndex:
    return TeamIndex(
        short_names=[
            "KC",
            "LAC",
        ],
        short_to_id={
            "KC": 0,
            "LAC": 1,
        },
        long_to_short={
            "Kansas City Chiefs": "KC",
            "Los Angeles Chargers": "LAC",
        },
    )


def _regular_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [
                "2026-2027",
            ]
            * 4,
            "week": [
                1,
                2,
                3,
                4,
            ],
            "game_date": [
                "2026-09-10",
                "2026-09-17",
                "2026-09-24",
                "2026-10-01",
            ],
            "game_time": [
                "20:20:00",
            ]
            * 4,
            "game_id": [
                "2026_01_KC_LAC",
                "2026_02_KC_LAC",
                "2026_03_KC_LAC",
                "2026_04_KC_LAC",
            ],
            "away_team": [
                "Kansas City Chiefs",
            ]
            * 4,
            "home_team": [
                "Los Angeles Chargers",
            ]
            * 4,
        }
    )


def _regular_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "YEAR": ["2026-2027"] * 4,
            "WEEK_NUM": [1, 2, 3, 4],
            "GAME_ID": [
                "2026_01_KC_LAC",
                "2026_02_KC_LAC",
                "2026_03_KC_LAC",
                "2026_04_KC_LAC",
            ],
            "AWAY_SCORE": [
                27.0,
                17.0,
                21.0,
                float("nan"),
            ],
            "HOME_SCORE": [
                20.0,
                24.0,
                21.0,
                float("nan"),
            ],
        }
    )


class TestBuildScheduleArrays:
    """Encode canonical Away/Home game outcomes."""

    def test_encodes_all_result_states(
        self,
        team_index: TeamIndex,
    ) -> None:
        schedule, final_actual_week = build_schedule_arrays(
            _regular_schedule(),
            _regular_results(),
            team_index,
            "2026-2027",
        )

        assert schedule.result.tolist() == [
            int(AWAY_WIN),
            int(HOME_WIN),
            int(TIE),
            int(UNPLAYED),
        ]
        assert final_actual_week == 3

    def test_uses_canonical_scores_only(
        self,
        team_index: TeamIndex,
    ) -> None:
        results = _regular_results()

        assert "WINNER" not in results.columns
        assert "WIN_OR_TIE" not in results.columns
        assert "GAME_LOCATION" not in results.columns

        schedule, _ = build_schedule_arrays(
            _regular_schedule(),
            results,
            team_index,
            "2026-2027",
        )

        assert schedule.result[0] == AWAY_WIN
        assert schedule.result[1] == HOME_WIN

    def test_unplayed_later_week_does_not_advance_actual_week(
        self,
        team_index: TeamIndex,
    ) -> None:
        _, final_actual_week = build_schedule_arrays(
            _regular_schedule(),
            _regular_results(),
            team_index,
            "2026-2027",
        )

        assert final_actual_week == 3

    def test_no_completed_games_returns_week_zero(
        self,
        team_index: TeamIndex,
    ) -> None:
        results = _regular_results()
        results["AWAY_SCORE"] = float("nan")
        results["HOME_SCORE"] = float("nan")

        schedule, final_actual_week = build_schedule_arrays(
            _regular_schedule(),
            results,
            team_index,
            "2026-2027",
        )

        assert final_actual_week == 0
        assert np.all(schedule.result == UNPLAYED)

    @pytest.mark.parametrize(
        "missing_column",
        [
            "season",
            "week",
            "game_id",
            "away_team",
            "home_team",
        ],
    )
    def test_requires_rich_schedule_columns(
        self,
        team_index: TeamIndex,
        missing_column: str,
    ) -> None:
        schedule = _regular_schedule().drop(
            columns=[
                missing_column,
            ]
        )

        with pytest.raises(
            ValueError,
            match=missing_column,
        ):
            build_schedule_arrays(
                schedule,
                _regular_results(),
                team_index,
                "2026-2027",
            )

    def test_uses_explicit_rich_team_identity(
        self,
        team_index: TeamIndex,
    ) -> None:
        schedule = _regular_schedule()
        schedule["game_id"] = [
            "opaque-1",
            "opaque-2",
            "opaque-3",
            "opaque-4",
        ]

        results = _regular_results()
        results["GAME_ID"] = schedule["game_id"].tolist()

        arrays, _ = build_schedule_arrays(
            schedule,
            results,
            team_index,
            "2026-2027",
        )

        assert arrays.away.tolist() == [
            0,
            0,
            0,
            0,
        ]
        assert arrays.home.tolist() == [
            1,
            1,
            1,
            1,
        ]


def _playoff_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_19_KC_LAC",
                "2026_19_LAC_KC",
            ],
            "away_team": [
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            ],
            "home_team": [
                "Los Angeles Chargers",
                "Kansas City Chiefs",
            ],
        }
    )


def _playoff_results(
    *,
    first_away_score: float = 27.0,
    first_home_score: float = 20.0,
    second_away_score: float = 17.0,
    second_home_score: float = 24.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "YEAR": ["2026-2027"] * 2,
            "WEEK_NUM": [19, 19],
            "GAME_ID": [
                "2026_19_KC_LAC",
                "2026_19_LAC_KC",
            ],
            "AWAY_SCORE": [
                first_away_score,
                second_away_score,
            ],
            "HOME_SCORE": [
                first_home_score,
                second_home_score,
            ],
        }
    )


class TestExtractFixedPlayoffWinners:
    """Resolve playoff winners from canonical scores."""

    def test_records_away_and_home_winners(
        self,
        team_index: TeamIndex,
    ) -> None:
        fixed = extract_fixed_playoff_winners(
            df_wk_by_wk=_playoff_results(),
            df_schedule=_playoff_schedule(),
            team_index=team_index,
            season_year="2026-2027",
        )

        assert fixed[ROUND_WC, 0, 1] == 0

        home_only = extract_fixed_playoff_winners(
            df_wk_by_wk=_playoff_results(
                first_away_score=float("nan"),
                first_home_score=float("nan"),
            ),
            df_schedule=_playoff_schedule(),
            team_index=team_index,
            season_year="2026-2027",
        )

        assert home_only[ROUND_WC, 0, 1] == 0

    @pytest.mark.parametrize(
        (
            "away_score",
            "home_score",
        ),
        [
            (
                21.0,
                21.0,
            ),
            (
                float("nan"),
                float("nan"),
            ),
        ],
    )
    def test_tied_or_unplayed_game_remains_unset(
        self,
        team_index: TeamIndex,
        away_score: float,
        home_score: float,
    ) -> None:
        results = pd.DataFrame(
            {
                "YEAR": ["2026-2027"],
                "WEEK_NUM": [19],
                "GAME_ID": [
                    "2026_19_KC_LAC",
                ],
                "AWAY_SCORE": [away_score],
                "HOME_SCORE": [home_score],
            }
        )

        fixed = extract_fixed_playoff_winners(
            df_wk_by_wk=results,
            df_schedule=_playoff_schedule().iloc[[0]],
            team_index=team_index,
            season_year="2026-2027",
        )

        assert fixed[ROUND_WC, 0, 1] == -1

    def test_does_not_require_winner_field(
        self,
        team_index: TeamIndex,
    ) -> None:
        results = _playoff_results()

        assert "WINNER" not in results.columns

        fixed = extract_fixed_playoff_winners(
            df_wk_by_wk=results,
            df_schedule=_playoff_schedule(),
            team_index=team_index,
            season_year="2026-2027",
        )

        assert fixed[ROUND_WC, 0, 1] >= 0

    def test_records_away_winner(
        self,
        team_index: TeamIndex,
    ) -> None:
        fixed = extract_fixed_playoff_winners(
            df_wk_by_wk=_playoff_results().iloc[[0]],
            df_schedule=_playoff_schedule().iloc[[0]],
            team_index=team_index,
            season_year="2026-2027",
        )

        assert fixed[ROUND_WC, 0, 1] == 0
