# tests/unit/ratings/test_elo_table.py

"""Tests for deterministic synthetic Elo Week 1 state."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ratings.elo.table import (
    EloTableConfig,
    _add_next_season_week_one,
    _latest_season_ratings_by_team,
    _max_week_for_year,
    _next_season_label,
    validate_complete_elo_history,
)


def _latest_season_games() -> DataFrame:
    """Create a historical season ending in postseason Week 22."""
    return pd.DataFrame(
        {
            "YEAR": [
                "2025-2026",
                "2025-2026",
            ],
            "WEEK_NUM": [
                21,
                22,
            ],
        }
    )


def _complete_history(
    seasons: list[int] | None = None,
) -> DataFrame:
    """Create compact contiguous completed history for validation tests."""
    represented: list[int] = seasons or list(range(1999, 2027))
    rows: list[dict[str, int | str]] = []
    for season in represented:
        rows.append(
            {
                "GAME_ID": f"{season}_01_A_B",
                "YEAR": f"{season}-{season + 1}",
                "WEEK_NUM": 1,
                "AWAY_TEAM": "Team A",
                "HOME_TEAM": "Team B",
                "AWAY_SCORE": 20,
                "HOME_SCORE": 24,
            }
        )

    return DataFrame(rows)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (
            "2024-2025",
            "2025-2026",
        ),
        (
            "2025-2026",
            "2026-2027",
        ),
        (
            "2099-2100",
            "2100-2101",
        ),
    ],
)
def test_next_season_is_derived_from_history(
    year: str,
    expected: str,
) -> None:
    assert _next_season_label(year) == expected


@pytest.mark.parametrize(
    "year",
    [
        "2026",
        "2026-27",
        "2026-2028",
        "not-a-season",
        "",
    ],
)
def test_rejects_invalid_historical_season_labels(
    year: str,
) -> None:
    with pytest.raises(ValueError):
        _next_season_label(year)


class TestCompleteEloHistoryValidation:
    """Tests for the canonical Elo reconstruction input contract."""

    def test_accepts_contiguous_history_with_partial_latest_season(
        self,
    ) -> None:
        games = _complete_history()

        validate_complete_elo_history(games)

    def test_rejects_empty_history(self) -> None:
        with pytest.raises(
            ValueError,
            match="Canonical Elo history must not be empty",
        ):
            validate_complete_elo_history(
                _complete_history().iloc[0:0],
            )

    def test_rejects_missing_required_columns(self) -> None:
        games = _complete_history().drop(
            columns=["HOME_SCORE"],
        )

        with pytest.raises(
            ValueError,
            match="missing required columns: HOME_SCORE",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_duplicate_game_ids(self) -> None:
        games = _complete_history()
        games.loc[1, "GAME_ID"] = games.loc[0, "GAME_ID"]

        with pytest.raises(
            ValueError,
            match="duplicate game IDs: 1999_01_A_B",
        ):
            validate_complete_elo_history(games)

    @pytest.mark.parametrize(
        ("column", "value", "message"),
        [
            (
                "GAME_ID",
                "",
                "null or empty game identities",
            ),
            (
                "YEAR",
                "",
                "null or empty season identities",
            ),
            (
                "AWAY_TEAM",
                "",
                "null or empty AWAY_TEAM identities",
            ),
            (
                "HOME_TEAM",
                None,
                "null or empty HOME_TEAM identities",
            ),
        ],
    )
    def test_rejects_invalid_identities(
        self,
        column: str,
        value: object,
        message: str,
    ) -> None:
        games = _complete_history()
        games.loc[0, column] = value

        with pytest.raises(
            ValueError,
            match=message,
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_same_team_on_both_sides(self) -> None:
        games = _complete_history()
        games.loc[0, "HOME_TEAM"] = games.loc[0, "AWAY_TEAM"]

        with pytest.raises(
            ValueError,
            match="identical Away and Home teams",
        ):
            validate_complete_elo_history(
                games,
            )

    @pytest.mark.parametrize(
        "label",
        [
            "2024",
            "2024-25",
            "2024-2026",
            "not-a-season",
        ],
    )
    def test_rejects_invalid_season_labels(
        self,
        label: str,
    ) -> None:
        games = _complete_history()
        games.loc[0, "YEAR"] = label

        with pytest.raises(
            ValueError,
            match="Invalid NFL season label",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_history_beginning_after_required_floor(
        self,
    ) -> None:
        games = _complete_history(
            seasons=list(range(2000, 2027)),
        )

        with pytest.raises(
            ValueError,
            match=("must begin with season 1999-2000; earliest represented season is 2000-2001"),
        ):
            validate_complete_elo_history(games)

    def test_rejects_missing_intermediate_season(
        self,
    ) -> None:
        games = _complete_history(
            seasons=[season for season in range(1999, 2027) if season != 2012],
        )

        with pytest.raises(
            ValueError,
            match=r"missing intermediate season\(s\): 2012-2013",
        ):
            validate_complete_elo_history(games)

    @pytest.mark.parametrize(
        "week",
        [
            None,
            0,
            1.5,
            "invalid",
        ],
    )
    def test_rejects_invalid_week_identity(
        self,
        week: object,
    ) -> None:
        games = _complete_history()
        games["WEEK_NUM"] = games["WEEK_NUM"].astype(object)
        games.loc[0, "WEEK_NUM"] = week

        with pytest.raises(
            ValueError,
            match="invalid week identities",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_one_sided_score_availability(
        self,
    ) -> None:
        games = _complete_history()
        games.loc[0, "HOME_SCORE"] = None

        with pytest.raises(
            ValueError,
            match="scores to be present together",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_unplayed_games(self) -> None:
        games = _complete_history()
        games.loc[0, ["AWAY_SCORE", "HOME_SCORE"]] = None

        with pytest.raises(
            ValueError,
            match="must contain completed games only",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_rejects_negative_scores(self) -> None:
        games = _complete_history()
        games.loc[0, "AWAY_SCORE"] = -1

        with pytest.raises(
            ValueError,
            match="negative game scores",
        ):
            validate_complete_elo_history(
                games,
            )

    def test_production_floor_rejects_current_season_only_history(
        self,
    ) -> None:
        games = _complete_history(
            seasons=[2026],
        )

        with pytest.raises(
            ValueError,
            match=("must begin with season 1999-2000; earliest represented season is 2026-2027"),
        ):
            validate_complete_elo_history(games)


def test_synthetic_week_one_uses_final_postgame_state() -> None:
    cfg = EloTableConfig(
        offseason_regress_frac=0.0,
    )
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            22,
        ): 1550.0,
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            22,
        ): 1450.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            }
        },
        cfg=cfg,
    )

    assert (
        elo[
            (
                "Kansas City Chiefs",
                "2026-2027",
                1,
            )
        ]
        == 1600.0
    )
    assert (
        elo[
            (
                "Los Angeles Chargers",
                "2026-2027",
                1,
            )
        ]
        == 1400.0
    )


def test_postseason_transition_uses_each_teams_latest_state() -> None:
    cfg = EloTableConfig(offseason_regress_frac=0.0)
    elo = {
        ("Eliminated Team", "2025-2026", 20): 1425.0,
        ("Eliminated Team", "2025-2026", 21): 1440.0,
        ("Conference Finalist", "2025-2026", 22): 1535.0,
        ("Super Bowl Team", "2025-2026", 22): 1600.0,
        ("Super Bowl Team", "2025-2026", 23): 1620.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=["2025-2026"],
        teams_by_year={
            "2025-2026": {
                "Eliminated Team",
                "Conference Finalist",
                "Super Bowl Team",
            }
        },
        cfg=cfg,
    )

    assert elo[("Eliminated Team", "2026-2027", 1)] == 1440.0
    assert elo[("Conference Finalist", "2026-2027", 1)] == 1535.0
    assert elo[("Super Bowl Team", "2026-2027", 1)] == 1620.0


def test_missing_returning_team_state_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"No prior-season Elo state found for returning team\(s\): Missing Team",
    ):
        _latest_season_ratings_by_team(
            {("Existing Team", "2025-2026", 22): 1510.0},
            year="2025-2026",
            teams={"Existing Team", "Missing Team"},
        )


def test_returning_teams_receive_offseason_regression() -> None:
    cfg = EloTableConfig(
        offseason_regress_frac=1 / 3.0,
    )
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            }
        },
        cfg=cfg,
    )

    assert elo[
        (
            "Kansas City Chiefs",
            "2026-2027",
            1,
        )
    ] == pytest.approx(1566.6666666667)
    assert elo[
        (
            "Los Angeles Chargers",
            "2026-2027",
            1,
        )
    ] == pytest.approx(1433.3333333333)


def test_synthetic_transition_is_reproducible() -> None:
    cfg = EloTableConfig()
    games = _latest_season_games()
    teams_by_year = {
        "2025-2026": {
            "Kansas City Chiefs",
            "Los Angeles Chargers",
        }
    }
    original = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    first = original.copy()
    second = original.copy()

    _add_next_season_week_one(
        first,
        games=games,
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year=teams_by_year,
        cfg=cfg,
    )
    _add_next_season_week_one(
        second,
        games=games,
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year=teams_by_year,
        cfg=cfg,
    )

    assert first == second


def test_only_next_season_week_one_is_created() -> None:
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
            }
        },
        cfg=EloTableConfig(),
    )

    future_keys = [key for key in elo if key[1] == "2026-2027"]

    assert future_keys == [
        (
            "Kansas City Chiefs",
            "2026-2027",
            1,
        )
    ]


def test_historical_rows_are_not_modified() -> None:
    historical_key = (
        "Kansas City Chiefs",
        "2025-2026",
        23,
    )
    elo = {
        historical_key: 1600.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
            }
        },
        cfg=EloTableConfig(),
    )

    assert elo[historical_key] == 1600.0


def test_empty_history_creates_no_synthetic_state() -> None:
    elo: dict[tuple[str, str, int], float] = {}

    _add_next_season_week_one(
        elo,
        games=DataFrame(
            columns=[
                "YEAR",
                "WEEK_NUM",
            ]
        ),
        sorted_years=[],
        teams_by_year={},
        cfg=EloTableConfig(),
    )

    assert elo == {}


def test_incomplete_latest_season_creates_no_synthetic_state() -> None:
    elo = {
        (
            "Kansas City Chiefs",
            "2026-2027",
            1,
        ): 1500.0,
        (
            "Kansas City Chiefs",
            "2026-2027",
            2,
        ): 1520.0,
    }
    original = elo.copy()

    _add_next_season_week_one(
        elo,
        games=DataFrame(
            {
                "YEAR": [
                    "2026-2027",
                ],
                "WEEK_NUM": [
                    1,
                ],
            }
        ),
        sorted_years=[
            "2026-2027",
        ],
        teams_by_year={
            "2026-2027": {
                "Kansas City Chiefs",
            },
        },
        cfg=EloTableConfig(),
    )

    assert elo == original
    assert not any(year == "2027-2028" for _, year, _ in elo)


def test_max_week_for_year_is_season_scoped() -> None:
    games = DataFrame(
        {
            "YEAR": [
                "2025-2026",
                "2025-2026",
                "2026-2027",
            ],
            "WEEK_NUM": [
                21,
                22,
                1,
            ],
        }
    )

    assert (
        _max_week_for_year(
            games,
            "2025-2026",
        )
        == 22
    )
    assert (
        _max_week_for_year(
            games,
            "2026-2027",
        )
        == 1
    )
    assert (
        _max_week_for_year(
            games,
            "2027-2028",
        )
        == 0
    )
