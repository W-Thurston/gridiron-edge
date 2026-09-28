# tests/unit/features/test_road_trips.py

"""Tests for canonical back-to-back road-game flags."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.road_trips import HomeAwayRoadTripsFeature


def _game(
    game_id: str,
    date: str,
    away: str,
    home: str,
    *,
    neutral: int = 0,
    year: str = "2025-2026",
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "GAME_DATE": date,
        "AWAY_TEAM": away,
        "HOME_TEAM": home,
        "IS_NEUTRAL_SITE": neutral,
    }


def _accessor(games: list[dict[str, object]], upcoming: pd.DataFrame | None = None) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = pd.DataFrame(games)
    if upcoming is None:
        datasets.schedule_upcoming_rich.side_effect = FileNotFoundError("none")
    else:
        datasets.schedule_upcoming_rich.return_value = upcoming
    return datasets


def _flags(games: list[dict[str, object]], game_id: str, **kwargs: object) -> tuple[object, object]:
    target = next(g for g in games if g["GAME_ID"] == game_id)
    frame = pd.DataFrame(
        {
            "GAME_ID": [game_id],
            "AWAY_TEAM": [target["AWAY_TEAM"]],
            "HOME_TEAM": [target["HOME_TEAM"]],
        }
    )
    row = HomeAwayRoadTripsFeature().compute(df=frame, datasets=_accessor(games, **kwargs)).iloc[0]
    return row["AWAY_BACK_TO_BACK_ROAD"], row["HOME_BACK_TO_BACK_ROAD"]


def test_registered_and_spec() -> None:
    assert FeatureRegistry.get("home_away_road_trips") is HomeAwayRoadTripsFeature
    assert HomeAwayRoadTripsFeature.spec.produces == [
        "AWAY_BACK_TO_BACK_ROAD",
        "HOME_BACK_TO_BACK_ROAD",
    ]


def test_road_after_road_is_flagged() -> None:
    games = [
        _game("g1", "2025-09-07", "A", "X"),
        _game("g2", "2025-09-14", "A", "Y"),
    ]

    assert _flags(games, "g2") == (1, 0)


def test_road_after_home_is_not_flagged() -> None:
    games = [
        _game("g1", "2025-09-07", "X", "A"),
        _game("g2", "2025-09-14", "A", "Y"),
    ]

    assert _flags(games, "g2")[0] == 0


def test_home_after_road_is_not_flagged() -> None:
    games = [
        _game("g1", "2025-09-07", "A", "X"),
        _game("g2", "2025-09-14", "Y", "A"),
    ]

    assert _flags(games, "g2")[1] == 0


def test_bye_week_between_road_games_resets() -> None:
    # 14 days apart: a bye week in between.
    games = [
        _game("g1", "2025-09-07", "A", "X"),
        _game("g2", "2025-09-21", "A", "Y"),
    ]

    assert _flags(games, "g2")[0] == 0


def test_twelve_day_gap_still_counts() -> None:
    games = [
        _game("g1", "2025-09-07", "A", "X"),
        _game("g2", "2025-09-19", "A", "Y"),
    ]

    assert _flags(games, "g2")[0] == 1


def test_first_game_of_season_is_zero_even_after_prior_season_road_game() -> None:
    games = [
        _game("old", "2025-01-05", "A", "X", year="2024-2025"),
        _game("g1", "2025-09-07", "A", "Y"),
    ]

    assert _flags(games, "g1")[0] == 0


def test_neutral_site_flags_both_sides_when_both_were_road() -> None:
    games = [
        _game("p1", "2025-09-07", "A", "X"),
        _game("p2", "2025-09-07", "H", "Y"),
        _game("g", "2025-09-14", "A", "H", neutral=1),
    ]

    assert _flags(games, "g") == (1, 1)


def test_previous_game_may_be_unplayed_but_scheduled() -> None:
    upcoming = pd.DataFrame(
        [
            {
                "game_id": "u1",
                "season": "2026-2027",
                "game_date": "2026-10-04",
                "away_team": "A",
                "home_team": "X",
                "neutral_site": False,
            },
            {
                "game_id": "u2",
                "season": "2026-2027",
                "game_date": "2026-10-11",
                "away_team": "A",
                "home_team": "Y",
                "neutral_site": False,
            },
        ]
    )
    frame = pd.DataFrame({"GAME_ID": ["u2"], "AWAY_TEAM": ["A"], "HOME_TEAM": ["Y"]})

    result = HomeAwayRoadTripsFeature().compute(df=frame, datasets=_accessor([], upcoming=upcoming))

    assert result.iloc[0]["AWAY_BACK_TO_BACK_ROAD"] == 1


def test_unscheduled_game_is_null() -> None:
    frame = pd.DataFrame({"GAME_ID": ["nope"], "AWAY_TEAM": ["A"], "HOME_TEAM": ["Y"]})

    result = HomeAwayRoadTripsFeature().compute(
        df=frame, datasets=_accessor([_game("g1", "2025-09-07", "A", "X")])
    )

    assert pd.isna(result.iloc[0]["AWAY_BACK_TO_BACK_ROAD"])


def test_preserves_row_order_and_extra_columns() -> None:
    games = [
        _game("g1", "2025-09-07", "A", "X"),
        _game("g2", "2025-09-14", "A", "Y"),
    ]
    frame = pd.DataFrame(
        {
            "GAME_ID": ["g2", "g1"],
            "AWAY_TEAM": ["A", "A"],
            "HOME_TEAM": ["Y", "X"],
            "MARK": [1, 2],
        }
    )

    result = HomeAwayRoadTripsFeature().compute(df=frame, datasets=_accessor(games))

    assert result["GAME_ID"].tolist() == ["g2", "g1"]
    assert result["MARK"].tolist() == [1, 2]
    assert result["AWAY_BACK_TO_BACK_ROAD"].tolist() == [1, 0]


def test_missing_input_columns_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        HomeAwayRoadTripsFeature().compute(
            df=pd.DataFrame({"GAME_ID": ["g"]}), datasets=_accessor([])
        )
