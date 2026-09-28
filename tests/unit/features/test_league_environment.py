# tests/unit/features/test_league_environment.py

"""Tests for the season-to-date league scoring-environment feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.league_environment import (
    _PRIOR_SEASON_WEIGHT_GAMES,
    LeagueEnvironmentFeature,
)


def _games(rows: list[tuple[str, int, int | None, int | None]]) -> pd.DataFrame:
    """Rows are (season, week, away_score, home_score)."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": f"{season}-{week}-{index}",
                "YEAR": season,
                "WEEK_NUM": week,
                "AWAY_SCORE": away,
                "HOME_SCORE": home,
            }
            for index, (season, week, away, home) in enumerate(rows)
        ]
    )


def _accessor(games: pd.DataFrame) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games
    return datasets


def _target(season: str, week: int) -> pd.DataFrame:
    return pd.DataFrame({"GAME_ID": ["target"], "YEAR": [season], "WEEK_NUM": [week]})


def _level(games: pd.DataFrame, season: str, week: int) -> float:
    result = LeagueEnvironmentFeature().compute(df=_target(season, week), datasets=_accessor(games))
    return float(result["LEAGUE_PPG_TO_DATE"].iloc[0])


def test_registered_and_spec() -> None:
    assert FeatureRegistry.get("league_environment") is LeagueEnvironmentFeature
    assert LeagueEnvironmentFeature.spec.produces == ["LEAGUE_PPG_TO_DATE"]


def test_first_season_uses_only_to_date_mean() -> None:
    games = _games([("2000-2001", 1, 20, 30), ("2000-2001", 1, 10, 20), ("2000-2001", 2, 99, 99)])

    # Games before week 2: (25 + 15) / 2 points per team-game.
    assert _level(games, "2000-2001", 2) == pytest.approx(20.0)


def test_first_season_week_one_is_null() -> None:
    games = _games([("2000-2001", 1, 20, 30)])

    assert pd.isna(_level(games, "2000-2001", 1))


def test_week_one_takes_previous_season_mean() -> None:
    games = _games(
        [
            ("2000-2001", 1, 20, 30),  # 25 per team-game
            ("2000-2001", 2, 10, 20),  # 15 per team-game -> season mean 20
            ("2001-2002", 1, 40, 40),
        ]
    )

    assert _level(games, "2001-2002", 1) == pytest.approx(20.0)


def test_blends_previous_season_with_to_date_by_games() -> None:
    games = _games(
        [
            ("2000-2001", 1, 20, 20),  # previous-season mean 20
            ("2001-2002", 1, 30, 30),
            ("2001-2002", 1, 30, 30),
        ]
    )
    weight = _PRIOR_SEASON_WEIGHT_GAMES

    expected = (weight * 20.0 + 60.0) / (weight + 2.0)
    assert _level(games, "2001-2002", 2) == pytest.approx(expected)


def test_current_and_future_weeks_do_not_leak() -> None:
    base = _games([("2000-2001", 1, 20, 20), ("2001-2002", 1, 30, 30)])
    with_future = pd.concat(
        [base, _games([("2001-2002", 2, 99, 99), ("2001-2002", 3, 99, 99)])], ignore_index=True
    )

    assert _level(with_future, "2001-2002", 2) == pytest.approx(_level(base, "2001-2002", 2))


def test_playoffs_and_unplayed_games_are_excluded() -> None:
    games = _games(
        [
            ("2000-2001", 1, 20, 20),
            ("2000-2001", 19, 60, 60),  # playoff: never feeds the level
            ("2001-2002", 1, None, None),  # unplayed
        ]
    )

    assert _level(games, "2001-2002", 3) == pytest.approx(20.0)


def test_playoff_target_uses_regular_season_to_date() -> None:
    games = _games([("2000-2001", 1, 20, 20), ("2000-2001", 18, 30, 30)])

    assert _level(games, "2000-2001", 19) == pytest.approx(25.0)


def test_preserves_row_order_and_columns() -> None:
    games = _games([("2000-2001", 1, 20, 20)])
    frame = pd.DataFrame(
        {"GAME_ID": ["b", "a"], "YEAR": ["2000-2001"] * 2, "WEEK_NUM": [2, 2], "MARK": [1, 2]}
    )

    result = LeagueEnvironmentFeature().compute(df=frame, datasets=_accessor(games))

    assert result["GAME_ID"].tolist() == ["b", "a"]
    assert result["MARK"].tolist() == [1, 2]


def test_missing_frame_columns_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        LeagueEnvironmentFeature().compute(
            df=pd.DataFrame({"GAME_ID": ["x"]}), datasets=_accessor(_games([]))
        )
