# tests/unit/evaluation/test_week_schedule.py

"""Tests for one week's regular-season schedule read from the raw games file."""

from __future__ import annotations

from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.week_schedule import (
    completed_game_ids,
    finished_weeks,
    scheduled_games,
    week_completion,
)

pytestmark = pytest.mark.unit


def _games(played: int, scheduled: int = 3, *, week: int = 4) -> DataFrame:
    rows = [
        {
            "season": 2026,
            "week": week,
            "game_type": "REG",
            "result": 3.0 if index < played else None,
        }
        for index in range(scheduled)
    ]
    rows.append({"season": 2026, "week": week + 1, "game_type": "REG", "result": None})
    rows.append({"season": 2026, "week": week, "game_type": "POST", "result": None})
    return DataFrame(rows)


def test_week_completion_counts_regular_season_games_of_the_week_only() -> None:
    assert week_completion(_games(3), season_int=2026, week=4) == (3, 3)
    assert week_completion(_games(2), season_int=2026, week=4) == (3, 2)
    assert week_completion(_games(0), season_int=2026, week=9) == (0, 0)


def test_completed_games_are_the_regular_season_games_with_a_result() -> None:
    games = DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [4, 4, 4, 5],
            "game_type": ["REG", "REG", "POST", "REG"],
            "game_id": ["a", "b", "c", "d"],
            "result": [3.0, None, 1.0, 2.0],
        }
    )

    assert completed_game_ids(games, season_int=2026, week=4) == {"a"}


def test_scheduled_games_keep_unplayed_games_and_drop_other_weeks_and_types() -> None:
    scoped = scheduled_games(_games(1), season_int=2026, week=4)

    assert len(scoped) == 3
    assert set(scoped["game_type"]) == {"REG"}
    assert int(scoped["result"].isna().sum()) == 2


def test_scheduled_games_accept_a_file_without_game_type() -> None:
    games = DataFrame({"season": [2026, 2026], "week": [4, 5], "result": [1.0, None]})

    assert len(scheduled_games(games, season_int=2026, week=4)) == 1


def test_finished_weeks_are_the_regular_season_weeks_with_every_result() -> None:
    games = DataFrame(
        {
            "season": [2026, 2026, 2026, 2026, 2026, 2025],
            "week": [1, 1, 2, 2, 3, 4],
            "game_type": ["REG", "REG", "REG", "POST", "REG", "REG"],
            "result": [3.0, -7.0, 1.0, None, None, 2.0],
        }
    )

    # Week 2's unplayed game is not regular season; week 3 is unplayed; week 4 is another season.
    assert finished_weeks(games, season_int=2026) == (1, 2)
