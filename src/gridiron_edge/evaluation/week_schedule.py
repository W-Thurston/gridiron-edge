# src/gridiron_edge/evaluation/week_schedule.py

"""One week's regular-season schedule, read from the raw nflverse games file.

The raw file holds every scheduled game, played or not, keyed by the integer starting year of
the season (``2026`` for ``2026-2027``); a game has been played once ``result`` is set.
"""

from __future__ import annotations

from pandas import DataFrame


def scheduled_games(games: DataFrame, *, season_int: int, week: int) -> DataFrame:
    """Return the regular-season games of one week of the raw games file."""
    scoped = games.loc[(games["season"].astype(int) == season_int) & (games["week"] == week)]
    if "game_type" in scoped.columns:
        scoped = scoped.loc[scoped["game_type"] == "REG"]
    return scoped


def week_completion(games: DataFrame, *, season_int: int, week: int) -> tuple[int, int]:
    """Return ``(scheduled, played)`` regular-season games for one week of the raw games file."""
    scoped = scheduled_games(games, season_int=season_int, week=week)
    return len(scoped), int(scoped["result"].notna().sum())


def completed_game_ids(games: DataFrame, *, season_int: int, week: int) -> set[str]:
    """Return the regular-season games of one week that have a result in the raw games file."""
    scoped = scheduled_games(games, season_int=season_int, week=week)
    return set(scoped.loc[scoped["result"].notna(), "game_id"].astype(str))
