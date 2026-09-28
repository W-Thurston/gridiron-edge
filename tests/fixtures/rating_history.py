# tests/fixtures/rating_history.py

"""Compact contiguous game history for recomputed-rating tests."""

from __future__ import annotations

import pandas as pd

type GameSpec = tuple[int, int, str, str, int, int]  # season, week, away, home, away pts, home pts


def make_rating_history(
    extra_games: list[GameSpec] | None = None,
    *,
    latest_season: int = 2026,
) -> pd.DataFrame:
    """Build history that passes complete-history validation.

    Each season before ``latest_season`` gets one filler Week 22 game between
    two teams that never appear in ``extra_games``, so the contiguous-history
    rule holds without touching the ratings under test.
    """
    rows: list[dict[str, object]] = []
    for season in range(1999, latest_season):
        rows.append(_row(season, 22, "Filler A", "Filler B", 24, 20, suffix="F"))
    for season, week, away, home, away_pts, home_pts in extra_games or []:
        rows.append(_row(season, week, away, home, away_pts, home_pts))
    if not any(row["YEAR"] == f"{latest_season}-{latest_season + 1}" for row in rows):
        rows.append(_row(latest_season, 1, "Filler A", "Filler B", 24, 20, suffix="F"))
    return pd.DataFrame(rows)


def make_qb_starters(
    games: pd.DataFrame, overrides: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Starter IDs per game: ``qb_<team>`` unless overridden by GAME_ID."""
    overrides = overrides or {}
    rows = []
    for game in games.itertuples(index=False):
        away_qb, home_qb = overrides.get(
            game.GAME_ID, (f"qb_{game.AWAY_TEAM}", f"qb_{game.HOME_TEAM}")
        )
        rows.append({"GAME_ID": game.GAME_ID, "AWAY_QB_ID": away_qb, "HOME_QB_ID": home_qb})
    return pd.DataFrame(rows)


def _row(
    season: int,
    week: int,
    away: str,
    home: str,
    away_pts: int,
    home_pts: int,
    *,
    suffix: str = "",
) -> dict[str, object]:
    return {
        "GAME_ID": f"{season}_{week:02d}_{away}_{home}{suffix}",
        "YEAR": f"{season}-{season + 1}",
        "WEEK_NUM": week,
        "AWAY_TEAM": away,
        "HOME_TEAM": home,
        "AWAY_SCORE": away_pts,
        "HOME_SCORE": home_pts,
    }
