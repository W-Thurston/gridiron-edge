# src/gridiron_edge/ratings/history.py

"""Shared chronology helpers for ratings recomputed from complete game history.

Path-dependent ratings that are recomputed rather than persisted (``DECISIONS.md``
D69) depend on every game since 1999, so a truncated games file would silently
change every value. ``prepare_history`` therefore reuses the canonical
complete-history validation Elo enforces and fails closed.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.ratings.elo.table import validate_complete_elo_history

#: A chronological position: ``(season start year, week number)``.
type CutoffKey = tuple[int, int]

HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
)


def season_start(label: object) -> int:
    """Return the starting year of a canonical ``YYYY-YYYY`` season label."""
    text = str(label).strip()
    parts = text.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid NFL season label {text!r}. Expected format YYYY-YYYY.")
    start, end = int(parts[0]), int(parts[1])
    if end != start + 1:
        raise ValueError(
            f"Invalid NFL season label {text!r}. Ending year must be one greater than start."
        )
    return start


def season_label(start: int) -> str:
    """Return the canonical season label for a starting year."""
    return f"{start}-{start + 1}"


def prepare_history(games: DataFrame) -> DataFrame:
    """Validate complete history and return it in chronological order.

    Adds integer ``SEASON`` and ``WEEK`` columns, sorted by season, week, then
    ``GAME_ID`` so replays are deterministic.

    Raises:
        ValueError: If history is empty, malformed, duplicated, late-starting,
            or discontinuous (see ``validate_complete_elo_history``).
    """
    validate_complete_elo_history(games)
    history = games.loc[:, list(HISTORY_COLUMNS)].copy()
    history["SEASON"] = history["YEAR"].map(season_start).astype(int)
    history["WEEK"] = pd.to_numeric(history["WEEK_NUM"]).astype(int)
    history["AWAY_SCORE"] = pd.to_numeric(history["AWAY_SCORE"]).astype(float)
    history["HOME_SCORE"] = pd.to_numeric(history["HOME_SCORE"]).astype(float)
    history["GAME_ID"] = history["GAME_ID"].astype(str)
    return history.sort_values(["SEASON", "WEEK", "GAME_ID"], kind="stable", ignore_index=True)


def away_outcome(away_score: float, home_score: float) -> float:
    """Return the Away team's result: 1 win, 0.5 tie, 0 loss."""
    if away_score > home_score:
        return 1.0
    if away_score == home_score:
        return 0.5
    return 0.0
