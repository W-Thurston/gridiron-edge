# src/gridiron_edge/features/player/game_context.py
"""Game context features for player prop models.

Joins game-level data (spread, total, roof, date) from the cleaned games
dataset to player game logs and derives team-perspective features.

These features are all **known pre-game** (Vegas lines are set before
kickoff), so no shift(1) is needed — they are legitimate predictors at
prediction time.

Features produced:
- is_home         — binary: player's team is the home team
- game_spread     — spread from the player's team perspective
                    (negative = favored, positive = underdog)
- over_under      — total points line
- implied_team_total — (over_under - game_spread) / 2
- is_dome         — binary: game played under dome/closed roof
- rest_days       — calendar days since the team's previous game

Usage::

    from gridiron_edge.features.player.game_context import build_game_context_features

    df = build_game_context_features()
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team code normalization (historical → current)
# Duplicated from transform/clean/player_stats.py to avoid cross-module deps.
# ---------------------------------------------------------------------------

_TEAM_CODE_MAP: Final[dict[str, str]] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
}

# ---------------------------------------------------------------------------
# Full team name → abbreviation mapping
# Covers all 32 current franchises + historical names (1999-present).
# Abbreviations are era-appropriate; normalize via _TEAM_CODE_MAP afterward.
# ---------------------------------------------------------------------------

_FULL_NAME_TO_ABBREV: Final[dict[str, str]] = {
    # Current names (2024)
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Historical names
    "Oakland Raiders": "OAK",
    "San Diego Chargers": "SD",
    "St. Louis Rams": "STL",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
}

# ROOF values that indicate dome-like conditions.
_DOME_ROOFS: Final[set[str]] = {"dome", "closed"}


def _normalize_abbrev(s: Series) -> Series:
    """Normalize team abbreviations to current codes."""
    return s.map(lambda x: _TEAM_CODE_MAP.get(x, x))


def _load_games(repo: Path) -> DataFrame:
    """Load cleaned games data and select columns needed for context features."""
    games_path = repo / "data" / "cleaned" / "NFL_wk_by_wk_cleaned.csv"
    if not games_path.exists():
        msg = f"Cleaned games data not found: {games_path}"
        raise FileNotFoundError(msg)

    df: DataFrame = pd.read_csv(games_path)
    keep_cols: list[str] = [
        "GAME_ID",
        "VEGAS_LINE",
        "OVER_UNDER",
        "FAVORITED",
        "ROOF",
        "GAME_DATE",
    ]
    return df.loc[:, keep_cols].copy()


def _join_game_data(player_logs: DataFrame, games: DataFrame) -> DataFrame:
    """Join game-level data to player rows on game_id = GAME_ID.

    Uses a left join so player rows without matching games retain NaN
    for context features (rather than being dropped).
    """
    merged: DataFrame = player_logs.merge(
        games,
        left_on="game_id",
        right_on="GAME_ID",
        how="left",
    )
    n_unmatched: int = merged["GAME_ID"].isna().sum()
    if n_unmatched:
        logger.warning(
            "%d player rows did not match a game in the games dataset",
            n_unmatched,
        )
    return merged.drop(columns=["GAME_ID"])


def _derive_is_home(df: DataFrame) -> DataFrame:
    """Derive is_home from game_id format YYYY_WW_AWAY_HOME.

    The 4th segment of the game_id is the home team abbreviation.
    Normalizes the extracted code to current abbreviations before
    comparing with the player's (already-normalized) team column.
    """
    gid_parts: DataFrame = df["game_id"].str.split("_", expand=True)
    home_team: Series = _normalize_abbrev(gid_parts[3])
    df["is_home"] = df["team"] == home_team
    return df


def _derive_spread(df: DataFrame) -> DataFrame:
    """Compute game_spread from the player's team perspective.

    Uses abs(VEGAS_LINE) as the spread magnitude and FAVORITED to
    determine direction.  The favorite gets a negative spread (expected
    to win); the underdog gets a positive spread.

    When FAVORITED is NaN (pick'em), game_spread = 0.
    """
    # Map full team name → abbreviation → current code
    fav_abbrev: Series[str] = df["FAVORITED"].map(_FULL_NAME_TO_ABBREV)
    unmapped: Series[bool] = df["FAVORITED"].notna() & fav_abbrev.isna()
    if unmapped.any():
        bad_names = df.loc[unmapped, "FAVORITED"].unique()
        logger.warning("Unmapped team names in FAVORITED: %s", bad_names)

    fav_abbrev = _normalize_abbrev(fav_abbrev)

    spread_magnitude: Series = df["VEGAS_LINE"].abs()
    is_favorite: Series[bool] = df["team"] == fav_abbrev

    df["game_spread"] = np.where(
        df["VEGAS_LINE"].isna(),
        np.nan,
        np.where(is_favorite, -spread_magnitude, spread_magnitude),
    )
    # Pick'em: FAVORITED is NaN but VEGAS_LINE is 0
    pickem: Series[bool] = df["FAVORITED"].isna() & (df["VEGAS_LINE"] == 0)
    df.loc[pickem, "game_spread"] = 0.0

    return df


def _derive_implied_total(df: DataFrame) -> DataFrame:
    """Compute implied_team_total = (over_under - game_spread) / 2.

    A team favored by 3 in a game with a 47-point total has an implied
    total of (47 - (-3)) / 2 = 25.  The underdog's implied total is
    (47 - 3) / 2 = 22.  Together they sum to 47.
    """
    df["over_under"] = df["OVER_UNDER"]
    df["implied_team_total"] = (df["over_under"] - df["game_spread"]) / 2
    return df


def _derive_dome(df: DataFrame) -> DataFrame:
    """Derive is_dome from ROOF column.

    'dome' and 'closed' (retractable roof closed) are treated as dome
    conditions.  'outdoors' and 'open' (retractable roof open) are not.
    NaN defaults to False.
    """
    df["is_dome"] = df["ROOF"].str.lower().isin(_DOME_ROOFS).fillna(False)
    return df


def _derive_rest_days(df: DataFrame) -> DataFrame:
    """Compute calendar days since the team's previous game.

    Crosses season boundaries intentionally — a team's first game of a
    new season shows ~200 days rest (meaningful signal for the model).
    Week 1 of the earliest season in the data will be NaN.
    """
    df["_game_date"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["team", "_game_date"])

    df["_prev_game_date"] = df.groupby("team")["_game_date"].shift(1)
    # pyrefly: ignore [missing-attribute]
    df["rest_days"] = (df["_game_date"] - df["_prev_game_date"]).dt.days

    df = df.drop(columns=["_prev_game_date", "_game_date"])
    return df


def _drop_raw_game_columns(df: DataFrame) -> DataFrame:
    """Drop intermediate columns from the games join."""
    drop_cols: list[str] = [
        "VEGAS_LINE",
        "OVER_UNDER",
        "FAVORITED",
        "ROOF",
        "GAME_DATE",
    ]
    existing: list[str] = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def build_game_context_features(
    *,
    repo: Path | None = None,
) -> DataFrame:
    """Build game context features for player prop models.

    Loads player game logs and cleaned games data, joins them, and
    derives team-perspective context features.

    Args:
        repo: Repository root override.

    Returns:
        DataFrame with original player columns plus game context features:
        is_home, game_spread, over_under, implied_team_total, is_dome,
        rest_days.

    Raises:
        FileNotFoundError: If player game logs or games data not found.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    logs_path: Path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"

    if not logs_path.exists():
        msg: str = f"Cleaned player game logs not found: {logs_path}"
        raise FileNotFoundError(msg)

    df: DataFrame = pd.read_parquet(logs_path)
    logger.info("Loaded %d player game logs for game context features", len(df))

    games: DataFrame = _load_games(resolved_repo)
    logger.info("Loaded %d games for context join", len(games))

    df = _join_game_data(df, games)
    df = _derive_is_home(df)
    df = _derive_spread(df)
    df = _derive_implied_total(df)
    df = _derive_dome(df)
    df = _derive_rest_days(df)
    df = _drop_raw_game_columns(df)

    context_cols: list[str] = [
        "is_home",
        "game_spread",
        "over_under",
        "implied_team_total",
        "is_dome",
        "rest_days",
    ]
    logger.info("Built %d game context features for %d player-games", len(context_cols), len(df))
    for col in context_cols:
        nan_rate: float = df[col].isna().mean()
        logger.debug("  %s: %.1f%% NaN", col, nan_rate * 100)

    return df
