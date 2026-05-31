# src/gridiron_edge/ingest/odds/_game_id.py
"""Resolve DraftKings events to canonical game_id format."""

from __future__ import annotations

from pandas import DataFrame, Series

from gridiron_edge.transform.clean._nflverse_common import NFLVERSE_SHORT_TO_LONG

# Relocation codes that should NOT win the reverse lookup
_HISTORICAL_CODES: set[str] = {"OAK", "SD", "STL"}

_LONG_TO_SHORT: dict[str, str] = {}
for short, long in NFLVERSE_SHORT_TO_LONG.items():
    if short in _HISTORICAL_CODES:
        # Only set if no current code has claimed this name yet
        _LONG_TO_SHORT.setdefault(long, short)
    else:
        # Current code always wins
        _LONG_TO_SHORT[long] = short


def team_long_to_short(long_name: str) -> str | None:
    """Convert a full team name to its short code.

    Args:
        long_name: Full team name (e.g. "Kansas City Chiefs").

    Returns:
        Short code (e.g. "KC") or None if not found.
    """
    return _LONG_TO_SHORT.get(long_name.strip())


def build_game_id(
    *,
    away_team: str,
    home_team: str,
    season_year: int,
    week: int,
) -> str | None:
    """Build canonical game_id from DK event metadata.

    Args:
        away_team: Full away team name from DK (e.g. "Kansas City Chiefs").
        home_team: Full home team name from DK (e.g. "Los Angeles Chargers").
        season_year: NFL season year (e.g. 2025 for the 2025-2026 season).
        week: NFL week number (1-22).

    Returns:
        Canonical game_id like "2025_01_KC_LAC", or None if team mapping fails.
    """
    away_short: str | None = team_long_to_short(away_team)
    home_short: str | None = team_long_to_short(home_team)
    if away_short is None or home_short is None:
        return None
    return f"{season_year}_{week:02d}_{away_short}_{home_short}"


def resolve_dk_game_ids(
    df_wide: DataFrame,
    *,
    season_year: int,
    week: int,
) -> DataFrame:
    """Add a ``game_id`` column to a DK DataFrame.

    Handles both the intermediate format (``home_team`` / ``away_team``)
    and the wide format (``team`` / ``opponent`` / ``location``).
    """
    df: DataFrame = df_wide.copy()

    if "home_team" in df.columns and "away_team" in df.columns:
        away_col, home_col = "away_team", "home_team"
    elif "team" in df.columns and "opponent" in df.columns and "location" in df.columns:
        # location=1 means "team" is home
        is_home: Series[bool] = df["location"] == 1
        away_col = "_away_resolved"
        home_col = "_home_resolved"
        df[home_col] = df["team"].where(is_home, df["opponent"])
        df[away_col] = df["opponent"].where(is_home, df["team"])
    else:
        return df

    away_short = df[away_col].map(team_long_to_short)
    home_short = df[home_col].map(team_long_to_short)
    prefix: str = f"{season_year}_{week:02d}_"
    df["game_id"] = away_short.astype(str).str.cat(home_short.astype(str), sep="_").radd(prefix)

    # Mark rows where either team didn't resolve
    unresolved: Series[bool] = away_short.isna() | home_short.isna()
    df.loc[unresolved, "game_id"] = None

    # Clean up temp columns
    for col in ("_away_resolved", "_home_resolved"):
        if col in df.columns:
            df = df.drop(columns=[col])

    return df
