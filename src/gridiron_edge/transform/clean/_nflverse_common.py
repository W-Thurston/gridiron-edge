# src/gridiron_edge/transform/clean/_nflverse_common.py

"""Shared nflverse schema transformation utilities.

Used by both the games and schedule transform modules. Centralised here
so the helpers are public within the package and have a single owner.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# nflverse playoff week labels → integer week numbers.
# REG weeks are already integers; only postseason game_types need mapping.
GAME_TYPE_TO_WEEK: dict[str, int] = {
    "WC": 19,  # Wild Card
    "DIV": 20,  # Divisional
    "CON": 21,  # Conference Championship
    "SB": 22,  # Super Bowl
}

# nflverse short codes → long team names.
# This is the canonical short→long reference for the transform layer.
# The long names match the existing elo_state and modeling CSVs.
# Historical relocations map to the current franchise name.
NFLVERSE_SHORT_TO_LONG: dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    # Historical relocations
    "OAK": "Las Vegas Raiders",
    "SD": "Los Angeles Chargers",
    "STL": "Los Angeles Rams",
}


def season_label(season: int) -> str:
    """Convert integer season year to YYYY-YYYY+1 label.

    Args:
        season: The season start year (e.g. ``2025``).

    Returns:
        Season label string (e.g. ``"2025-2026"``).
    """
    return f"{season}-{season + 1}"


def gametime_to_hhmmss(gametime: str | float) -> str:
    """Normalise gametime to HH:MM:SS format.

    Args:
        gametime: Time string from nflverse (e.g. ``"20:20"``), or NaN.

    Returns:
        Time string in ``"HH:MM:SS"`` format, or ``""`` if missing or
        unrecognised.
    """
    if pd.isna(gametime) or not str(gametime).strip():
        return ""
    parts: list[str] = str(gametime).strip().split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    if len(parts) == 3:
        return str(gametime)
    return ""


def map_short_to_long(short: str) -> str:
    """Map a nflverse short team code to the canonical long team name.

    Args:
        short: nflverse short team code (e.g. ``"KC"``).

    Returns:
        Long team name (e.g. ``"Kansas City Chiefs"``), or the original
        short code if no mapping exists (logs a warning).
    """
    long_name: str | None = NFLVERSE_SHORT_TO_LONG.get(short)
    if long_name is None:
        logger.warning("No long-name mapping for nflverse short code: %s", short)
        return short
    return long_name
