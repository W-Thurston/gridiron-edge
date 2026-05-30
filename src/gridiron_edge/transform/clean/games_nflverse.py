# src/gridiron_edge/transform/clean/games_nflverse.py

"""Transform nflverse raw schedule data into the canonical games schema.

Maps the nflverse home/away-oriented schema to the WINNER/LOSER-oriented
canonical games schema used by Elo, features, and all downstream modules.

nflverse schema (key columns used):
    game_id         str     "2025_01_PHI_GB"  (YYYY_WW_AWAY_HOME, short codes)
    season          int     2025
    game_type       str     "REG" | "WC" | "DIV" | "CON" | "SB"
    week            int     1-22
    gameday         str     "2025-09-04"
    weekday         str     "Thursday"
    gametime        str     "20:20"  (Eastern, 24h)
    away_team       str     "PHI"  (nflverse short code)
    home_team       str     "GB"   (nflverse short code)
    away_score      int     20     (NaN if unplayed)
    home_score      int     17     (NaN if unplayed)
    location        str     "Home" | "Neutral"
    result          float   home_score - away_score (NaN if unplayed)
    stadium         str     "Lambeau Field"
    roof            str     "outdoors" | "open" | "closed" | "dome"
    surface         str     "grass" | "fieldturf" | ...
    spread_line     float   spread from home team perspective (+home favored)
    total_line      float   over/under
    div_game        int     1 if division game, 0 otherwise
    overtime        int     1 if OT, 0 otherwise

Canonical games schema (NFL_wk_by_wk_cleaned.csv):
    GAME_ID             str     "2025_01_PHI_GB"
    WEEK_NUM            int     1-22
    GAME_DAY_OF_WEEK    str     "Thursday"
    GAME_DATE           str     "2025-09-04"
    GAMETIME            str     "20:20:00"  (HH:MM:SS for backwards compat)
    WINNER              str     "Philadelphia Eagles"  (long name)
    GAME_LOCATION       str     "H" (home) | "@" (away) | "N" (neutral)
    LOSER               str     "Green Bay Packers"    (long name)
    BOXSCORE_LINK       str     ""  (deprecated — set to empty string)
    PTS_WINNER          int     20
    PTS_LOSER           int     17
    YARDS_WINNER        int     0   (not available from schedules — set to 0)
    TURNOVERS_WINNER    int     0   (not available from schedules — set to 0)
    YARDS_LOSER         int     0   (not available from schedules — set to 0)
    TURNOVERS_LOSER     int     0   (not available from schedules — set to 0)
    YEAR                str     "2025-2026"
    STADIUM             str     "Lambeau Field"
    ROOF                str     "outdoors"
    SURFACE             str     "grass"
    VEGAS_LINE          float   -3.0  (spread from winner perspective)
    OVER_UNDER          float   47.5
    FAVORITED           str     "Green Bay Packers"  (long name of favored team)
    WIN_OR_TIE          float   1.0 | 0.0 | 0.5
    DIV_GAME            int     1 if both teams are in the same division, else 0
    GAME_ID             str     (same as above, alias confirmation)

Notes:
    - YARDS/TURNOVERS are set to 0 because they are not in nflverse schedules.
      They were stored in PFR data but were never consumed by any downstream
      module (Elo, features, sim). Set to 0 to preserve schema compatibility.
    - GAMETIME is stored as HH:MM:SS to match the legacy PFR format.
    - YEAR uses the "YYYY-YYYY+1" season label format (e.g. "2025-2026").
    - GAME_LOCATION uses: "H" = home game, "@" = away game, "N" = neutral site.
    - nflverse uses short team codes. This module maps them to long names
      using the teams_long_short reference dataset so all downstream code
      (Elo table, features) continues to work unchanged.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    gametime_to_hhmmss,
    map_short_to_long,
    season_label,
)

logger: Logger = logging.getLogger(__name__)


def _game_location(location: str | float) -> str:
    """Map nflverse location field to PFR GAME_LOCATION convention.

    Args:
        location: nflverse location string (``"Home"`` or ``"Neutral"``).

    Returns:
        ``"H"`` for a standard home game, ``"N"`` for neutral site.
        Away games are represented by the winner/loser orientation in the
        canonical schema (GAME_LOCATION = ``"@"`` when the winner is the
        away team) and are handled separately in the winner derivation logic.
    """
    if pd.isna(location):
        return "H"
    loc: str = str(location).strip()
    if loc == "Neutral":
        return "N"
    return "H"


def clean_nflverse_games(
    *,
    repo: Path | None = None,
) -> Path:
    """Transform nflverse raw games CSV into the canonical cleaned games CSV.

    Reads ``data/raw/NFL_wk_by_wk_nflverse.csv``, filters to completed regular
    season and postseason games, maps to the canonical WINNER/LOSER schema,
    and writes to ``data/cleaned/NFL_wk_by_wk_cleaned.csv``.

    Unplayed games (``result = NaN``) are excluded — they belong in the
    upcoming schedule, not the historical games dataset.

    Args:
        repo: Absolute path to the repository root. Defaults to the value
            from ``get_settings()``.

    Returns:
        Absolute path to the written canonical games CSV.
    """
    settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    raw_path: Path = dataset_path(resolved_repo, "games_raw_nflverse")
    if not raw_path.exists():
        msg: str = (
            f"Raw nflverse games file not found: {raw_path}. "
            "Run `gridiron ingest nflverse-games` first."
        )
        raise FileNotFoundError(msg)

    logger.info("Reading raw nflverse games from %s", raw_path)
    df: DataFrame = pd.read_parquet(raw_path)

    # --- Filter to completed games only ---
    df = df.loc[df["result"].notna(), :].copy()

    # --- Filter out preseason ---
    df = df.loc[df["game_type"] != "PRE"].copy()

    logger.info("Processing %d completed games", len(df))

    if df.empty:
        logger.info("No completed games found in raw file — season may not have started.")
        out_path: Path = dataset_path(resolved_repo, "games")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(
            columns=[
                "GAME_ID",
                "WEEK_NUM",
                "GAME_DAY_OF_WEEK",
                "GAME_DATE",
                "GAMETIME",
                "WINNER",
                "GAME_LOCATION",
                "LOSER",
                "BOXSCORE_LINK",
                "PTS_WINNER",
                "PTS_LOSER",
                "YARDS_WINNER",
                "TURNOVERS_WINNER",
                "YARDS_LOSER",
                "TURNOVERS_LOSER",
                "YEAR",
                "STADIUM",
                "ROOF",
                "SURFACE",
                "VEGAS_LINE",
                "OVER_UNDER",
                "FAVORITED",
                "WIN_OR_TIE",
                "DIV_GAME",
            ]
        )
        empty.to_csv(out_path, index=False)
        return out_path

    # --- Normalise week numbers ---
    # REG games have integer weeks; postseason game_types map to 19-22.
    def _resolve_week(row: pd.Series) -> int:
        gt = str(row["game_type"])
        if gt in GAME_TYPE_TO_WEEK:
            return GAME_TYPE_TO_WEEK[gt]
        return int(row["week"])

    df["WEEK_NUM"] = df.apply(_resolve_week, axis=1)

    # --- Derive WINNER / LOSER from scores ---
    # home_score > away_score → home team won → WINNER = home, LOSER = away
    # away_score > home_score → away team won → WINNER = away, LOSER = home
    # equal scores → tie (use home as WINNER by convention, WIN_OR_TIE = 0.5)
    home_wins = df["home_score"] > df["away_score"]
    tie = df["home_score"] == df["away_score"]

    df["WINNER_SHORT"] = pd.Series(
        np.where(home_wins | tie, df["home_team"], df["away_team"]),
        index=df.index,
    )

    df["LOSER_SHORT"] = pd.Series(
        np.where(home_wins | tie, df["away_team"], df["home_team"]),
        index=df.index,
    )
    df["PTS_WINNER"] = np.where(
        home_wins | tie,
        df["home_score"].astype(int),
        df["away_score"].astype(int),
    )
    df["PTS_LOSER"] = np.where(
        home_wins | tie,
        df["away_score"].astype(int),
        df["home_score"].astype(int),
    )

    # WIN_OR_TIE: 1 = win, 0.5 = tie, 0 = loss (from winner perspective)
    df["WIN_OR_TIE"] = np.where(tie, 0.5, 1.0)

    # --- GAME_LOCATION from home/away winner perspective ---
    # "H" = home game, "@" = away game (winner was visitor), "N" = neutral site
    neutral_mask = df["location"].fillna("").str.strip() == "Neutral"
    away_won_mask = df["away_score"] > df["home_score"]

    df["GAME_LOCATION"] = np.where(
        neutral_mask,
        "N",
        np.where(away_won_mask, "@", "H"),
    )

    # --- Map short codes to long names ---
    df["WINNER"] = df["WINNER_SHORT"].astype(str).map(map_short_to_long)
    df["LOSER"] = df["LOSER_SHORT"].astype(str).map(map_short_to_long)

    # --- YEAR label ---
    df["YEAR"] = df["season"].astype(int).map(season_label)

    # --- GAMETIME ---
    df["GAMETIME"] = df["gametime"].apply(gametime_to_hhmmss)

    # --- GAME_ID: use nflverse alt_game_id if available, else game_id ---
    # nflverse game_id format: "2025_01_PHI_GB" — already matches our convention
    df["GAME_ID"] = df["game_id"].astype(str)

    # --- VEGAS_LINE / FAVORITED ---
    # nflverse spread_line is from HOME team perspective (positive = home favored).
    # Convert to: value = margin home is favored by, FAVORITED = favored long name.
    # Negative spread_line means away team is favored.
    df["VEGAS_LINE"] = pd.to_numeric(df["spread_line"], errors="coerce")
    home_favored = df["VEGAS_LINE"] > 0
    away_favored = df["VEGAS_LINE"] < 0

    df["FAVORITED"] = np.where(
        home_favored,
        df["home_team"].map(map_short_to_long),
        np.where(
            away_favored,
            df["away_team"].map(map_short_to_long),
            np.nan,  # Pick (line = 0)
        ),
    )
    # Negate for away-favored games so VEGAS_LINE is always negative spread
    # (matching the PFR convention where line is stored as a negative number)
    # pyrefly: ignore [missing-attribute]
    df["VEGAS_LINE"] = df["VEGAS_LINE"].abs() * np.where(away_favored, -1, 1)

    # --- Assemble canonical schema ---
    out = pd.DataFrame(
        {
            "GAME_ID": df["GAME_ID"],
            "WEEK_NUM": df["WEEK_NUM"].astype(int),
            "GAME_DAY_OF_WEEK": df["weekday"].fillna(""),
            "GAME_DATE": df["gameday"].fillna(""),
            "GAMETIME": df["GAMETIME"],
            "WINNER": df["WINNER"],
            "GAME_LOCATION": df["GAME_LOCATION"],
            "LOSER": df["LOSER"],
            "BOXSCORE_LINK": "",  # deprecated — not available from nflverse schedules
            "PTS_WINNER": df["PTS_WINNER"],
            "PTS_LOSER": df["PTS_LOSER"],
            "YARDS_WINNER": 0,  # not in schedules; set to 0 (unused downstream)
            "TURNOVERS_WINNER": 0,
            "YARDS_LOSER": 0,
            "TURNOVERS_LOSER": 0,
            "YEAR": df["YEAR"],
            "STADIUM": df["stadium"].fillna(""),
            "ROOF": df["roof"].fillna(""),
            "SURFACE": df["surface"].fillna(""),
            "VEGAS_LINE": df["VEGAS_LINE"],
            "OVER_UNDER": pd.to_numeric(df["total_line"], errors="coerce"),
            "FAVORITED": df["FAVORITED"],
            "WIN_OR_TIE": df["WIN_OR_TIE"],
            "DIV_GAME": pd.to_numeric(df["div_game"], errors="coerce").fillna(0).astype(int),
        }
    )

    # Sort to match legacy ordering
    out: DataFrame = out.sort_values(
        ["GAME_DATE", "GAMETIME", "GAME_ID"],
        ascending=True,
        ignore_index=True,
    )

    out_path = dataset_path(resolved_repo, "games")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info("Wrote %d canonical game rows to %s", len(out), out_path)
    return out_path
