# src/gridiron_edge/transform/clean/games_nflverse.py

"""Transform nflverse raw schedule data into the canonical games schema.

Preserves nflverse home/away identity and scores as the canonical game
orientation.

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
    GAMETIME            str     "20:20:00"  (HH:MM:SS)
    AWAY_TEAM           str     "Philadelphia Eagles"  (long name)
    HOME_TEAM           str     "Green Bay Packers"    (long name)
    AWAY_SCORE          int     20
    HOME_SCORE          int     17
    IS_NEUTRAL_SITE     int     0 | 1
    YEAR                str     "2025-2026"
    STADIUM             str     "Lambeau Field"
    ROOF                str     "outdoors"
    SURFACE             str     "grass"
    VEGAS_LINE          float   -3.0  (spread from winner perspective)
    OVER_UNDER          float   47.5
    FAVORITED           str     "Green Bay Packers"  (long name of favored team)
    DIV_GAME            int     1 if both teams are in the same division, else 0

Notes:
    - GAMETIME is stored as HH:MM:SS.
    - YEAR uses the "YYYY-YYYY+1" season label format (e.g. "2025-2026").
    - AWAY_TEAM, HOME_TEAM, AWAY_SCORE, and HOME_SCORE preserve nflverse
      schedule orientation directly.
    - nflverse uses short team codes. This module maps them to long names
      using the team_metadata reference dataset (long/short name columns)
    - IS_NEUTRAL_SITE is 1 only when nflverse location is "Neutral".
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    gametime_to_hhmmss,
    map_short_to_long,
    season_label,
)

logger: Logger = logging.getLogger(__name__)

_GAMES_COLUMNS: list[str] = [
    "GAME_ID",
    "WEEK_NUM",
    "GAME_DAY_OF_WEEK",
    "GAME_DATE",
    "GAMETIME",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
    "IS_NEUTRAL_SITE",
    "YEAR",
    "STADIUM",
    "ROOF",
    "SURFACE",
    "VEGAS_LINE",
    "OVER_UNDER",
    "FAVORITED",
    "DIV_GAME",
]


def _handle_empty_games(out_path: Path) -> Path:
    """Handle the no-completed-games case without clobbering history.

    Offseason / upcoming-season fetches filter to zero completed games.
    If a populated history table already exists, refuse to overwrite it
    (leave intact, warn). Otherwise write the empty schema (first run).
    """
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
        except (pd.errors.EmptyDataError, OSError):
            existing = pd.DataFrame()
        if not existing.empty:
            logger.warning(
                "clean-games produced 0 completed games but existing "
                "history has %d rows — refusing to overwrite (offseason / "
                "upcoming-season fetch?). Existing table left intact.",
                len(existing),
            )
            return out_path

    logger.info("No completed games found and no existing history — writing empty games schema.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=_GAMES_COLUMNS).to_csv(out_path, index=False)
    return out_path


def _validate_home_away_games(  # noqa: PLR0912
    games: DataFrame,
) -> None:
    """Validate canonical home/away identity and score invariants."""
    required = {
        "GAME_ID",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_SCORE",
        "HOME_SCORE",
        "IS_NEUTRAL_SITE",
    }
    missing: list[str] = sorted(required - set(games.columns))
    if missing:
        raise ValueError("Cleaned games are missing required columns: " + ", ".join(missing))

    if games["GAME_ID"].isna().any():
        raise ValueError("Cleaned game IDs must not contain nulls.")

    if games["GAME_ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Cleaned game IDs must not contain empty values.")

    if games["GAME_ID"].duplicated().any():
        duplicated: list[str] = sorted(
            games.loc[
                games["GAME_ID"].duplicated(keep=False),
                "GAME_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError("Cleaned games contain duplicate game IDs: " + ", ".join(duplicated))

    for column in (
        "AWAY_TEAM",
        "HOME_TEAM",
    ):
        if games[column].isna().any():
            raise ValueError(f"{column} must not contain nulls.")

        if games[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must not contain empty values.")

    same_team: Series[bool] = games["AWAY_TEAM"].astype(str) == games["HOME_TEAM"].astype(str)
    if same_team.any():
        game_ids: list[str] = sorted(
            games.loc[
                same_team,
                "GAME_ID",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError("Away and home team must differ for games: " + ", ".join(game_ids))

    away_scores = pd.to_numeric(
        games["AWAY_SCORE"],
        errors="coerce",
    )
    home_scores = pd.to_numeric(
        games["HOME_SCORE"],
        errors="coerce",
    )

    # pyrefly: ignore [missing-attribute]
    away_present = away_scores.notna()
    # pyrefly: ignore [missing-attribute]
    home_present = home_scores.notna()

    if not away_present.equals(home_present):
        raise ValueError("AWAY_SCORE and HOME_SCORE must both be present or both be missing.")

    completed = away_present & home_present

    # pyrefly: ignore [missing-attribute]
    if (away_scores.loc[completed] < 0).any():
        raise ValueError("AWAY_SCORE must not contain negative values.")

    # pyrefly: ignore [missing-attribute]
    if (home_scores.loc[completed] < 0).any():
        raise ValueError("HOME_SCORE must not contain negative values.")

    if games["IS_NEUTRAL_SITE"].isna().any():
        raise ValueError("IS_NEUTRAL_SITE must not contain nulls.")

    neutral_values = set(
        # pyrefly: ignore [missing-attribute]
        pd.to_numeric(
            games["IS_NEUTRAL_SITE"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if neutral_values - {0, 1}:
        raise ValueError("IS_NEUTRAL_SITE must contain only 0 or 1.")


def clean_nflverse_games(
    *,
    repo: Path | None = None,
) -> Path:
    """Transform nflverse raw games CSV into the canonical cleaned games CSV.

    Reads the registered nflverse historical schedule, filters to completed
    regular-season and postseason games, preserves canonical Away/Home
    identity and scores, and writes the cleaned games dataset.

    Unplayed games (``result = NaN``) are excluded - they belong in the
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
        return _handle_empty_games(dataset_path(resolved_repo, "games"))

    # --- Normalise week numbers ---
    # REG games have integer weeks; postseason game_types map to 19-22.
    def _resolve_week(row: pd.Series) -> int:
        gt = str(row["game_type"])
        if gt in GAME_TYPE_TO_WEEK:
            return GAME_TYPE_TO_WEEK[gt]
        return int(row["week"])

    df["WEEK_NUM"] = df.apply(_resolve_week, axis=1)

    # --- Preserve canonical home/away identity and scores ---
    df["AWAY_TEAM"] = df["away_team"].astype(str).map(map_short_to_long)
    df["HOME_TEAM"] = df["home_team"].astype(str).map(map_short_to_long)

    # pyrefly: ignore [missing-attribute]
    df["AWAY_SCORE"] = pd.to_numeric(
        df["away_score"],
        errors="raise",
    ).astype(int)
    # pyrefly: ignore [missing-attribute]
    df["HOME_SCORE"] = pd.to_numeric(
        df["home_score"],
        errors="raise",
    ).astype(int)

    neutral_mask = df["location"].fillna("").astype(str).str.strip().eq("Neutral")
    df["IS_NEUTRAL_SITE"] = neutral_mask.astype(int)

    # --- YEAR label ---
    df["YEAR"] = df["season"].astype(int).map(season_label)

    # --- GAMETIME ---
    df["GAMETIME"] = df["gametime"].apply(gametime_to_hhmmss)

    # --- GAME_ID: use nflverse alt_game_id if available, else game_id ---
    # nflverse game_id format: "2025_01_PHI_GB" - already matches our convention
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
        df["HOME_TEAM"],
        np.where(
            away_favored,
            df["AWAY_TEAM"],
            np.nan,
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
            "AWAY_TEAM": df["AWAY_TEAM"],
            "HOME_TEAM": df["HOME_TEAM"],
            "AWAY_SCORE": df["AWAY_SCORE"],
            "HOME_SCORE": df["HOME_SCORE"],
            "IS_NEUTRAL_SITE": df["IS_NEUTRAL_SITE"],
            "YEAR": df["YEAR"],
            "STADIUM": df["stadium"].fillna(""),
            "ROOF": df["roof"].fillna(""),
            "SURFACE": df["surface"].fillna(""),
            "VEGAS_LINE": df["VEGAS_LINE"],
            "OVER_UNDER": pd.to_numeric(df["total_line"], errors="coerce"),
            "FAVORITED": df["FAVORITED"],
            # pyrefly: ignore [missing-attribute]
            "DIV_GAME": pd.to_numeric(df["div_game"], errors="coerce").fillna(0).astype(int),
        }
    )

    # Sort deterministically by scheduled date, time, and game ID.
    out: DataFrame = out.sort_values(
        ["GAME_DATE", "GAMETIME", "GAME_ID"],
        ascending=True,
        ignore_index=True,
    )

    _validate_home_away_games(out)

    out_path = dataset_path(resolved_repo, "games")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info("Wrote %d canonical game rows to %s", len(out), out_path)
    return out_path
