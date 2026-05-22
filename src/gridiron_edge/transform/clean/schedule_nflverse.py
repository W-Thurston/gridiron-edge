# src/gridiron_edge/transform/clean/schedule_nflverse.py

"""Transform nflverse upcoming schedule data into the canonical schedule schema.

Maps unplayed nflverse games to the AWAY_TEAM/HOME_TEAM-oriented canonical
upcoming schedule schema used by Elo predict and simulation.

Canonical schedule schema (NFL_upcoming_schedule_cleaned.csv):
    WEEK_NUM            int     1-22
    GAME_DAY_OF_WEEK    str     "Sunday"
    GAME_DATE           str     "2025-10-05"
    AWAY_TEAM           str     "Kansas City Chiefs"   (long name)
    HOME_TEAM           str     "Los Angeles Chargers" (long name)
    GAMETIME            str     "16:25:00"
    YEAR                str     "2025-2026"
    GAME_ID             str     "2025_05_KC_LAC"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.transform.clean.games_nflverse import (
    _GAME_TYPE_TO_WEEK,
    _gametime_to_hhmmss,
    _map_short_to_long,
    _season_label,
)

logger = logging.getLogger(__name__)


def clean_nflverse_upcoming(
    *,
    repo: Path | None = None,
) -> Path:
    """Transform nflverse raw upcoming schedule into the canonical schedule CSV.

    Reads ``data/raw/NFL_upcoming_schedule_nflverse.csv``, maps to the
    canonical AWAY_TEAM/HOME_TEAM schema, and writes to
    ``data/cleaned/NFL_upcoming_schedule_cleaned.csv``.

    Args:
        repo: Absolute path to the repository root. Defaults to the value
            from ``get_settings()``.

    Returns:
        Absolute path to the written canonical upcoming schedule CSV.
    """
    settings = get_settings()
    resolved_repo = repo or settings.repo_root

    raw_path = dataset_path(resolved_repo, "schedule_upcoming_raw_nflverse")
    if not raw_path.exists():
        msg = (
            f"Raw nflverse upcoming schedule not found: {raw_path}. "
            "Run `gridiron ingest nflverse-upcoming` first."
        )
        raise FileNotFoundError(msg)

    logger.info("Reading raw nflverse upcoming schedule from %s", raw_path)
    df = pd.read_parquet(raw_path)

    # Confirm all rows are unplayed
    df = df.loc[df["result"].isna()].copy()

    if df.empty:
        logger.info("No upcoming games found in %s — season may be complete.", raw_path)
        out_path: Path = dataset_path(resolved_repo, "schedule_upcoming")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write empty CSV with correct column headers so downstream readers
        # don't fail on a missing file.
        empty = pd.DataFrame(
            columns=[
                "WEEK_NUM",
                "GAME_DAY_OF_WEEK",
                "GAME_DATE",
                "AWAY_TEAM",
                "HOME_TEAM",
                "GAMETIME",
                "YEAR",
                "GAME_ID",
            ]
        )
        empty.to_csv(out_path, index=False)
        return out_path

    logger.info("Processing %d upcoming games", len(df))

    def _resolve_week(row: pd.Series) -> int:
        gt = str(row["game_type"])
        if gt in _GAME_TYPE_TO_WEEK:
            return _GAME_TYPE_TO_WEEK[gt]
        return int(row["week"])

    df["WEEK_NUM"] = df.apply(_resolve_week, axis=1)

    # --- Map short codes to long names ---
    df["AWAY_TEAM"] = df["away_team"].map(_map_short_to_long)
    df["HOME_TEAM"] = df["home_team"].map(_map_short_to_long)

    # --- Other fields ---
    df["YEAR"] = df["season"].astype(int).map(_season_label)
    df["GAMETIME"] = df["gametime"].apply(_gametime_to_hhmmss)
    df["GAME_ID"] = df["game_id"].astype(str)

    out = pd.DataFrame(
        {
            "WEEK_NUM": df["WEEK_NUM"].astype(int),
            "GAME_DAY_OF_WEEK": df["weekday"].fillna("NULL_VALUE"),
            "GAME_DATE": df["gameday"].fillna("NULL_VALUE"),
            "AWAY_TEAM": df["AWAY_TEAM"],
            "HOME_TEAM": df["HOME_TEAM"],
            "GAMETIME": df["GAMETIME"],
            "YEAR": df["YEAR"],
            "GAME_ID": df["GAME_ID"],
        }
    )

    out = out.sort_values(
        ["WEEK_NUM", "GAME_DATE", "GAMETIME"],
        ascending=True,
        ignore_index=True,
    )

    out_path = dataset_path(resolved_repo, "schedule_upcoming")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info("Wrote %d upcoming game rows to %s", len(out), out_path)
    return out_path
