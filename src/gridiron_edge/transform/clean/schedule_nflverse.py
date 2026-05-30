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
from logging import Logger
from pathlib import Path

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


def _check_stadium_coverage(
    raw_df: DataFrame,
    stadiums_df: DataFrame,
    season_label: str,
) -> None:
    """Warn about any stadium in the upcoming schedule not in the reference CSV.

    Compares the ``stadium`` column from the raw nflverse schedule against the
    ``STADIUM`` column of the stadium reference dataset.  Any stadium name
    present in the schedule but absent from the reference will result in a
    missing-coordinates failure during weather ingest.

    Emits one WARNING log per missing stadium including the teams that play
    there and the game count, so you have enough context to add the entry to
    ``NFL_stadium_reference.csv`` before running the weather pipeline.

    Args:
        raw_df: Raw nflverse upcoming schedule DataFrame (pre-clean).
        stadiums_df: Stadium reference DataFrame loaded from the registry.
        season_label: Human-readable season label for log messages
            (e.g. ``"2025-2026"``).
    """
    # Upcoming games may not have a stadium assigned yet (neutral site TBD, etc.)
    schedule_stadiums: set[str] = set(
        raw_df["stadium"].dropna().astype(str).str.strip().tolist()
    ) - {""}

    if not schedule_stadiums:
        logger.debug("No stadium names found in upcoming schedule — skipping coverage check.")
        return

    reference_stadiums: set[str] = set(
        stadiums_df["STADIUM"].dropna().astype(str).str.strip().tolist()
    ) - {""}

    missing: set[str] = schedule_stadiums - reference_stadiums

    if not missing:
        logger.info(
            "Stadium coverage check passed — all %d upcoming stadiums are in the reference.",
            len(schedule_stadiums),
        )
        return

    # Build a per-stadium summary: which teams play there and how many games
    games_at: dict[str, list[str]] = {}
    for _, row in raw_df.iterrows():
        stadium = str(row.get("stadium", "")).strip()
        if stadium not in missing:
            continue
        away = str(row.get("away_team", "")).strip()
        home = str(row.get("home_team", "")).strip()
        matchup = f"{away} @ {home}"
        games_at.setdefault(stadium, []).append(matchup)

    logger.warning(
        "Stadium coverage check FAILED for season %s — "
        "%d stadium(s) in the upcoming schedule have no entry in NFL_stadium_reference.csv. "
        "Weather ingest will skip these games until coordinates are added.",
        season_label,
        len(missing),
    )
    for stadium in sorted(missing):
        matchups = games_at.get(stadium, [])
        n_games = len(matchups)
        # Show up to 3 example matchups to keep the log readable
        sample = ", ".join(matchups[:3])
        if n_games > 3:
            sample += f", ... (+{n_games - 3} more)"
        logger.warning(
            "  Missing stadium: '%s' | %d game(s) | e.g. %s",
            stadium,
            n_games,
            sample,
        )
    logger.warning(
        "  → Add the missing stadium(s) to NFL_stadium_reference.csv with "
        "STADIUM, HOME_TEAM, YEAR, LATITUDE, LONGITUDE, ALTITUDE columns "
        "before running `gridiron ingest weather-backfill`.",
    )


def clean_nflverse_upcoming(
    *,
    repo: Path | None = None,
) -> Path:
    """Transform nflverse raw upcoming schedule into the canonical schedule CSV.

    Reads ``data/raw/NFL_upcoming_schedule_nflverse.csv``, maps to the
    canonical AWAY_TEAM/HOME_TEAM schema, and writes to
    ``data/cleaned/NFL_upcoming_schedule_cleaned.csv``.

    Also performs a stadium coverage check: any stadium in the upcoming
    schedule that is absent from the stadium reference CSV is logged as a
    WARNING so it can be added before weather ingest runs.

    Args:
        repo: Absolute path to the repository root. Defaults to the value
            from ``get_settings()``.

    Returns:
        Absolute path to the written canonical upcoming schedule CSV.
    """
    settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    raw_path: Path = dataset_path(resolved_repo, "schedule_upcoming_raw_nflverse")
    if not raw_path.exists():
        msg: str = (
            f"Raw nflverse upcoming schedule not found: {raw_path}. "
            "Run `gridiron ingest nflverse-upcoming` first."
        )
        raise FileNotFoundError(msg)

    logger.info("Reading raw nflverse upcoming schedule from %s", raw_path)
    df: DataFrame = pd.read_parquet(raw_path)

    # Confirm all rows are unplayed
    df = df.loc[df["result"].isna(), :].copy()

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

    # ── Stadium coverage check ────────────────────────────────────────────
    # Run before the transform so we're working with raw nflverse stadium
    # names, which is exactly what gets written to the games CSV and used
    # as the join key in weather ingest.
    stadiums_path: Path = dataset_path(resolved_repo, "stadiums")
    if stadiums_path.exists():
        stadiums_df: DataFrame = pd.read_csv(stadiums_path)
        # Derive season label from the first row's season value for logging
        season_int: int = int(df["season"].iloc[0]) if "season" in df.columns else 0
        season_lbl: str = season_label(season_int) if season_int else "unknown"
        _check_stadium_coverage(df, stadiums_df, season_lbl)
    else:
        logger.warning(
            "Stadium reference file not found at %s — skipping coverage check.",
            stadiums_path,
        )

    # ── Transform ─────────────────────────────────────────────────────────
    def _resolve_week(row: pd.Series) -> int:
        gt = str(row["game_type"])
        if gt in GAME_TYPE_TO_WEEK:
            return GAME_TYPE_TO_WEEK[gt]
        return int(row["week"])

    df["WEEK_NUM"] = df.apply(_resolve_week, axis=1)

    # --- Map short codes to long names ---
    df["AWAY_TEAM"] = df["away_team"].map(map_short_to_long)
    df["HOME_TEAM"] = df["home_team"].map(map_short_to_long)

    # --- Other fields ---
    df["YEAR"] = df["season"].astype(int).map(season_label)
    df["GAMETIME"] = df["gametime"].apply(gametime_to_hhmmss)
    df["GAME_ID"] = df["game_id"].astype(str)

    out = pd.DataFrame(
        {
            "WEEK_NUM": df["WEEK_NUM"].astype(int),
            "GAME_DAY_OF_WEEK": df["weekday"].fillna(""),
            "GAME_DATE": df["gameday"].fillna(""),
            "AWAY_TEAM": df["AWAY_TEAM"],
            "HOME_TEAM": df["HOME_TEAM"],
            "GAMETIME": df["GAMETIME"],
            "YEAR": df["YEAR"],
            "GAME_ID": df["GAME_ID"],
        }
    )

    out: DataFrame = out.sort_values(
        ["WEEK_NUM", "GAME_DATE", "GAMETIME"],
        ascending=True,
        ignore_index=True,
    )

    out_path = dataset_path(resolved_repo, "schedule_upcoming")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    logger.info("Wrote %d upcoming game rows to %s", len(out), out_path)
    return out_path
