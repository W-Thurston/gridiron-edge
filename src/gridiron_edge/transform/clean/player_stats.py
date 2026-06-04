# src/gridiron_edge/transform/clean/player_stats.py

"""Clean and transform raw player game logs for downstream feature engineering.

Reads cached per-season Parquet files from ``data/raw/player_stats/``,
normalizes team codes, constructs ``game_id`` via schedule join, tags
skill positions, and writes a single ``data/cleaned/player_game_logs.parquet``.

Usage::

    from gridiron_edge.transform.clean.player_stats import clean_player_stats

    path = clean_player_stats()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

# pyrefly: ignore [missing-import]
import nflreadpy as nfl
import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.ingest.nflverse.player_stats import load_player_stats

logger = logging.getLogger(__name__)

# Historical team code normalization: relocated/renamed franchises.
# Maps old abbreviation → current abbreviation.
_TEAM_CODE_MAP: Final[dict[str, str]] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
}

# Skill positions used for prop models.
_SKILL_POSITIONS: Final[set[str]] = {"QB", "RB", "WR", "TE", "FB"}


def _normalize_team_codes(df: DataFrame) -> DataFrame:
    """Replace historical team abbreviations with current ones."""
    df["team"] = df["team"].replace(_TEAM_CODE_MAP)
    df["opponent_team"] = df["opponent_team"].replace(_TEAM_CODE_MAP)
    return df


def _build_schedule_lookup() -> DataFrame:
    """Fetch all nflverse schedules and return a (season, week, home, away, game_id) lookup.

    Normalizes team codes in the schedule to match player stats.
    """
    current_year: int = pd.Timestamp.now().year
    seasons: list[int] = list(range(1999, current_year + 1))

    try:
        sched: DataFrame = nfl.load_schedules(seasons).to_pandas()
    except Exception:
        logger.warning("Failed to fetch full schedule range, trying year-by-year")
        frames: list[DataFrame] = []
        for s in seasons:
            try:
                frames.append(nfl.load_schedules([s]).to_pandas())
            except Exception:
                logger.debug("No schedule for %d", s)
        sched = pd.concat(frames, ignore_index=True)

    sched = sched.loc[:, ["game_id", "season", "week", "home_team", "away_team"]].copy()
    sched["home_team"] = sched["home_team"].replace(_TEAM_CODE_MAP)
    sched["away_team"] = sched["away_team"].replace(_TEAM_CODE_MAP)
    return sched


def _join_game_id(df: DataFrame, schedule: DataFrame) -> DataFrame:
    """Join game_id from schedule onto player stats.

    Player stats have (season, week, team, opponent_team) but not a
    reliable game_id. We join against the schedule twice — once assuming
    the player's team is home, once assuming away — then coalesce.
    """
    # Drop the unreliable game_id from player stats if present
    if "game_id" in df.columns:
        df = df.drop(columns=["game_id"])

    # Join as home team
    # pyrefly: ignore [bad-assignment]
    home_join: DataFrame = df.merge(
        schedule.rename(columns={"game_id": "_gid_home"}),
        left_on=["season", "week", "team"],
        right_on=["season", "week", "home_team"],
        how="left",
    )[["_gid_home"]]

    # Join as away team
    # pyrefly: ignore [bad-assignment]
    away_join: DataFrame = df.merge(
        schedule.rename(columns={"game_id": "_gid_away"}),
        left_on=["season", "week", "team"],
        right_on=["season", "week", "away_team"],
        how="left",
    )[["_gid_away"]]

    df["game_id"] = home_join["_gid_home"].fillna(away_join["_gid_away"])

    matched: int = df["game_id"].notna().sum()
    total: int = len(df)
    logger.info("game_id matched: %d / %d (%.1f%%)", matched, total, matched / total * 100)

    return df


def clean_player_stats(
    *,
    repo: Path | None = None,
) -> Path:
    """Clean raw player stats and write to Parquet.

    Steps:
        1. Load all cached raw player stats
        2. Normalize team codes (OAK→LV, SD→LAC, STL→LA, JAC→JAX)
        3. Construct game_id via schedule join
        4. Tag skill positions
        5. Drop rows with zero total stats (active but no recorded stats)
        6. Write to data/cleaned/player_game_logs.parquet

    Args:
        repo: Repository root.

    Returns:
        Path to the written Parquet file.
    """
    resolved_repo: Path = repo or get_settings().repo_root

    # 1. Load raw
    df: DataFrame = load_player_stats(repo=resolved_repo)
    n_raw: int = len(df)
    logger.info("Raw player stats: %d rows", n_raw)

    # 2. Normalize team codes
    df = _normalize_team_codes(df)

    # 3. Construct game_id
    schedule: DataFrame = _build_schedule_lookup()
    df = _join_game_id(df, schedule)

    # 4. Tag skill positions
    df["is_skill"] = df["position"].isin(_SKILL_POSITIONS)

    # 5. Drop rows with no meaningful stats
    # A player who was active but recorded zero across all stat columns
    stat_cols: list[str] = [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
    ]
    available_stat_cols: list[str] = [c for c in stat_cols if c in df.columns]
    # pyrefly: ignore [bad-argument-type]
    has_stats: pd.Series = df[available_stat_cols].fillna(0).sum(axis=1) > 0
    n_empty: int = (~has_stats).sum()
    df = df.loc[has_stats, :].copy()
    logger.info("Dropped %d rows with zero stats (%d → %d)", n_empty, n_raw, len(df))

    # 6. Write
    out_path: Path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_mb: float = out_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Cleaned player stats written to %s (%.1f MB, %d rows, %d players)",
        out_path,
        size_mb,
        len(df),
        df["player_id"].nunique(),
    )
    return out_path
