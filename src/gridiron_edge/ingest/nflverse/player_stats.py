# src/gridiron_edge/ingest/nflverse/player_stats.py

"""Fetch and cache weekly player statistics from nflverse.

Uses ``nflreadpy.load_player_stats()`` which provides pre-aggregated
player-game-level statistics including passing, rushing, receiving,
EPA, and usage metrics.

Data is cached as per-season Parquet files at
``data/raw/player_stats/player_stats_{season}.parquet``.

Usage::

    from gridiron_edge.ingest.nflverse.player_stats import fetch_player_stats

    paths = fetch_player_stats()  # current season only
    paths = fetch_player_stats(all_years=True)  # 1999-present
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

# pyrefly: ignore [missing-import]
import nflreadpy as nfl
import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# First season with reliable nflverse player stats.
_STATS_RELIABLE_FROM: Final[int] = 1999

# Columns to retain from the 115 available in load_player_stats().
# Covers identity, passing, rushing, receiving, usage, and advanced metrics.
# Excludes fantasy scoring, headshot URLs, 2pt conversions, and sack fumble
# details (already captured at team level).
_KEEP_COLUMNS: Final[list[str]] = [
    # --- Identity ---
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "position_group",
    "team",
    "opponent_team",
    "game_id",
    "season",
    "season_type",
    "week",
    # --- Passing ---
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_epa",
    "passing_first_downs",
    "passing_cpoe",
    # --- Rushing ---
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_epa",
    # --- Receiving ---
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_fumbles_lost",
    "receiving_epa",
    # --- Usage / Advanced ---
    "target_share",
    "air_yards_share",
    "wopr",
    "pacr",
    "racr",
]


def _player_stats_dir(repo: Path) -> Path:
    """Return the player stats storage directory, creating it if needed."""
    directory: Path = repo / "data" / "raw" / "player_stats"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _player_stats_path(repo: Path, season: int) -> Path:
    """Return the Parquet path for a given season."""
    return _player_stats_dir(repo) / f"player_stats_{season}.parquet"


def fetch_player_stats(
    *,
    all_years: bool = False,
    seasons: list[int] | None = None,
    repo: Path | None = None,
) -> list[Path]:
    """Fetch weekly player stats and write per-season Parquet files.

    Args:
        all_years: If ``True``, fetch all seasons from 1999 to present.
            If ``False``, fetch only the current season.
        seasons: Explicit list of seasons to fetch. Overrides ``all_years``.
        repo: Repository root. Defaults to settings.

    Returns:
        List of paths to written Parquet files.
    """
    resolved_repo: Path = repo or get_settings().repo_root

    if seasons is not None:
        target_seasons: list[int] = seasons
    elif all_years:
        current_year: int = pd.Timestamp.now().year
        target_seasons = list(range(_STATS_RELIABLE_FROM, current_year + 1))
    else:
        current_year = pd.Timestamp.now().year
        # During off-season (month < 6), current season hasn't started
        season_year: int = current_year if pd.Timestamp.now().month >= 6 else current_year - 1
        target_seasons = [season_year]

    written: list[Path] = []

    for season in target_seasons:
        path: Path = _player_stats_path(resolved_repo, season)

        if path.exists() and not all_years:
            logger.info("Player stats %d already cached at %s", season, path)
            written.append(path)
            continue

        try:
            df: DataFrame = nfl.load_player_stats([season]).to_pandas()
        except Exception:  # nflreadpy may raise varied errors (network, parse, schema)
            logger.warning("Failed to fetch player stats for season %d", season)
            continue

        # Retain only columns that exist in this season's data
        available: list[str] = [c for c in _KEEP_COLUMNS if c in df.columns]
        missing: list[str] = [c for c in _KEEP_COLUMNS if c not in df.columns]
        if missing:
            logger.debug("Player stats %d missing columns: %s", season, missing)
        df = df.loc[:, available].copy()

        df.to_parquet(path, index=False)
        size_mb: float = path.stat().st_size / (1024 * 1024)
        logger.info(
            "Player stats %d written to %s (%.1f MB, %d rows)",
            season,
            path,
            size_mb,
            len(df),
        )
        written.append(path)

    return written


def load_player_stats(
    *,
    seasons: list[int] | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Load cached player stats into a single DataFrame.

    Args:
        seasons: Seasons to load. If ``None``, loads all cached seasons.
        repo: Repository root.

    Returns:
        Combined DataFrame of all requested seasons.

    Raises:
        FileNotFoundError: If no cached player stats are found.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    stats_dir: Path = _player_stats_dir(resolved_repo)

    if seasons is not None:
        paths: list[Path] = [_player_stats_path(resolved_repo, s) for s in seasons]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(stats_dir.glob("player_stats_*.parquet"))

    if not paths:
        msg: str = (
            f"No player stats found in {stats_dir}. "
            "Run: gridiron run-data-pipeline --only fetch-player-stats"
        )
        raise FileNotFoundError(msg)

    frames: list[DataFrame] = [pd.read_parquet(p) for p in paths]
    combined: DataFrame = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d player-game rows across %d seasons",
        len(combined),
        len(paths),
    )
    return combined
