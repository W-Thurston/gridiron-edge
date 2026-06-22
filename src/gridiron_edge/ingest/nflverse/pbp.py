# src/gridiron_edge/ingest/nflverse/pbp.py

"""Fetch and store NFL play-by-play data from nflverse.

Downloads per-season PBP Parquet files from nflverse and stores them
permanently under ``data/raw/pbp/``. At ~20MB per season the full history
(1999-present) is ~540MB — small enough to keep permanently rather than
re-fetching on demand.

Each season's file is written once when complete and never overwritten,
matching the nflverse games ingest pattern. The current season is
refreshed weekly during the season.

Storage layout::

    data/raw/pbp/
        play_by_play_1999.parquet
        play_by_play_2000.parquet
        ...
        play_by_play_2025.parquet

Key PBP columns used downstream:
    game_id         str     "2025_01_PHI_GB"
    season          int     2025
    week            int     1-18 (reg season) or 19-22 (postseason)
    posteam         str     "KC"  (possession team, short code)
    defteam         str     "LAC" (defending team, short code)
    play_type       str     "pass" | "run" | "punt" | "kickoff" | ...
    pass            int     1 if dropback play
    rush            int     1 if designed run
    epa             float   Expected Points Added for this play
    success         int     1 if epa > 0
    qb_epa          float   QB-credited EPA (handles fumbles after catch)
    cp              float   Completion probability
    cpoe            float   Completion percentage over expected
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import current_nfl_season, get_settings

logger: Logger = logging.getLogger(__name__)

# Columns we actually need — dropping the rest reduces file size ~70%
# and speeds up downstream reads significantly.
_KEEP_COLUMNS: Final[list[str]] = [
    "game_id",
    "season",
    "week",
    "game_date",
    "posteam",
    "posteam_type",  # "home" or "away"
    "defteam",
    "play_type",
    "pass",
    "rush",
    "epa",
    "success",
    "qb_epa",
    "cp",
    "cpoe",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "first_down",
    "touchdown",
    "interception",
    "fumble_lost",
    "sack",
    "penalty",
    "home_team",
    "away_team",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
]


def _pbp_dir(repo: Path) -> Path:
    """Return the PBP raw storage directory, creating it if needed."""
    directory: Path = repo / "data" / "raw" / "pbp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _pbp_path(repo: Path, season: int) -> Path:
    """Return the Parquet path for a given season."""
    return _pbp_dir(repo) / f"play_by_play_{season}.parquet"


def _is_season_complete(season: int) -> bool:
    """Return True if the given season year is fully complete.

    A season starting in year N ends in February of year N+1.
    If the current date is past March of year N+1 the season is complete.

    Args:
        season: Season start year (e.g. 2024 for the 2024-2025 season).

    Returns:
        ``True`` if the season is complete and the Parquet file should
        not be overwritten on incremental updates.
    """
    now: datetime = datetime.now(UTC)
    # Season ends Super Bowl weekend ~Feb of year N+1.
    # Conservative: treat as complete after March 1 of N+1.
    cutoff_year: int = season + 1
    return now.year > cutoff_year or (now.year == cutoff_year and now.month >= 3)


def fetch_pbp(
    seasons: list[int] | None = None,
    *,
    start_season: int = 1999,
    repo: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Fetch and store PBP data for one or more seasons.

    Skips seasons whose Parquet file already exists unless ``force=True``
    or the season is the current (incomplete) season.

    Args:
        seasons: Specific season years to fetch (e.g. ``[2024, 2025]``).
            If ``None``, fetches from ``start_season`` to current season.
        start_season: First season to fetch when ``seasons`` is ``None``.
            Defaults to 1999 (nflverse coverage start).
        repo: Repository root. Defaults to settings repo root.
        force: If ``True``, re-fetch and overwrite existing files.

    Returns:
        List of Parquet file paths written.
    """
    # pyrefly: ignore [missing-import]
    import nflreadpy as nfl

    resolved_repo: Path = repo or get_settings().repo_root
    current: int = current_nfl_season()

    if seasons is None:
        seasons = list(range(start_season, current + 1))

    written: list[Path] = []

    for season in seasons:
        path: Path = _pbp_path(resolved_repo, season)
        complete: bool = _is_season_complete(season)

        if path.exists() and complete and not force:
            logger.debug("PBP %d already cached — skipping.", season)
            continue

        logger.info("Fetching PBP data for season %d...", season)
        try:
            df = nfl.load_pbp([season]).to_pandas()
        except Exception as exc:
            logger.warning("Failed to fetch PBP for season %d: %s", season, exc)
            continue

        # Keep only the columns we need
        available: list[str] = [c for c in _KEEP_COLUMNS if c in df.columns]
        df = df[available].copy()

        df.to_parquet(path, index=False)
        size_mb: float = path.stat().st_size / (1024 * 1024)
        logger.info("PBP %d written to %s (%.1f MB)", season, path, size_mb)
        written.append(path)

    return written


def fetch_pbp_refresh(*, repo: Path | None = None) -> list[Path]:
    """Refresh PBP data for the current season only.

    Called during weekly data refresh. Re-fetches the current season
    file since it grows throughout the season.

    Args:
        repo: Repository root.

    Returns:
        List containing the current season's Parquet path, or empty
        list if fetch failed.
    """
    current: int = current_nfl_season()
    return fetch_pbp(seasons=[current], repo=repo, force=True)


def load_pbp(
    seasons: list[int] | None = None,
    *,
    repo: Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load cached PBP data for one or more seasons.

    Args:
        seasons: Season years to load. If ``None``, loads all cached seasons.
        repo: Repository root.
        columns: Subset of columns to load. If ``None``, loads all stored
            columns. Passing only the columns you need is significantly
            faster for large multi-season loads.

    Returns:
        Combined PBP DataFrame. Empty DataFrame if no cached files exist.

    Raises:
        FileNotFoundError: If ``seasons`` are specified but none of their
            files exist in the cache.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    pbp_dir: Path = _pbp_dir(resolved_repo)

    if seasons is None:
        paths: list[Path] = sorted(pbp_dir.glob("play_by_play_*.parquet"))
    else:
        paths = [_pbp_path(resolved_repo, s) for s in seasons]
        missing: list[Path] = [p for p in paths if not p.exists()]
        if missing and len(missing) == len(paths):
            raise FileNotFoundError(
                f"No PBP files found for seasons {seasons}. Run 'gridiron ingest pbp' first."
            )
        paths = [p for p in paths if p.exists()]

    if not paths:
        return pd.DataFrame()

    frames: list[DataFrame] = [pd.read_parquet(p, columns=columns) for p in paths]
    return pd.concat(frames, ignore_index=True)
