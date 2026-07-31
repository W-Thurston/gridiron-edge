# src/gridiron_edge/ingest/nflverse/games.py

"""Fetch historical NFL game results from nflverse via nflreadpy.

Pulls ``nflreadpy.load_schedules()`` for the requested season range and writes the
raw response to ``data/raw/NFL_wk_by_wk_nflverse.parquet``. The transform
layer (``transform.clean.games_nflverse``) is responsible for mapping the
nflverse schema to the canonical games schema.

nflverse schedules are updated nightly after each game day during the season.
The cleanest weekly snapshot is available Thursday (after the NFL incorporates
stat corrections from Mon-Wed).

Available from 1999 onwards.
"""

from __future__ import annotations

import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
import nflreadpy as nfl
import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import Settings, current_nfl_season, get_settings
from gridiron_edge.datasets.registry import dataset_path

logger = logging.getLogger(__name__)

# Earliest season nflverse reliably covers.
NFLVERSE_MIN_SEASON: int = 1999


def fetch_nflverse_games(
    *,
    seasons: list[int] | None = None,
    start_season: int = NFLVERSE_MIN_SEASON,
    end_season: int | None = None,
    repo: Path | None = None,
) -> Path:
    """Fetch NFL game results from nflverse and write to raw Parquet.

    Fetches ``nflreadpy.load_schedules()`` for the requested season range
    and overwrites the raw Parquet file. Seasons are specified as the calendar
    year the season starts (e.g. ``2025`` for the 2025-2026 season).

    Args:
        seasons: Explicit list of season years to fetch. If provided,
            ``start_season`` and ``end_season`` are ignored.
        start_season: First season year when ``seasons`` is not provided.
            Defaults to ``1999`` (earliest nflverse coverage).
        end_season: Last season year when ``seasons`` is not provided.
            Defaults to the current NFL season.
        repo: Absolute path to the repository root.

    Returns:
        Absolute path to the written raw Parquet file.

    Raises:
        ValueError: If the resolved season list is empty.
    """
    settings = get_settings()
    resolved_repo = repo or settings.repo_root

    if seasons is not None:
        season_list = sorted(seasons)
    else:
        resolved_end = end_season or current_nfl_season()
        season_list = list(range(start_season, resolved_end + 1))

    season_list = [s for s in season_list if s >= NFLVERSE_MIN_SEASON]
    if not season_list:
        msg = (
            f"No valid seasons to fetch. All seasons are before the "
            f"nflverse minimum of {NFLVERSE_MIN_SEASON}."
        )
        raise ValueError(msg)

    logger.info("Fetching nflverse schedules for seasons: %s", season_list)

    df: pd.DataFrame = nfl.load_schedules(season_list).to_pandas()

    out_path = dataset_path(resolved_repo, "games_raw_nflverse")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info("Wrote %d rows to %s", len(df), out_path)
    return out_path


def refresh_nflverse_game_seasons(
    *,
    seasons: list[int],
    repo: Path | None = None,
) -> Path:
    """Refresh selected seasons while preserving all other raw history.

    Fetches the complete nflverse schedule for each requested season,
    removes existing rows for only those seasons, and writes the combined
    historical artifact.

    Args:
        seasons: Season years to refresh.
        repo: Absolute path to the repository root.

    Returns:
        Absolute path to the updated raw Parquet file.

    Raises:
        ValueError: If no valid nflverse seasons are supplied.
    """
    settings: Settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    season_list: list[int] = sorted({season for season in seasons if season >= NFLVERSE_MIN_SEASON})
    if not season_list:
        raise ValueError(
            "No valid seasons to refresh. All seasons are before the "
            f"nflverse minimum of {NFLVERSE_MIN_SEASON}."
        )

    logger.info(
        "Refreshing nflverse seasons: %s",
        season_list,
    )

    refreshed: DataFrame = nfl.load_schedules(season_list).to_pandas()

    raw_path: Path = dataset_path(
        resolved_repo,
        "games_raw_nflverse",
    )
    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if raw_path.exists():
        existing: DataFrame = pd.read_parquet(raw_path)
        retained = existing.loc[
            ~existing["season"].astype(int).isin(season_list),
            :,
        ].copy()
        combined: DataFrame = pd.concat(
            [
                retained,
                refreshed,
            ],
            ignore_index=True,
        )
    else:
        combined = refreshed

    combined = combined.sort_values(
        [
            "season",
            "week",
            "game_id",
        ],
        kind="stable",
        ignore_index=True,
    )

    if combined["game_id"].duplicated().any():
        duplicated: list = sorted(
            combined.loc[
                combined["game_id"].duplicated(keep=False),
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Refreshed nflverse games contain duplicate game IDs: " + ", ".join(duplicated)
        )

    combined.to_parquet(
        raw_path,
        index=False,
    )

    logger.info(
        "Seasons %s refreshed. Total rows: %d written to %s",
        season_list,
        len(combined),
        raw_path,
    )
    return raw_path


def fetch_nflverse_games_refresh(
    *,
    season: int | None = None,
    repo: Path | None = None,
) -> Path:
    """Refresh a single season in the raw Parquet file.

    Reads the existing raw Parquet (if present), drops all rows for the
    target season, fetches fresh data from nflverse, and writes back.

    This is the standard weekly-update path. Since nflverse always returns
    the full season in one response (completed games with scores + future
    games with ``result = NaN``), there is no need to specify which weeks
    to refresh. Running this any time during the season brings all completed
    results and the remaining schedule fully up to date.

    Args:
        season: The season year to refresh (e.g. ``2025``). Defaults to the
            current NFL season inferred from today's date.
        repo: Absolute path to the repository root.

    Returns:
        Absolute path to the written raw Parquet file.
    """
    target: int = season or current_nfl_season()

    return refresh_nflverse_game_seasons(
        seasons=[target],
        repo=repo,
    )
