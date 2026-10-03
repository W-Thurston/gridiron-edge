# src/gridiron_edge/ingest/nflverse/weekly_availability.py

"""Fetch and cache nflverse depth charts and injury reports.

Both datasets are cached as-is, one Parquet file per season, at
``data/raw/depth_charts/depth_charts_{season}.parquet`` and
``data/raw/injuries/injuries_{season}.parquet``. No column is dropped or
renamed: the 2025 depth chart format differs from earlier seasons and each
consumer owns its own interpretation. Depth charts begin in 2001 and injury
reports in 2009.

Usage::

    from gridiron_edge.ingest.nflverse.weekly_availability import fetch_injuries

    paths = fetch_injuries(all_years=True)
"""

from __future__ import annotations

from collections.abc import Callable
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

DEPTH_CHARTS_FIRST_SEASON: Final[int] = 2001
INJURIES_FIRST_SEASON: Final[int] = 2009

_DEPTH_CHARTS: Final[str] = "depth_charts"
_INJURIES: Final[str] = "injuries"


def _season_path(repo: Path, dataset: str, season: int) -> Path:
    """Return the cache path of one dataset season."""
    return repo / "data" / "raw" / dataset / f"{dataset}_{season}.parquet"


def _fetch_cached(
    *,
    dataset: str,
    loader: Callable[[list[int]], object],
    first_season: int,
    all_years: bool,
    seasons: list[int] | None,
    force: bool,
    repo: Path | None,
) -> list[Path]:
    """Fetch seasons of one dataset into the per-season cache.

    A cached season is skipped unless ``all_years`` or ``force`` is set. A fetch
    that fails or returns no rows never replaces a cached file.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    now = pd.Timestamp.now()
    if seasons is not None:
        target_seasons: list[int] = seasons
    elif all_years:
        target_seasons = list(range(first_season, now.year + 1))
    else:
        target_seasons = [now.year if now.month >= 6 else now.year - 1]

    written: list[Path] = []
    for season in target_seasons:
        path: Path = _season_path(resolved_repo, dataset, season)
        if path.exists() and not (all_years or force):
            logger.info("%s %d already cached at %s", dataset, season, path)
            written.append(path)
            continue
        try:
            frame: DataFrame = loader([season]).to_pandas()  # type: ignore[attr-defined]
        except Exception:  # nflreadpy may raise varied errors (network, parse, schema)
            logger.warning("Failed to fetch %s for season %d", dataset, season)
            continue
        if frame.empty:
            logger.warning("%s for season %d returned no rows; cache unchanged", dataset, season)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        logger.info("%s %d written to %s (%d rows)", dataset, season, path, len(frame))
        written.append(path)
    return written


def _load_cached(
    dataset: str,
    *,
    seasons: list[int] | None,
    repo: Path | None,
) -> dict[int, DataFrame]:
    """Load cached seasons of one dataset, keyed by season."""
    resolved_repo: Path = repo or get_settings().repo_root
    directory: Path = resolved_repo / "data" / "raw" / dataset
    if seasons is not None:
        paths: list[Path] = [_season_path(resolved_repo, dataset, s) for s in seasons]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(directory.glob(f"{dataset}_*.parquet"))
    if not paths:
        command: str = dataset.replace("_", "-")
        msg: str = f"No cached {dataset} found in {directory}. Run: gridiron ingest {command}"
        raise FileNotFoundError(msg)
    return {int(p.stem.rsplit("_", 1)[1]): pd.read_parquet(p) for p in paths}


def fetch_depth_charts(
    *,
    all_years: bool = False,
    seasons: list[int] | None = None,
    force: bool = False,
    repo: Path | None = None,
) -> list[Path]:
    """Fetch nflverse depth charts (2001 onward) into the per-season cache."""
    return _fetch_cached(
        dataset=_DEPTH_CHARTS,
        loader=nfl.load_depth_charts,
        first_season=DEPTH_CHARTS_FIRST_SEASON,
        all_years=all_years,
        seasons=seasons,
        force=force,
        repo=repo,
    )


def fetch_injuries(
    *,
    all_years: bool = False,
    seasons: list[int] | None = None,
    force: bool = False,
    repo: Path | None = None,
) -> list[Path]:
    """Fetch nflverse injury reports (2009 onward) into the per-season cache."""
    return _fetch_cached(
        dataset=_INJURIES,
        loader=nfl.load_injuries,
        first_season=INJURIES_FIRST_SEASON,
        all_years=all_years,
        seasons=seasons,
        force=force,
        repo=repo,
    )


def load_depth_charts(
    *,
    seasons: list[int] | None = None,
    repo: Path | None = None,
) -> dict[int, DataFrame]:
    """Load cached depth charts keyed by season; raises if none are cached."""
    return _load_cached(_DEPTH_CHARTS, seasons=seasons, repo=repo)


def load_injuries(
    *,
    seasons: list[int] | None = None,
    repo: Path | None = None,
) -> dict[int, DataFrame]:
    """Load cached injury reports keyed by season; raises if none are cached."""
    return _load_cached(_INJURIES, seasons=seasons, repo=repo)
