# src/gridiron_edge/ingest/nflverse/schedule.py

"""Fetch upcoming (unplayed) NFL game schedule from nflverse.

nflverse schedules include the full season for a given year, with completed
games having ``result`` populated and future games having ``result = NaN``.
This module isolates the upcoming-games slice and writes it as the raw
upcoming schedule for downstream transform.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path

logger = logging.getLogger(__name__)


def fetch_nflverse_upcoming(
    *,
    season: int,
    repo: Path | None = None,
) -> Path:
    """Fetch upcoming (unplayed) games for a season and write to raw CSV.

    Pulls the full season schedule from nflverse, filters to rows where
    ``result`` is null (game not yet played), and writes to the raw upcoming
    schedule path.

    Args:
        season: The season year to fetch upcoming games for (e.g. ``2025``).
        repo: Absolute path to the repository root. Defaults to the value
            from ``get_settings()``.

    Returns:
        Absolute path to the written raw upcoming schedule CSV.
    """
    settings = get_settings()
    resolved_repo = repo or settings.repo_root

    logger.info("Fetching nflverse upcoming schedule for season %d", season)

    df: pd.DataFrame = nfl.import_schedules([season])

    # Upcoming games have no result yet
    upcoming = df.loc[df["result"].isna()].copy()

    logger.info(
        "Found %d upcoming games out of %d total for season %d",
        len(upcoming),
        len(df),
        season,
    )

    out_path = dataset_path(resolved_repo, "schedule_upcoming_raw_nflverse")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    upcoming.to_parquet(out_path, index=False)

    return out_path
