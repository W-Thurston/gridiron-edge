# src/gridiron_edge/transform/clean/qb_coach_sidecar.py

"""Build the QB-and-coach identity sidecar from raw nflverse schedule pulls.

Starting-QB identity (``home_qb_id``/``away_qb_id``) and head-coach identity
(``home_coach``/``away_coach``) are nflverse schedule fields with no home in
any existing cleaned dataset: ``games`` and ``schedule_upcoming_rich`` both
project their raw nflverse source down to a fixed column list that excludes
them. Per ``ROADMAP.md`` program rule 5 ("New inputs stay out of cleaned
games"), new schedule fields go in a separate sidecar dataset rather than
into ``NFL_wk_by_wk_cleaned.csv``.

This reads the same two raw nflverse caches ``games``/``schedule_upcoming_rich``
already read (``games_raw_nflverse``, ``schedule_upcoming_raw_nflverse``) and
projects the columns those clean steps intentionally drop. No new fetch
happens here - run the existing ``gridiron ingest nflverse-games``/
``gridiron ingest nflverse-upcoming`` commands first.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import Settings, get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    map_short_to_long,
    season_label,
)

logger: Logger = logging.getLogger(__name__)

SIDECAR_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_QB_ID",
    "AWAY_QB_NAME",
    "HOME_QB_ID",
    "HOME_QB_NAME",
    "AWAY_COACH",
    "HOME_COACH",
)

_REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_type",
    "game_id",
    "away_team",
    "home_team",
    "away_qb_id",
    "away_qb_name",
    "home_qb_id",
    "home_qb_name",
    "away_coach",
    "home_coach",
)


def _resolve_week(row: Series) -> int:
    """Map postseason game types to integer week numbers, else pass through."""
    game_type = str(row["game_type"])
    if game_type in GAME_TYPE_TO_WEEK:
        return GAME_TYPE_TO_WEEK[game_type]
    return int(row["week"])


def _empty_sidecar() -> DataFrame:
    """Return an empty sidecar with stable column types."""
    return DataFrame(
        {
            "GAME_ID": Series(dtype="string"),
            "YEAR": Series(dtype="string"),
            "WEEK_NUM": Series(dtype="int64"),
            "AWAY_TEAM": Series(dtype="string"),
            "HOME_TEAM": Series(dtype="string"),
            "AWAY_QB_ID": Series(dtype="string"),
            "AWAY_QB_NAME": Series(dtype="string"),
            "HOME_QB_ID": Series(dtype="string"),
            "HOME_QB_NAME": Series(dtype="string"),
            "AWAY_COACH": Series(dtype="string"),
            "HOME_COACH": Series(dtype="string"),
        }
    ).loc[:, list(SIDECAR_COLUMNS)]


def _build_sidecar_rows(raw: DataFrame) -> DataFrame:
    """Project one raw nflverse schedule slice to canonical sidecar rows."""
    missing = sorted(set(_REQUIRED_RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError("Raw nflverse schedule is missing required columns: " + ", ".join(missing))

    if raw.empty:
        return _empty_sidecar()

    frame = raw.copy()
    week_values = frame.apply(_resolve_week, axis=1).astype(int)

    return DataFrame(
        {
            "GAME_ID": frame["game_id"].astype("string"),
            "YEAR": frame["season"].astype(int).map(season_label).astype("string"),
            "WEEK_NUM": week_values,
            "AWAY_TEAM": frame["away_team"].astype(str).map(map_short_to_long).astype("string"),
            "HOME_TEAM": frame["home_team"].astype(str).map(map_short_to_long).astype("string"),
            "AWAY_QB_ID": frame["away_qb_id"].astype("string"),
            "AWAY_QB_NAME": frame["away_qb_name"].astype("string"),
            "HOME_QB_ID": frame["home_qb_id"].astype("string"),
            "HOME_QB_NAME": frame["home_qb_name"].astype("string"),
            "AWAY_COACH": frame["away_coach"].astype("string"),
            "HOME_COACH": frame["home_coach"].astype("string"),
        }
    ).loc[:, list(SIDECAR_COLUMNS)]


def build_qb_coach_sidecar(*, repo: Path | None = None) -> Path:
    """Build the canonical QB-and-coach identity sidecar Parquet.

    Combines completed, non-preseason games from the raw historical
    nflverse schedule cache with unplayed games from the raw upcoming
    nflverse schedule cache (mirroring each source's own existing filter
    in ``games_nflverse.py``/``schedule_nflverse.py`` respectively).

    Args:
        repo: Absolute path to the repository root. Defaults to the value
            from ``get_settings()``.

    Returns:
        Absolute path to the written sidecar Parquet.

    Raises:
        FileNotFoundError: If the historical raw schedule cache is
            missing. The upcoming raw cache is optional (offseason).
        ValueError: If the combined sidecar contains duplicate game IDs.
    """
    settings: Settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    games_raw_path: Path = dataset_path(resolved_repo, "games_raw_nflverse")
    if not games_raw_path.exists():
        raise FileNotFoundError(
            f"Raw nflverse games file not found: {games_raw_path}. "
            "Run `gridiron ingest nflverse-games` first."
        )

    logger.info("Reading raw nflverse games from %s", games_raw_path)
    games_raw: DataFrame = pd.read_parquet(games_raw_path)
    completed = games_raw.loc[
        games_raw["result"].notna() & (games_raw["game_type"] != "PRE"),
        :,
    ].copy()
    historical_rows = _build_sidecar_rows(completed)

    upcoming_raw_path: Path = dataset_path(resolved_repo, "schedule_upcoming_raw_nflverse")
    if upcoming_raw_path.exists():
        logger.info("Reading raw upcoming nflverse schedule from %s", upcoming_raw_path)
        upcoming_raw: DataFrame = pd.read_parquet(upcoming_raw_path)
        upcoming = upcoming_raw.loc[upcoming_raw["result"].isna(), :].copy()
        upcoming_rows = _build_sidecar_rows(upcoming)
    else:
        logger.warning(
            "Raw upcoming nflverse schedule not found at %s - sidecar will contain "
            "historical rows only.",
            upcoming_raw_path,
        )
        upcoming_rows = _empty_sidecar()

    combined: DataFrame = pd.concat(
        [historical_rows, upcoming_rows],
        ignore_index=True,
    )

    duplicated = combined["GAME_ID"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(combined.loc[duplicated, "GAME_ID"].astype(str).unique().tolist())
        raise ValueError(
            "QB/coach sidecar contains duplicate game IDs: " + ", ".join(duplicate_ids)
        )

    combined = combined.sort_values(
        ["YEAR", "WEEK_NUM", "GAME_ID"],
        kind="stable",
        ignore_index=True,
    ).loc[:, list(SIDECAR_COLUMNS)]

    out_path: Path = write_parquet(resolved_repo, "qb_coach_sidecar", combined)
    logger.info("Wrote %d QB/coach sidecar rows to %s", len(combined), out_path)
    return out_path
