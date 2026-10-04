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

from collections.abc import Hashable
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import Settings, get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.features.team.projected_starter import (
    UNAVAILABLE_PRIMARY,
    latest_snapshot_chart,
    ordered_candidates,
    prepare_snapshots,
    project_starter,
)
from gridiron_edge.ingest.nflverse.weekly_availability import load_depth_charts, load_injuries
from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    map_short_to_long,
    season_label,
)

logger: Logger = logging.getLogger(__name__)

BASE_COLUMNS: Final[tuple[str, ...]] = (
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

#: Projected starter for an unplayed game (``DECISIONS.md`` D75), with its source and
#: the depth chart snapshot time it came from. Null for every game without a projection.
PROJECTION_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_QB_ID_PROJECTED",
    "AWAY_QB_PROJECTION_SOURCE",
    "AWAY_QB_PROJECTION_AS_OF",
    "HOME_QB_ID_PROJECTED",
    "HOME_QB_PROJECTION_SOURCE",
    "HOME_QB_PROJECTION_AS_OF",
)

SIDECAR_COLUMNS: Final[tuple[str, ...]] = (*BASE_COLUMNS, *PROJECTION_COLUMNS)

PROJECTION_SOURCE: Final[str] = "depth_chart_report"

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
            **{column: Series(dtype="string") for column in PROJECTION_COLUMNS},
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
            **{
                column: Series(pd.NA, index=frame.index, dtype="string")
                for column in PROJECTION_COLUMNS
            },
        }
    ).loc[:, list(SIDECAR_COLUMNS)]


def project_unplayed_starters(
    upcoming: DataFrame,
    snapshots: DataFrame | None,
    injuries: DataFrame | None,
) -> DataFrame:
    """Project the starting QB for the current week's unplayed games.

    Only unplayed regular-season games in the earliest unplayed week are
    projected, and only when that week already has injury report rows and the
    team's latest depth chart snapshot lists a quarterback. Everything else is
    left null so the caller keeps today's behavior (``DECISIONS.md`` D65, D75).

    Args:
        upcoming: Raw upcoming schedule rows that are unplayed.
        snapshots: Timestamped depth chart (2025 format) for the season, or ``None``.
        injuries: Injury reports for the season, or ``None``.

    Returns:
        One row per ``game_id`` in ``upcoming`` with the projection columns
        (null when not projected), indexed like ``upcoming``.
    """
    cells: dict[str, dict[Hashable, str]] = {column: {} for column in PROJECTION_COLUMNS}

    def finish() -> DataFrame:
        return DataFrame(
            {
                "game_id": upcoming["game_id"].astype("string"),
                **{
                    column: Series(cells[column], dtype="string").reindex(upcoming.index)
                    for column in PROJECTION_COLUMNS
                },
            }
        )

    if snapshots is None or injuries is None or snapshots.empty or injuries.empty:
        return finish()
    if "dt" not in snapshots.columns:
        return finish()
    regular = upcoming[upcoming["game_type"] == "REG"]
    if regular.empty:
        return finish()
    season = int(regular["season"].min())
    week = int(regular.loc[regular["season"] == season, "week"].min())
    reports = injuries[
        (injuries["season"].astype(int) == season)
        & (injuries["week"].astype(int) == week)
        & (injuries["game_type"] == "REG")
    ]
    if reports.empty:
        return finish()

    statuses: dict[str, dict[str, str | None]] = {}
    for team, player, status in zip(
        reports["team"], reports["gsis_id"], reports["report_status"], strict=True
    ):
        if isinstance(player, str):
            statuses.setdefault(team, {})[player] = status
    by_team = prepare_snapshots(snapshots)

    in_week = regular.loc[(regular["season"] == season) & (regular["week"] == week)]
    for row_index, game in in_week.iterrows():
        for side in ("away", "home"):
            team = game[f"{side}_team"]
            if team not in by_team:
                continue
            chart, as_of = latest_snapshot_chart(by_team[team])
            if as_of is None or not chart:
                continue
            pick = project_starter(
                ordered_candidates(chart, statuses.get(team, {}), UNAVAILABLE_PRIMARY)
            )
            if pick is None:
                continue
            prefix = side.upper()
            cells[f"{prefix}_QB_ID_PROJECTED"][row_index] = pick
            cells[f"{prefix}_QB_PROJECTION_SOURCE"][row_index] = PROJECTION_SOURCE
            cells[f"{prefix}_QB_PROJECTION_AS_OF"][row_index] = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
    return finish()


def _load_projection_inputs(repo: Path, season: int) -> tuple[DataFrame | None, DataFrame | None]:
    """Load the cached depth chart and injury reports for a season, else ``None``."""
    try:
        snapshots = load_depth_charts(seasons=[season], repo=repo).get(season)
        injuries = load_injuries(seasons=[season], repo=repo).get(season)
    except FileNotFoundError:
        logger.warning("Depth chart or injury cache missing for %d; no starters projected", season)
        return None, None
    return snapshots, injuries


def build_qb_coach_sidecar(*, repo: Path | None = None) -> Path:
    """Build the canonical QB-and-coach identity sidecar Parquet.

    See :func:`build_qb_coach_sidecar_frame` for how the rows are combined.

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
    resolved_repo: Path = repo or get_settings().repo_root
    combined = build_qb_coach_sidecar_frame(repo=resolved_repo)
    out_path: Path = write_parquet(resolved_repo, "qb_coach_sidecar", combined)
    logger.info("Wrote %d QB/coach sidecar rows to %s", len(combined), out_path)
    return out_path


def build_qb_coach_sidecar_frame(*, repo: Path | None = None) -> DataFrame:
    """Combine the raw nflverse caches and the cached starter inputs into sidecar rows.

    Combines completed, non-preseason games from the raw historical
    nflverse schedule cache with unplayed games from the raw upcoming
    nflverse schedule cache (mirroring each source's own existing filter
    in ``games_nflverse.py``/``schedule_nflverse.py`` respectively), and
    projects starters for the unplayed games from the cached depth charts
    and injury reports. Nothing is written.
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
        if not upcoming.empty:
            snapshots, injuries = _load_projection_inputs(
                resolved_repo, int(upcoming["season"].min())
            )
            projection = project_unplayed_starters(upcoming, snapshots, injuries)
            for column in PROJECTION_COLUMNS:
                upcoming_rows[column] = projection[column].to_numpy()
            projected = int(projection["AWAY_QB_ID_PROJECTED"].notna().sum()) + int(
                projection["HOME_QB_ID_PROJECTED"].notna().sum()
            )
            logger.info("Projected %d starting quarterbacks for unplayed games", projected)
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
    return combined
