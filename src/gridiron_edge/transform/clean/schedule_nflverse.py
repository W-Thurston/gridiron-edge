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

from datetime import UTC, datetime, timedelta
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import Settings, get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import (
    write_csv,
    write_parquet,
)
from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    gametime_to_hhmmss,
    map_short_to_long,
    season_label,
)

logger: Logger = logging.getLogger(__name__)


RICH_UPCOMING_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "game_day_of_week",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
    "location",
    "stadium",
    "roof",
    "surface",
    "divisional",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
    "source",
    "ingested_at",
)

ELO_UPCOMING_COLUMNS: Final[tuple[str, ...]] = (
    "WEEK_NUM",
    "GAME_DAY_OF_WEEK",
    "GAME_DATE",
    "AWAY_TEAM",
    "HOME_TEAM",
    "GAMETIME",
    "YEAR",
    "GAME_ID",
)

_RICH_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
)


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
        logger.debug("No stadium names found in upcoming schedule - skipping coverage check.")
        return

    reference_stadiums: set[str] = set(
        stadiums_df["STADIUM"].dropna().astype(str).str.strip().tolist()
    ) - {""}

    missing: set[str] = schedule_stadiums - reference_stadiums

    if not missing:
        logger.info(
            "Stadium coverage check passed - all %d upcoming stadiums are in the reference.",
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
        "Stadium coverage check FAILED for season %s - "
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


def _empty_rich_upcoming_schedule() -> DataFrame:
    """Return an empty rich schedule with stable column types."""
    return DataFrame(
        {
            "season": Series(dtype="string"),
            "week": Series(dtype="int64"),
            "game_id": Series(dtype="string"),
            "game_day_of_week": Series(dtype="string"),
            "game_date": Series(dtype="string"),
            "game_time": Series(dtype="string"),
            "away_team": Series(dtype="string"),
            "home_team": Series(dtype="string"),
            "neutral_site": Series(dtype="boolean"),
            "location": Series(dtype="string"),
            "stadium": Series(dtype="string"),
            "roof": Series(dtype="string"),
            "surface": Series(dtype="string"),
            "divisional": Series(dtype="Int64"),
            "away_rest": Series(dtype="Float64"),
            "home_rest": Series(dtype="Float64"),
            "away_moneyline": Series(dtype="Float64"),
            "home_moneyline": Series(dtype="Float64"),
            "spread_line": Series(dtype="Float64"),
            "away_spread_odds": Series(dtype="Float64"),
            "home_spread_odds": Series(dtype="Float64"),
            "total_line": Series(dtype="Float64"),
            "over_odds": Series(dtype="Float64"),
            "under_odds": Series(dtype="Float64"),
            "source": Series(dtype="string"),
            "ingested_at": Series(dtype="datetime64[ns, UTC]"),
        }
    ).loc[:, list(RICH_UPCOMING_COLUMNS)]


def _validate_ingested_at(
    value: datetime,
) -> None:
    """Require a timezone-aware UTC ingestion timestamp."""
    if value.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware UTC.")

    if value.utcoffset() != timedelta(0):
        raise ValueError("ingested_at must use UTC.")


def _optional_string(
    frame: DataFrame,
    column: str,
) -> Series:
    """Return one nullable string source column."""
    if column not in frame.columns:
        return Series(
            pd.NA,
            index=frame.index,
            dtype="string",
        )

    return frame[column].astype("string")


def _optional_numeric(
    frame: DataFrame,
    column: str,
) -> Series:
    """Return one nullable floating-point source column."""
    if column not in frame.columns:
        return Series(
            pd.NA,
            index=frame.index,
            dtype="Float64",
        )

    # pyrefly: ignore [missing-attribute]
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).astype("Float64")


def _optional_integer(
    frame: DataFrame,
    column: str,
) -> Series:
    """Return one nullable integer source column."""
    if column not in frame.columns:
        return Series(
            pd.NA,
            index=frame.index,
            dtype="Int64",
        )

    # pyrefly: ignore [missing-attribute]
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).astype("Int64")


def _resolve_week(
    row: Series,
) -> int:
    """Map postseason game types or return the source week."""
    game_type = str(row["game_type"])
    if game_type in GAME_TYPE_TO_WEEK:
        return GAME_TYPE_TO_WEEK[game_type]

    return int(row["week"])


def _map_team_names(
    values: Series,
) -> Series:
    """Map nflverse short codes to canonical long names."""
    source: Series[str] = values.astype("string")
    mapped: Series[str] = source.map(map_short_to_long)

    return mapped.where(
        mapped.notna(),
        source,
    ).astype("string")


def _neutral_site(
    frame: DataFrame,
) -> Series:
    """Derive nullable neutral-site state from source location."""
    if "location" not in frame.columns:
        return Series(
            pd.NA,
            index=frame.index,
            dtype="boolean",
        )

    location: Series[str] = frame["location"].astype("string")
    result: Series[bool] = location.str.casefold().eq("neutral")
    return result.astype("boolean")


def build_rich_upcoming_schedule(
    raw: DataFrame,
    *,
    ingested_at: datetime | None = None,
) -> DataFrame:
    """Build schedule-complete rich upcoming-game rows.

    Every unplayed source row produces one rich output row. Optional venue,
    context, rest, and market fields remain nullable and never determine
    whether a scheduled game survives the transform.

    Args:
        raw: Raw nflverse upcoming schedule rows.
        ingested_at: UTC timestamp for this local ingestion invocation.

    Returns:
        Rich, typed upcoming schedule rows in canonical column order.
    """
    timestamp: datetime = ingested_at or datetime.now(UTC)
    _validate_ingested_at(timestamp)

    required: set[str] = {
        "season",
        "week",
        "game_type",
        "game_id",
        "weekday",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
    }
    missing: list[str] = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Raw upcoming schedule is missing required columns: " + ", ".join(missing))

    frame: DataFrame = raw.copy()

    if "result" in frame.columns:
        frame = frame.loc[
            frame["result"].isna(),
            :,
        ].copy()

    if frame.empty:
        return _empty_rich_upcoming_schedule()

    week_values = frame.apply(
        _resolve_week,
        axis=1,
    ).astype(int)

    away_team: Series = _map_team_names(frame["away_team"])
    home_team: Series = _map_team_names(frame["home_team"])

    rich = DataFrame(
        {
            "season": (frame["season"].astype(int).map(season_label).astype("string")),
            "week": week_values,
            "game_id": frame["game_id"].astype("string"),
            "game_day_of_week": _optional_string(
                frame,
                "weekday",
            ),
            "game_date": _optional_string(
                frame,
                "gameday",
            ),
            "game_time": frame["gametime"].apply(gametime_to_hhmmss).astype("string"),
            "away_team": away_team,
            "home_team": home_team,
            "neutral_site": _neutral_site(frame),
            "location": _optional_string(
                frame,
                "location",
            ),
            "stadium": _optional_string(
                frame,
                "stadium",
            ),
            "roof": _optional_string(
                frame,
                "roof",
            ),
            "surface": _optional_string(
                frame,
                "surface",
            ),
            "divisional": _optional_integer(
                frame,
                "div_game",
            ),
            "away_rest": _optional_numeric(
                frame,
                "away_rest",
            ),
            "home_rest": _optional_numeric(
                frame,
                "home_rest",
            ),
            "away_moneyline": _optional_numeric(
                frame,
                "away_moneyline",
            ),
            "home_moneyline": _optional_numeric(
                frame,
                "home_moneyline",
            ),
            "spread_line": _optional_numeric(
                frame,
                "spread_line",
            ),
            "away_spread_odds": _optional_numeric(
                frame,
                "away_spread_odds",
            ),
            "home_spread_odds": _optional_numeric(
                frame,
                "home_spread_odds",
            ),
            "total_line": _optional_numeric(
                frame,
                "total_line",
            ),
            "over_odds": _optional_numeric(
                frame,
                "over_odds",
            ),
            "under_odds": _optional_numeric(
                frame,
                "under_odds",
            ),
            "source": Series(
                "nflverse",
                index=frame.index,
                dtype="string",
            ),
            "ingested_at": pd.to_datetime(
                Series(
                    timestamp,
                    index=frame.index,
                ),
                utc=True,
            ),
        }
    )

    return rich.loc[
        :,
        list(RICH_UPCOMING_COLUMNS),
    ].sort_values(
        [
            "week",
            "game_date",
            "game_time",
            "game_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def build_elo_upcoming_schedule(
    rich: DataFrame,
) -> DataFrame:
    """Project rich upcoming rows onto the focused Elo schedule schema."""
    missing: list[str] = sorted(
        {
            "season",
            "week",
            "game_id",
            "game_day_of_week",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        }
        - set(rich.columns)
    )
    if missing:
        raise ValueError(
            "Rich upcoming schedule is missing required columns: " + ", ".join(missing)
        )

    focused = DataFrame(
        {
            "WEEK_NUM": rich["week"].astype(int),
            "GAME_DAY_OF_WEEK": (rich["game_day_of_week"].fillna("").astype(str)),
            "GAME_DATE": (rich["game_date"].fillna("").astype(str)),
            "AWAY_TEAM": rich["away_team"].astype(str),
            "HOME_TEAM": rich["home_team"].astype(str),
            "GAMETIME": (rich["game_time"].fillna("").astype(str)),
            "YEAR": rich["season"].astype(str),
            "GAME_ID": rich["game_id"].astype(str),
        }
    )

    return focused.loc[
        :,
        list(ELO_UPCOMING_COLUMNS),
    ].reset_index(drop=True)


def clean_nflverse_upcoming(
    *,
    repo: Path | None = None,
    ingested_at: datetime | None = None,
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
        ingested_at: Optional timezone-aware UTC timestamp attached to all
            rich rows. Defaults to the current UTC time.

    Returns:
        Absolute path to the written canonical upcoming schedule CSV.
    """
    settings: Settings = get_settings()
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
    # The raw artifact is expected to be upcoming-only. Retain the
    # defensive filter for callers that provide a broader schedule.
    upcoming = df.loc[
        df["result"].isna(),
        :,
    ].copy()

    logger.info(
        "Processing %d upcoming games",
        len(upcoming),
    )

    if not upcoming.empty:
        stadiums_path: Path = dataset_path(
            resolved_repo,
            "stadiums",
        )
        if stadiums_path.exists():
            stadiums_df: DataFrame = pd.read_csv(stadiums_path)
            season_int = int(upcoming["season"].iloc[0])
            _check_stadium_coverage(
                upcoming,
                stadiums_df,
                season_label(season_int),
            )
        else:
            logger.warning(
                "Stadium reference file not found at %s - skipping coverage check.",
                stadiums_path,
            )

    rich: DataFrame = build_rich_upcoming_schedule(
        upcoming,
        ingested_at=ingested_at,
    )
    focused: DataFrame = build_elo_upcoming_schedule(rich)

    rich_path: Path = write_parquet(
        resolved_repo,
        "schedule_upcoming_rich",
        rich,
    )
    focused_path: Path = write_csv(
        resolved_repo,
        "schedule_upcoming",
        focused,
    )

    logger.info(
        "Wrote %d rich upcoming game rows to %s",
        len(rich),
        rich_path,
    )
    logger.info(
        "Wrote %d focused Elo schedule rows to %s",
        len(focused),
        focused_path,
    )

    return focused_path
