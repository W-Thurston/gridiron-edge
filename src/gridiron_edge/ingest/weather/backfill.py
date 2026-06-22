# src/gridiron_edge/ingest/weather/backfill.py

"""Historical weather backfill via OpenWeatherMap One Call API 3.0.

Fetches weather conditions at kickoff time for every completed NFL game
not already present in the weather_enriched dataset.  Designed to be run
repeatedly until all historical gaps are filled - safe to interrupt and
resume because it skips GAME_IDs already present in the output file.

Usage (via CLI):
    gridiron ingest weather-backfill --season-year 2024-2025
    gridiron ingest weather-backfill --all-years
    gridiron ingest weather-backfill --all-years --dry-run

API requirements:
    Requires an OpenWeatherMap One Call API 3.0 subscription (paid tier).
    Free accounts cannot access the timemachine endpoint for historical data.
    Set OWM_API_KEY environment variable or pass --api-key explicitly.

Rate limiting:
    One Call API 3.0 allows 1,000 calls/day on the base subscription.
    A full historical backfill (~7,000 games since 1999) therefore takes
    approximately 7 days of daily runs.  The --season-year flag lets you
    target specific seasons to spread the load.
    Within a single run, a 0.1s sleep between calls keeps well within
    the per-minute rate limit.

Output:
    Appends new rows to data/cleaned/weather_enriched.csv.
    Columns match the existing weekly-ingest schema:
        GAME_ID, TEMP, FEELS_LIKE, PRESSURE, HUMIDITY, DEW_POINT,
        CLOUDS, VISIBILITY, WIND_SPEED, WIND_DEG, WEATHER_MAIN,
        WEATHER_DESC
    Games that fail (API error, missing coordinates) are written to
    data/cleaned/weather_backfill_failed.csv for manual inspection.
"""

from __future__ import annotations

from datetime import datetime
import logging
from logging import Logger
from pathlib import Path
import time
from typing import Any, Final, Literal

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [untyped-import]
import pytz

# pyrefly: ignore [untyped-import]
import requests

# pyrefly: ignore [untyped-import]
from requests import Response

# pyrefly: ignore [untyped-import]
from requests.adapters import HTTPAdapter
import timezonefinder

# pyrefly: ignore [untyped-import]
from tqdm import tqdm
from urllib3.util.retry import Retry

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.metrics.travel.geo import to_decimal_degrees

logger: Logger = logging.getLogger(__name__)

# Seconds to sleep between API calls - keeps well within OWM rate limits
_CALL_SLEEP_S: Final[float] = 0.15

# OWM One Call 3.0 timemachine endpoint
_OWM_URL: Final[str] = (
    "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    "?lat={lat}&lon={lon}&dt={dt}&appid={appid}"
)

# Columns written to weather_enriched.csv
_OUTPUT_COLS: Final[list[str]] = [
    "GAME_ID",
    "TEMP",
    "FEELS_LIKE",
    "PRESSURE",
    "HUMIDITY",
    "DEW_POINT",
    "CLOUDS",
    "VISIBILITY",
    "WIND_SPEED",
    "WIND_DEG",
    "WEATHER_MAIN",
    "WEATHER_DESC",
]


# ---------------------------------------------------------------------------
# Core backfill function
# ---------------------------------------------------------------------------


def backfill_weather(
    *,
    season_year: str | None = None,
    owm_api_key: str,
    repo: Path | None = None,
    dry_run: bool = False,
    call_sleep: float = _CALL_SLEEP_S,
    max_calls: int | None = None,
) -> tuple[int, int]:
    """Fetch weather for all historical games not already in the archive.

    Loads the canonical games file, resolves stadium coordinates, identifies
    which GAME_IDs are already in weather_enriched.csv, and fetches the
    remainder from OWM.  Safe to run multiple times - already-fetched games
    are skipped automatically so this only fetches genuinely missing data.

    Args:
        season_year: If provided, only backfill games from this season
            (e.g. ``"2024-2025"``).  If None, all seasons are processed.
        owm_api_key: OpenWeatherMap API key with One Call 3.0 access.
        repo: Repository root.  Defaults to settings repo root.
        dry_run: If True, log what would be fetched but make no API calls
            and write nothing to disk.
        call_sleep: Seconds to sleep between API calls (default 0.15).
        max_calls: If provided, stop after this many API calls.  Use this
            to stay within the OWM daily limit (1,000 for the base
            subscription).  The run stops cleanly and logs how many games
            remain for the next run.  None means no limit.

    Returns:
        Tuple of (n_fetched, n_failed) counts.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    weather_path: Path = dataset_path(resolved_repo, "weather_enriched")
    failed_path: Path = resolved_repo / "data" / "cleaned" / "weather_backfill_failed.csv"

    # ── Load games + stadium coordinates ──────────────────────────────────
    games_df: DataFrame = pd.read_csv(dataset_path(resolved_repo, "games"))
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"], errors="coerce")

    stadiums_df: DataFrame = pd.read_csv(dataset_path(resolved_repo, "stadiums"))

    # Only completed games (WIN_OR_TIE is non-null for completed games)
    completed: DataFrame = games_df.loc[games_df["WIN_OR_TIE"].notna(), :].copy()

    if season_year is not None:
        completed = completed.loc[completed["YEAR"] == season_year, :].copy()
        if completed.empty:
            logger.warning("No completed games found for season %s", season_year)
            return 0, 0

    # Merge stadium coordinates - join on YEAR + STADIUM name
    # The stadium reference uses HOME_TEAM + YEAR to assign stadiums;
    # the games file already has the STADIUM name directly, so we join
    # by (YEAR, STADIUM) to get lat/lon.
    coords: DataFrame = stadiums_df.loc[
        :, ["YEAR", "STADIUM", "LATITUDE", "LONGITUDE"]
    ].drop_duplicates(subset=["YEAR", "STADIUM"])
    work: DataFrame = completed[["GAME_ID", "GAME_DATE", "GAMETIME", "YEAR", "STADIUM"]].merge(
        coords, on=["YEAR", "STADIUM"], how="left"
    )

    # Drop games where coordinates could not be resolved
    missing_coords: DataFrame = work.loc[work["LATITUDE"].isna(), :]
    if not missing_coords.empty:
        logger.warning(
            "%d games have no stadium coordinates and will be skipped: %s",
            len(missing_coords),
            missing_coords["GAME_ID"].tolist(),
        )
    work = work.loc[work["LATITUDE"].notna(), :].copy()

    # ── Identify already-fetched GAME_IDs ─────────────────────────────────
    already_fetched: set[str] = _load_existing_game_ids(weather_path)
    pending: DataFrame = work.loc[~work["GAME_ID"].isin(already_fetched), :].copy()

    logger.info(
        "Weather backfill: %d total games, %d already fetched, %d to fetch",
        len(work),
        len(already_fetched),
        len(pending),
    )

    if dry_run:
        logger.info("DRY RUN - no API calls will be made.")
        seasons: Series[int] = pending["YEAR"].value_counts().sort_index()
        for season, count in seasons.items():
            logger.info("  %s: %d games", season, count)
        if max_calls is not None:
            cap: int = min(max_calls, len(pending))
            logger.info(
                "With --max-calls %d: would fetch %d games this run, %d remaining after.",
                max_calls,
                cap,
                len(pending) - cap,
            )
        return 0, 0

    if pending.empty:
        logger.info("Nothing to fetch - all games already have weather data.")
        return 0, 0

    # Apply daily call cap - truncate pending to the first N games
    if max_calls is not None and len(pending) > max_calls:
        logger.info(
            "Capping this run at %d calls (%d games remain after). Run again tomorrow to continue.",
            max_calls,
            len(pending) - max_calls,
        )
        pending = pending.head(max_calls).copy()

    # ── Set up HTTP session with retry logic ──────────────────────────────
    sess = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
    # pyrefly: ignore [bad-argument-type]
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    tfinder = timezonefinder.TimezoneFinder()

    return _run_fetch_loop(
        pending=pending,
        owm_api_key=owm_api_key,
        session=sess,
        tf=tfinder,
        weather_path=weather_path,
        failed_path=failed_path,
        call_sleep=call_sleep,
    )


def _run_fetch_loop(
    *,
    pending: DataFrame,
    owm_api_key: str,
    session: requests.Session,
    tf: timezonefinder.TimezoneFinder,
    weather_path: Path,
    failed_path: Path,
    call_sleep: float,
) -> tuple[int, int]:
    """Execute the fetch loop over pending games and flush results to disk.

    Separated from ``backfill_weather`` to keep the main function within
    statement-count limits.

    Args:
        pending: DataFrame of games to fetch, with GAME_ID, YEAR,
            GAME_DATE, GAMETIME, LATITUDE, LONGITUDE, STADIUM columns.
        owm_api_key: OWM API key.
        session: Requests session with retry logic.
        tf: TimezoneFinder instance.
        weather_path: Path to weather_enriched.csv.
        failed_path: Path to failure log CSV.
        call_sleep: Seconds to sleep between API calls.

    Returns:
        Tuple of (n_fetched, n_failed).
    """
    fetched_rows: list[dict] = []
    failed_rows: list[dict] = []

    # Extract typed columns up front so Pyrefly can resolve concrete types.
    # itertuples() returns a union of all pandas scalar types per field,
    # which makes every field inaccessible to static analysis.
    # pyrefly: ignore [bad-assignment]
    game_ids: list[str] = pending["GAME_ID"].astype(str).tolist()

    game_dates: ndarray[tuple[Any, ...]] = np.array(
        # pyrefly: ignore [missing-attribute]
        pd.to_datetime(pending["GAME_DATE"]).dt.to_pydatetime().tolist()
    )
    # pyrefly: ignore [bad-assignment]
    gametimes: list[str] = pending["GAMETIME"].astype(str).tolist()
    # pyrefly: ignore [bad-assignment]
    latitudes: list[float] = pending["LATITUDE"].astype(float).tolist()
    # pyrefly: ignore [bad-assignment]
    longitudes: list[float] = pending["LONGITUDE"].astype(float).tolist()
    # pyrefly: ignore [bad-assignment]
    years: list[str] = pending["YEAR"].astype(str).tolist()
    # pyrefly: ignore [bad-assignment]
    stadiums: list[str] = pending["STADIUM"].astype(str).tolist()

    bar: tqdm[int] = tqdm(
        range(len(pending)),
        total=len(pending),
        desc="  weather backfill",
        unit="game",
        ncols=88,
        colour="cyan",
    )

    for i in bar:
        game_id: str = game_ids[i]
        game_date: datetime = game_dates[i]
        gametime: str = gametimes[i]
        latitude: float = latitudes[i]
        longitude: float = longitudes[i]
        year: str = years[i]
        stadium: str = stadiums[i]

        bar.set_postfix(season=year, game=game_id[-10:], refresh=False)

        result: dict | None = _fetch_one_game(
            game_id=game_id,
            game_date=game_date,
            gametime=gametime,
            latitude=latitude,
            longitude=longitude,
            owm_api_key=owm_api_key,
            session=session,
            tf=tf,
        )

        if result is not None:
            fetched_rows.append(result)
        else:
            failed_rows.append(
                {
                    "GAME_ID": game_id,
                    "YEAR": year,
                    "GAME_DATE": str(game_date),
                    "STADIUM": stadium,
                }
            )

        time.sleep(call_sleep)

        # Flush every 50 games so progress survives interruption
        if len(fetched_rows) >= 50:
            _append_to_weather_file(fetched_rows, weather_path)
            fetched_rows = []

    if fetched_rows:
        _append_to_weather_file(fetched_rows, weather_path)

    if failed_rows:
        _append_to_failed_file(failed_rows, failed_path)
        logger.warning(
            "%d games failed - see %s for details",
            len(failed_rows),
            failed_path,
        )

    n_fetched: int = len(pending) - len(failed_rows)
    logger.info("Weather backfill complete: %d fetched, %d failed", n_fetched, len(failed_rows))
    return n_fetched, len(failed_rows)


# ---------------------------------------------------------------------------
# Per-game fetch
# ---------------------------------------------------------------------------


def _fetch_one_game(
    *,
    game_id: str,
    game_date: datetime,
    gametime: str,
    latitude: float,
    longitude: float,
    owm_api_key: str,
    session: requests.Session,
    tf: timezonefinder.TimezoneFinder,
) -> dict | None:
    """Fetch weather for a single game from the OWM timemachine endpoint.

    Args:
        game_id: Canonical GAME_ID string (e.g. ``"2024_01_KC_LV"``).
        game_date: Game date as a datetime object.
        gametime: Kickoff time in HH:MM:SS format (24-hour).
        latitude: Stadium latitude in decimal degrees.
        longitude: Stadium longitude in decimal degrees.
        owm_api_key: OpenWeatherMap API key.
        session: Shared requests.Session with retry logic.
        tf: TimezoneFinder instance (reused across calls for efficiency).

    Returns:
        Dict with OWM weather fields keyed by column name, or None if the
        API call fails or the response is malformed.
    """
    try:
        lat: float = to_decimal_degrees(latitude)
        lon: float = to_decimal_degrees(longitude)

        # Resolve local timezone from coordinates
        tz_name: str | None = tf.certain_timezone_at(lat=lat, lng=lon)
        local_tz = pytz.timezone(tz_name if tz_name is not None else "UTC")

        # Build local kickoff datetime and convert to UTC Unix timestamp
        # GAMETIME is stored as HH:MM:SS in the canonical games schema
        date_str: str = game_date.strftime("%Y-%m-%d")
        naive: datetime = datetime.strptime(f"{date_str} {gametime}", "%Y-%m-%d %H:%M:%S")
        local_dt: datetime = local_tz.localize(naive, is_dst=None)
        utc_ts: int = int(local_dt.astimezone(pytz.utc).timestamp())

        # OWM One Call 3.0 timemachine request
        url: str = _OWM_URL.format(lat=lat, lon=lon, dt=utc_ts, appid=owm_api_key)
        resp: Response = session.get(url, timeout=15)
        resp.raise_for_status()
        owm: dict = resp.json()

        data: dict = owm["data"][0]
        weather_block: dict = data.get("weather", [{}])[0]

        return {
            "GAME_ID": game_id,
            "TEMP": data.get("temp"),
            "FEELS_LIKE": data.get("feels_like"),
            "PRESSURE": data.get("pressure"),
            "HUMIDITY": data.get("humidity"),
            "DEW_POINT": data.get("dew_point"),
            "CLOUDS": data.get("clouds"),
            "VISIBILITY": data.get("visibility"),
            "WIND_SPEED": data.get("wind_speed"),
            "WIND_DEG": data.get("wind_deg"),
            "WEATHER_MAIN": weather_block.get("main"),
            "WEATHER_DESC": weather_block.get("description"),
        }

    except requests.HTTPError as e:
        status: Literal["unknown"] | int = (
            e.response.status_code if e.response is not None else "unknown"
        )
        logger.warning("HTTP %s for game %s - skipping", status, game_id)
        return None
    except (KeyError, IndexError) as e:
        logger.warning("Malformed OWM response for game %s: %s", game_id, e)
        return None
    except Exception as e:
        logger.warning("Unexpected error for game %s: %s", game_id, e)
        return None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_existing_game_ids(weather_path: Path) -> set[str]:
    """Return the set of GAME_IDs already present in the weather archive."""
    if not weather_path.exists():
        return set()
    try:
        existing: DataFrame = pd.read_csv(weather_path, usecols=["GAME_ID"])
        return set(existing["GAME_ID"].dropna().tolist())
    except (pd.errors.EmptyDataError, KeyError):
        return set()


def _append_to_weather_file(rows: list[dict], path: Path) -> None:
    """Append fetched weather rows to the weather_enriched CSV.

    Creates the file with a header if it does not yet exist; appends
    without a header if it does.  Only writes the canonical output columns.
    """
    df: DataFrame = pd.DataFrame(rows)
    # Keep only the defined output columns, in order
    df = df.reindex(columns=_OUTPUT_COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header: bool = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)
    logger.debug("Flushed %d rows to %s", len(df), path)


def _append_to_failed_file(rows: list[dict], path: Path) -> None:
    """Append failed game records to the backfill failure log."""
    df: DataFrame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header: bool = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)
