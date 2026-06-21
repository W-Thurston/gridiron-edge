# src/gridiron_edge/ingest/weather/openweather.py

"""OpenWeatherMap historical weather ingestion.

Enriches the cleaned games dataset with weather conditions at game time by
calling the OWM One Call API timemachine endpoint for each game in the most
recently completed NFL week.

Idempotency:
    The fetch loop is guarded against re-fetching games already present
    in ``weather_enriched``. See the ``# Idempotency check`` block in
    :func:`fetch_weather`. The append path therefore relies on these
    guards rather than performing an explicit dedup at write time
    (see ``weather/H1`` from audit_2026_06_18.md, which is closed as
    no-longer-applicable).
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [untyped-import]
import requests

# pyrefly: ignore [untyped-import]
from requests.adapters import HTTPAdapter
import timezonefinder

# pyrefly: ignore [untyped-import]
from tqdm import tqdm
from urllib3.util.retry import Retry

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.metrics.travel.geo import to_decimal_degrees

tqdm.pandas()  # Registers DataFrame.progress_apply

logger = logging.getLogger(__name__)


def _convert_12hour_to_24hour(time_str: str) -> str:
    """Normalize a game time string to 24-hour ``HH:MM:SS`` format.

    Accepts either 12-hour input (``"6:30PM"`` or ``"6:30 PM"``) or
    24-hour input (``"18:30:00"`` or ``"18:30"``). The cleaned games
    file stores times in 24-hour format with seconds, so the common
    case is a pass-through.

    Args:
        time_str: Game time string in any of the supported formats.

    Returns:
        Time string in ``"%H:%M:%S"`` format.

    Raises:
        ValueError: If the input does not match any known format.
    """
    # Try common formats in order. If the string already matches HH:MM:SS
    # we still parse-and-reformat it to validate it's well-formed.
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M%p", "%I:%M %p", "%I:%M:%S%p", "%I:%M:%S %p"):
        try:
            in_time = datetime.strptime(time_str.strip(), fmt)
            return datetime.strftime(in_time, "%H:%M:%S")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized time format: {time_str!r}")


def _enrich_row(
    row: pd.Series,
    tf: timezonefinder.TimezoneFinder,
    session: requests.Session,
    owm_api_key: str,
) -> pd.Series:
    """Pull weather data for a single game row from OpenWeatherMap.

    Converts the game's location and time to a UTC Unix timestamp,
    calls the OWM One Call API timemachine endpoint, and appends
    weather fields to the row. Returns the original row unchanged on
    API failure.

    Args:
        row: A pandas Series representing one game, containing at minimum
            ``LATITUDE``, ``LONGITUDE``, ``GAME_DATE``, and ``GAMETIME``.
        tf: A ``TimezoneFinder`` instance for timezone resolution.
        session: A ``requests.Session`` configured with retry logic.
        owm_api_key: OpenWeatherMap API key.

    Returns:
        The input ``row`` with weather fields appended, or the original
        row on API failure.
    """
    try:
        lat: float = to_decimal_degrees(row.LATITUDE)
        lon: float = to_decimal_degrees(row.LONGITUDE)

        tz_name: str = tf.certain_timezone_at(lat=lat, lng=lon) or "UTC"
        time_str: str = _convert_12hour_to_24hour(row.GAMETIME)
        game_time: datetime = datetime.combine(
            row.GAME_DATE.date(),
            datetime.strptime(time_str, "%H:%M:%S").time(),
            tzinfo=ZoneInfo(tz_name),
        )
        utc_timestamp: int = int(game_time.timestamp())

        url: str = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
            f"?lat={lat}&lon={lon}&dt={utc_timestamp}&appid={owm_api_key}"
        )
        owm_response = session.get(url).json()
        data: dict = owm_response["data"][0]

        row["TEMP"] = data.get("temp")
        row["FEELS_LIKE"] = data.get("feels_like")
        row["PRESSURE"] = data.get("pressure")
        row["HUMIDITY"] = data.get("humidity")
        row["DEW_POINT"] = data.get("dew_point")
        row["CLOUDS"] = data.get("clouds")
        row["VISIBILITY"] = data.get("visibility")
        row["WIND_SPEED"] = data.get("wind_speed")
        row["WIND_DEG"] = data.get("wind_deg")
        row["WEATHER_MAIN"] = data["weather"][0].get("main")
        row["WEATHER_DESC"] = data["weather"][0].get("description")

        return row

    except requests.RequestException as exc:
        logger.warning("Weather API request failed for %s: %s", row.GAME_ID, exc, exc_info=True)
        return row
    except (KeyError, IndexError) as exc:
        logger.warning("Unexpected OWM response schema for %s: %s", row.GAME_ID, exc, exc_info=True)
        return row


def fetch_weather(*, season_year: str, owm_api_key: str, repo: Path | None = None) -> None:
    """Pull historical weather for the most recently completed week in a season.

    Reads the cleaned games and stadium datasets, resolves stadium coordinates,
    and calls the OpenWeatherMap API for each game in the most recently
    completed season week. Appends results to the weather-enriched dataset.

    Idempotent: if weather data already exists for every game in the target
    week, the function logs a message and returns without making any API calls.

    Args:
        season_year: NFL season label (e.g. ``"2025-2026"``).
        owm_api_key: OpenWeatherMap API key.
        repo: Repository root path. Defaults to ``get_settings().repo_root``.
    """
    resolved_repo: Path = repo or get_settings().repo_root

    df: DataFrame = pd.read_csv(dataset_path(resolved_repo, "games"))
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values(["GAME_DATE", "GAMETIME", "GAME_ID"], ascending=True, inplace=True)

    df_stadium: DataFrame = pd.read_csv(dataset_path(resolved_repo, "stadiums"))

    temp_df = df.loc[:, ["GAME_ID", "GAME_DATE", "GAMETIME", "YEAR", "STADIUM"]].copy()
    temp_df = temp_df.merge(
        df_stadium.loc[:, ["YEAR", "STADIUM", "LATITUDE", "LONGITUDE"]],
        how="left",
        on=["YEAR", "STADIUM"],
    ).drop_duplicates()
    temp_df.sort_values(
        ["GAME_DATE", "GAMETIME", "GAME_ID"],
        ascending=True,
        inplace=True,
        ignore_index=True,
    )

    # Reduce to just the most recently completed week for the requested season
    latest_week: int = df.loc[df["YEAR"] == season_year, :].iloc[-1]["WEEK_NUM"]
    week_mask = (df["YEAR"] == season_year) & (df["WEEK_NUM"] == latest_week)
    temp_df = temp_df.iloc[df.loc[week_mask, :].index, :]

    # Idempotency check — skip if all games for this week are already enriched
    weather_path = dataset_path(resolved_repo, "weather_enriched")
    if weather_path.exists():
        existing: DataFrame = pd.read_csv(weather_path, usecols=["GAME_ID"])
        already_fetched: set[str] = set(existing["GAME_ID"].astype(str))
        target_game_ids: set[str] = set(temp_df["GAME_ID"].astype(str))
        if target_game_ids.issubset(already_fetched):
            logger.info(
                "Weather already exists for all %d games in %s week %d — skipping API calls.",
                len(temp_df),
                season_year,
                latest_week,
            )
            return
        # Only fetch games not yet in the enriched file
        temp_df = temp_df.loc[~temp_df["GAME_ID"].astype(str).isin(already_fetched)].copy()
        logger.info(
            "%d of %d games in %s week %d already have weather — fetching %d new.",
            len(target_game_ids) - len(temp_df),
            len(target_game_ids),
            season_year,
            latest_week,
            len(temp_df),
        )

    tzf = timezonefinder.TimezoneFinder()
    sess = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    # pyrefly: ignore [bad-argument-type]
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    logger.info(
        "Fetching weather for %s week %d (%d games)", season_year, latest_week, len(temp_df)
    )

    temp_df = temp_df.progress_apply(  # type: ignore[attr-defined]
        lambda x: _enrich_row(row=x, tf=tzf, session=sess, owm_api_key=owm_api_key),
        axis=1,
    )

    temp_df.drop(
        ["GAME_DATE", "GAMETIME", "YEAR", "STADIUM", "LATITUDE", "LONGITUDE"],
        axis=1,
    ).to_csv(weather_path, mode="a", index=False, header=False)

    logger.info("Weather data appended to %s", weather_path)
