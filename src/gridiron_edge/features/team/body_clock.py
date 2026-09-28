# src/gridiron_edge/features/team/body_clock.py

"""Canonical Away/Home body-clock kickoff-time features.

Converts the kickoff time into each team's *home* time zone, as a decimal
hour (a 1 pm ET kickoff is 10.0 for a Pacific-time team). Extends the
timezone-shift features: the same 1 pm ET game is a different body-clock
game for a West Coast team than for an East Coast team, and neutral-site
games keep each side's own home clock rather than the venue's.

nflverse ``gametime`` is Eastern time. The conversion is evaluated at the
game's own date, so it is deterministic and daylight-saving-correct
(Arizona does not observe DST); it never depends on the day the feature is
built.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Final
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame
from timezonefinder import TimezoneFinder

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)
from gridiron_edge.features.team.travel import _build_team_origin_lookup

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

_SCHEDULE_ZONE: Final[ZoneInfo] = ZoneInfo("America/New_York")
_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2}):(\d{2})(?::\d{2})?$")

_OUTPUT_COLUMNS: Final[tuple[str, str]] = (
    "AWAY_BODY_CLOCK_KICKOFF_HOUR",
    "HOME_BODY_CLOCK_KICKOFF_HOUR",
)
_INPUT_COLUMNS: Final[tuple[str, ...]] = ("GAME_ID", "YEAR", "AWAY_TEAM", "HOME_TEAM")


def _is_missing(value: object) -> bool:
    """Return whether a scalar schedule value is unavailable."""
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _kickoff_datetime(game_date: object, gametime: object) -> pd.Timestamp | None:
    """Return the Eastern-time kickoff instant, or None when unavailable."""
    if _is_missing(game_date) or _is_missing(gametime):
        return None
    match = _TIME_PATTERN.match(str(gametime).strip())
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    date = pd.to_datetime(str(game_date), errors="coerce")
    if pd.isna(date):
        return None
    return pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=_SCHEDULE_ZONE)


def _zone_name_lookup(origins: DataFrame) -> dict[tuple[str, str], str]:
    """Map (team, season) to the IANA zone of its home coordinates."""
    finder = TimezoneFinder()
    by_coordinates: dict[tuple[float, float], str | None] = {}
    zones: dict[tuple[str, str], str] = {}
    for team, year, latitude, longitude in zip(
        origins["HOME_TEAM"].astype(str),
        origins["YEAR"].astype(str),
        pd.to_numeric(origins["LATITUDE"], errors="coerce"),
        pd.to_numeric(origins["LONGITUDE"], errors="coerce"),
        strict=True,
    ):
        if pd.isna(latitude) or pd.isna(longitude):
            continue
        key = (round(float(latitude), 4), round(float(longitude), 4))
        if key not in by_coordinates:
            by_coordinates[key] = finder.certain_timezone_at(lat=key[0], lng=key[1])
        name = by_coordinates[key]
        if name is not None:
            zones[(team, year)] = name
    return zones


def _local_hour(
    kickoff: pd.Timestamp | None,
    zone_name: str | None,
) -> float:
    """Return kickoff as a decimal hour in ``zone_name``, NaN when unavailable."""
    if kickoff is None or zone_name is None:
        return float("nan")
    local = kickoff.tz_convert(ZoneInfo(zone_name))
    return local.hour + local.minute / 60.0


@FeatureRegistry.register("home_away_body_clock")
class HomeAwayBodyClockFeature:
    """Attach each side's kickoff hour on its own home clock."""

    spec = FeatureSpec(
        name="home_away_body_clock",
        produces=list(_OUTPUT_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach kickoff hours in the Away and Home teams' home time zones."""
        missing = sorted(set(_INPUT_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(
                "Home/away game frame is missing required columns: " + ", ".join(missing)
            )

        source = df.copy().drop(columns=list(_OUTPUT_COLUMNS), errors="ignore")

        schedule = build_game_metadata_lookup(
            historical=datasets.games(),
            upcoming=load_optional_upcoming_metadata(datasets),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "GAME_DATE": "GAME_DATE",
                "GAMETIME": "GAMETIME",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "game_date": "GAME_DATE",
                "game_time": "GAMETIME",
            },
        )
        kickoffs = {
            str(row.GAME_ID): _kickoff_datetime(row.GAME_DATE, row.GAMETIME)
            for row in schedule.itertuples(index=False)
        }
        zones = _zone_name_lookup(_build_team_origin_lookup(datasets.stadiums()))

        for side, column in (("AWAY", _OUTPUT_COLUMNS[0]), ("HOME", _OUTPUT_COLUMNS[1])):
            source[column] = [
                _local_hour(
                    kickoffs.get(str(game_id)),
                    zones.get((str(team), str(year))),
                )
                for game_id, team, year in zip(
                    source["GAME_ID"],
                    source[f"{side}_TEAM"],
                    source["YEAR"],
                    strict=True,
                )
            ]

        return source.reset_index(drop=True)
