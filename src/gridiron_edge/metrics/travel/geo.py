# src/gridiron_edge/metrics/travel/geo.py

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import re
from re import Pattern
from typing import Literal
from zoneinfo import ZoneInfo

from geopy.distance import distance as geopy_distance
from timezonefinder import TimezoneFinder

LatLon = tuple[float, float]
Tude = str | float


_DMS_PATTERN: Pattern[str] = re.compile(r"[°\'″]")


def to_decimal_degrees(value: Tude) -> float:
    """Convert a latitude/longitude value into signed decimal degrees.

    Supports:
      - float: returned rounded to 4 decimals
      - str in DMS form, e.g. "39°44′21″N"
      - str in degrees+direction form, e.g. "104°59′W" or "104°W"
      - str in degrees+direction short form, e.g. "39°N"

    Returns:
      Signed decimal degrees, rounded to 4 decimals.

    Notes:
      - N/E => positive, S/W => negative
      - This mirrors the legacy conversion behavior but is now documented and typed.

    """  # noqa: RUF002
    if isinstance(value, float):
        return round(value, 4)

    if not isinstance(value, str):
        msg: str = f"Expected str or float for lat/lon, got {type(value)}"
        raise TypeError(msg)

    value = value.strip()

    # DMS form contains the double-prime character (″).
    if "″" in value:
        multiplier: Literal[-1, 1] = 1 if value[-1] in ("N", "E") else -1
        deg, minutes, seconds, _direction = re.split(_DMS_PATTERN, value)
        decimal: float = float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
        return round(multiplier * decimal, 4)

    # Degrees + direction form (e.g. "104°W", "39°N")
    if "°" in value:
        deg_str, direction = re.split(r"[°]", value)
        multiplier = 1 if direction in ("N", "E") else -1
        return round(multiplier * float(deg_str), 4)

    msg = f"Unrecognized lat/lon string format: {value!r}"
    raise ValueError(msg)


def measure_distance(
    home_lat_lon: Sequence[Tude],
    game_lat_lon: Sequence[Tude],
    *,
    metric: str = "km",
) -> float:
    """Measure great-circle distance between two points.

    Args:
      home_lat_lon: (lat, lon) for the team's home location
      game_lat_lon: (lat, lon) for the game location
      metric: "km" or "miles"

    Returns:
      Distance rounded to 6 decimals.

    Notes:
      - Uses geopy for distance calculations (as in legacy).
      - This function is pure (no I/O, no global state).

    """
    lat_h: float = to_decimal_degrees(home_lat_lon[0])
    lon_h: float = to_decimal_degrees(home_lat_lon[1])
    lat_g: float = to_decimal_degrees(game_lat_lon[0])
    lon_g: float = to_decimal_degrees(game_lat_lon[1])

    coords_home: LatLon = (lat_h, lon_h)
    coords_game: LatLon = (lat_g, lon_g)

    if metric == "km":
        return round(geopy_distance(coords_home, coords_game).km, 6)
    if metric == "miles":
        return round(geopy_distance(coords_home, coords_game).miles, 6)

    raise ValueError("metric must be 'km' or 'miles'")


def _utc_offset_hours(
    latitude: Tude,
    longitude: Tude,
    *,
    tz_finder: TimezoneFinder,
    when: datetime | None,
) -> int:
    """Compute the UTC offset (in hours) at a given lat/lon.

    Args:
      latitude: coordinates (str or float)
      longitude: coordinates (str or float)
      tz_finder: a TimezoneFinder instance (pass one in for performance)
      when: optional datetime to evaluate offset (defaults to "now" in UTC)

    Returns:
      Offset in hours, as an integer (e.g. -5, -7)

    Notes:
      - The legacy code used pytz + "now". We use ZoneInfo + UTC->local conversion.
      - DST can change offsets; using "now" matches legacy intent.

    """
    lat: float = to_decimal_degrees(latitude)
    lon: float = to_decimal_degrees(longitude)

    tz_name: str | None = tz_finder.certain_timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        msg: str = f"Could not determine timezone for lat={lat}, lon={lon}"
        raise ValueError(msg)

    when = when or datetime.now(UTC)
    local_dt: datetime = when.astimezone(ZoneInfo(tz_name))
    offset: timedelta | None = local_dt.utcoffset()
    if offset is None:
        msg = f"Could not determine UTC offset for timezone {tz_name}"
        raise ValueError(msg)

    # Convert seconds to whole hours (legacy effectively treated it as hours)
    return int(offset.total_seconds() // 3600)


def calculate_timezone_difference(
    lat_x: Tude,
    long_x: Tude,
    lat_y: Tude,
    long_y: Tude,
    *,
    tz_find: TimezoneFinder,
    when: datetime | None,
) -> int:
    """Compute timezone difference between two points.

    Returns:
      offset_x - offset_y (hours)

    This matches the legacy sign convention.

    """
    tz_x: int = _utc_offset_hours(lat_x, long_x, tz_finder=tz_find, when=when)
    tz_y: int = _utc_offset_hours(lat_y, long_y, tz_finder=tz_find, when=when)
    return tz_x - tz_y
