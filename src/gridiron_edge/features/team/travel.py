# src/gridiron_edge/features/team/travel.py

"""Canonical Away/Home travel and game-site features.

Resolves the actual historical or upcoming game venue, attaches the
designated Away and Home franchise-season origins, and computes each
side's great-circle travel distance and timezone shift.

All outputs use one canonical game row:

    GAME_SITE_ALTITUDE
    AWAY_KM_TRAVELED
    HOME_KM_TRAVELED
    AWAY_TZ_SHIFT
    HOME_TZ_SHIFT

Missing venue or origin coordinates remain explicit null values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

_HOME_AWAY_TRAVEL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_HOME_AWAY_TRAVEL_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_SITE_ALTITUDE",
    "AWAY_KM_TRAVELED",
    "HOME_KM_TRAVELED",
    "AWAY_TZ_SHIFT",
    "HOME_TZ_SHIFT",
)

_SPECIAL_VENUE_HOME_TEAMS: Final[frozenset[str]] = frozenset(
    {
        "Alternate",
        "International",
    }
)
_EARTH_RADIUS_KM: Final[float] = 6371.0


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Return vectorized great-circle distances in kilometers."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    haversine = (
        np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))


def _build_tz_cache(
    latitudes: pd.Series,
    longitudes: pd.Series,
    timezone_finder: TimezoneFinder,
) -> dict[tuple[float, float], int]:
    """Cache UTC offsets for unique coordinate pairs."""
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    cache: dict[
        tuple[float, float],
        int,
    ] = {}
    when = datetime.now(UTC)

    coordinate_pairs = set(
        zip(
            latitudes.round(4),
            longitudes.round(4),
            strict=False,
        )
    )

    for latitude, longitude in coordinate_pairs:
        if math.isnan(latitude) or math.isnan(longitude):
            continue

        timezone_name = timezone_finder.certain_timezone_at(
            lat=float(latitude),
            lng=float(longitude),
        )

        if timezone_name is None:
            cache[
                (
                    latitude,
                    longitude,
                )
            ] = 0
            continue

        local_time = when.astimezone(ZoneInfo(timezone_name))
        offset = local_time.utcoffset()

        cache[
            (
                latitude,
                longitude,
            )
        ] = int(offset.total_seconds() // 3600) if offset else 0

    return cache


def _require_home_away_travel_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require canonical travel input columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _normalize_venue_rows(
    frame: pd.DataFrame,
    *,
    game_id_column: str,
    stadium_column: str,
) -> pd.DataFrame:
    """Normalize one game-to-stadium source."""
    if frame.empty:
        return pd.DataFrame(columns=["GAME_ID", "STADIUM_GAME"])

    _require_home_away_travel_columns(
        frame,
        (game_id_column, stadium_column),
        label="Venue source",
    )

    rows = frame.loc[:, [game_id_column, stadium_column]].rename(
        columns={
            game_id_column: "GAME_ID",
            stadium_column: "STADIUM_GAME",
        }
    )
    rows["GAME_ID"] = rows["GAME_ID"].astype(str).str.strip()
    rows["STADIUM_GAME"] = rows["STADIUM_GAME"].fillna("").astype(str).str.strip()

    return rows.loc[
        rows["GAME_ID"].ne("") & rows["STADIUM_GAME"].ne(""),
        :,
    ].drop_duplicates(ignore_index=True)


def _build_game_venue_lookup(
    historical_games: pd.DataFrame,
    upcoming_games: pd.DataFrame,
) -> pd.DataFrame:
    """Combine historical and upcoming game venue identity."""
    historical = _normalize_venue_rows(
        historical_games,
        game_id_column="GAME_ID",
        stadium_column="STADIUM",
    )
    upcoming = _normalize_venue_rows(
        upcoming_games,
        game_id_column="game_id",
        stadium_column="stadium",
    )

    venues = pd.concat(
        [historical, upcoming],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)

    if venues.empty:
        return venues

    conflicts = venues.groupby("GAME_ID")["STADIUM_GAME"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Game venue sources contain conflicting stadium identities.")

    return venues.drop_duplicates(
        subset=["GAME_ID"],
        keep="first",
        ignore_index=True,
    )


def _unique_coordinate_lookup(
    rows: pd.DataFrame,
    *,
    identity_columns: list[str],
    label: str,
) -> pd.DataFrame:
    """Require one coordinate tuple for each lookup identity."""
    coordinate_columns = [
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE",
    ]
    lookup = rows.loc[
        :,
        [*identity_columns, *coordinate_columns],
    ].drop_duplicates(ignore_index=True)

    counts = lookup.groupby(identity_columns, dropna=False).size()
    if counts.gt(1).any():
        raise ValueError(f"{label} contains conflicting coordinate identities.")

    return lookup.drop_duplicates(
        subset=identity_columns,
        keep="first",
        ignore_index=True,
    )


def _build_site_lookup(stadiums: pd.DataFrame) -> pd.DataFrame:
    """Build actual stadium-name to site-coordinate lookup."""
    required = (
        "STADIUM",
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE",
    )
    _require_home_away_travel_columns(
        stadiums,
        required,
        label="Stadium reference",
    )

    rows = stadiums.loc[:, list(required)].copy()
    rows["STADIUM"] = rows["STADIUM"].astype(str).str.strip()
    rows = rows.loc[rows["STADIUM"].ne(""), :]

    lookup = _unique_coordinate_lookup(
        rows,
        identity_columns=["STADIUM"],
        label="Stadium reference",
    )
    return lookup.rename(
        columns={
            "STADIUM": "STADIUM_GAME",
            "LATITUDE": "_SITE_LATITUDE",
            "LONGITUDE": "_SITE_LONGITUDE",
            "ALTITUDE": "GAME_SITE_ALTITUDE",
        }
    )


def _build_team_origin_lookup(stadiums: pd.DataFrame) -> pd.DataFrame:
    """Build franchise-season home-coordinate lookup."""
    required = (
        "HOME_TEAM",
        "YEAR",
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE",
    )
    _require_home_away_travel_columns(
        stadiums,
        required,
        label="Stadium reference",
    )

    rows = stadiums.loc[:, list(required)].copy()
    rows["HOME_TEAM"] = rows["HOME_TEAM"].astype(str).str.strip()
    rows = rows.loc[
        ~rows["HOME_TEAM"].isin(_SPECIAL_VENUE_HOME_TEAMS),
        :,
    ]

    return _unique_coordinate_lookup(
        rows,
        identity_columns=["HOME_TEAM", "YEAR"],
        label="Franchise-season stadium reference",
    )


def _attach_side_origin(
    frame: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    """Attach one side's franchise-season origin coordinates."""
    team_column = f"{side}_TEAM"
    renamed = origins.rename(
        columns={
            "HOME_TEAM": team_column,
            "LATITUDE": f"_{side}_LATITUDE",
            "LONGITUDE": f"_{side}_LONGITUDE",
            "ALTITUDE": f"_{side}_HOME_ALTITUDE",
        }
    )
    return frame.merge(
        renamed,
        how="left",
        on=[team_column, "YEAR"],
        sort=False,
        validate="many_to_one",
    )


def _attach_travel_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate Away and Home distance and timezone shift."""
    result = frame.copy()
    tz_finder = TimezoneFinder()

    coordinate_series = [
        result["_AWAY_LATITUDE"],
        result["_AWAY_LONGITUDE"],
        result["_HOME_LATITUDE"],
        result["_HOME_LONGITUDE"],
        result["_SITE_LATITUDE"],
        result["_SITE_LONGITUDE"],
    ]
    all_lats = pd.concat(
        [coordinate_series[0], coordinate_series[2], coordinate_series[4]],
        ignore_index=True,
    )
    all_lons = pd.concat(
        [coordinate_series[1], coordinate_series[3], coordinate_series[5]],
        ignore_index=True,
    )
    tz_cache = _build_tz_cache(all_lats, all_lons, tz_finder)

    for side in ("AWAY", "HOME"):
        origin_lat = result[f"_{side}_LATITUDE"]
        origin_lon = result[f"_{side}_LONGITUDE"]
        site_lat = result["_SITE_LATITUDE"]
        site_lon = result["_SITE_LONGITUDE"]
        available = origin_lat.notna() & origin_lon.notna() & site_lat.notna() & site_lon.notna()

        distance_column = f"{side}_KM_TRAVELED"
        result[distance_column] = float("nan")
        if available.any():
            result.loc[available, distance_column] = _haversine_km(
                origin_lat.loc[available].to_numpy(dtype=float),
                origin_lon.loc[available].to_numpy(dtype=float),
                site_lat.loc[available].to_numpy(dtype=float),
                site_lon.loc[available].to_numpy(dtype=float),
            )

        shift_values: list[object] = []
        for origin_latitude, origin_longitude, site_latitude, site_longitude in zip(
            origin_lat,
            origin_lon,
            site_lat,
            site_lon,
            strict=True,
        ):
            if any(
                pd.isna(value)
                for value in (
                    origin_latitude,
                    origin_longitude,
                    site_latitude,
                    site_longitude,
                )
            ):
                shift_values.append(pd.NA)
                continue

            origin_key = (
                round(float(origin_latitude), 4),
                round(float(origin_longitude), 4),
            )
            site_key = (
                round(float(site_latitude), 4),
                round(float(site_longitude), 4),
            )
            shift_values.append(tz_cache.get(origin_key, 0) - tz_cache.get(site_key, 0))

        result[f"{side}_TZ_SHIFT"] = pd.Series(
            shift_values,
            index=result.index,
            dtype="Int64",
        )

    return result


@FeatureRegistry.register("home_away_travel")
class HomeAwayTravelFeature:
    """Compute canonical Away and Home travel to the actual game site."""

    spec = FeatureSpec(
        name="home_away_travel",
        produces=list(_HOME_AWAY_TRAVEL_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach schedule-complete travel and site-altitude features."""
        _require_home_away_travel_columns(
            df,
            _HOME_AWAY_TRAVEL_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=list(_HOME_AWAY_TRAVEL_COLUMNS),
            errors="ignore",
        )
        source["_TRAVEL_INPUT_ORDER"] = range(len(source))

        try:
            upcoming = datasets.schedule_upcoming_rich()
        except FileNotFoundError:
            upcoming = pd.DataFrame()

        venues = _build_game_venue_lookup(
            datasets.games(),
            upcoming,
        )
        stadiums = datasets.stadiums()
        sites = _build_site_lookup(stadiums)
        origins = _build_team_origin_lookup(stadiums)

        result = source.merge(
            venues,
            how="left",
            on="GAME_ID",
            sort=False,
            validate="many_to_one",
        )
        result = result.merge(
            sites,
            how="left",
            on="STADIUM_GAME",
            sort=False,
            validate="many_to_one",
        )
        result = _attach_side_origin(
            result,
            origins,
            side="AWAY",
        )
        result = _attach_side_origin(
            result,
            origins,
            side="HOME",
        )
        result = _attach_travel_values(result)

        return (
            result.sort_values(
                "_TRAVEL_INPUT_ORDER",
                kind="stable",
            )
            .loc[
                :,
                [
                    *source.columns.drop("_TRAVEL_INPUT_ORDER").tolist(),
                    *_HOME_AWAY_TRAVEL_COLUMNS,
                ],
            ]
            .reset_index(drop=True)
        )
