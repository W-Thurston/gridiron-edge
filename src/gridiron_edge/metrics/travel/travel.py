# src/gridiron_edge/metrics/travel/travel.py

"""Travel, timezone, and altitude features.

Computes per-game travel distance and timezone shift for TEAM_A.
The fast path vectorises distance using the haversine formula and caches
timezone lookups by unique coordinate pair - reducing the full-history
build from ~60s to under 1s.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

_KEY_COLS: Final[list[str]] = ["TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM"]
_GAMES_COLS: Final[list[str]] = ["YEAR", "WEEK_NUM", "WINNER", "LOSER", "STADIUM"]
_STADIUMS_TEAM_YEAR_COLS: Final[list[str]] = [
    "HOME_TEAM",
    "YEAR",
    "LATITUDE",
    "LONGITUDE",
    "ALTITUDE",
    "STADIUM",
]
_STADIUMS_BY_NAME_COLS: Final[list[str]] = [
    "STADIUM",
    "LATITUDE",
    "LONGITUDE",
    "ALTITUDE",
]

# Earth radius in km for haversine
_EARTH_RADIUS_KM: Final[float] = 6371.0

# Coordinate equality tolerance for is_true_home check. ~11m at equator.
# Smaller than any plausible inter-stadium distance.
_COORD_TOL_DEG: Final[float] = 1e-4


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance in kilometres.

    Equivalent to geopy_distance(...).km for typical Earth-surface distances
    but operates on entire arrays at once instead of one pair at a time.

    Args:
        lat1: Origin latitudes in decimal degrees.
        lon1: Origin longitudes in decimal degrees.
        lat2: Destination latitudes in decimal degrees.
        lon2: Destination longitudes in decimal degrees.

    Returns:
        Array of great-circle distances in kilometres.
    """
    r = _EARTH_RADIUS_KM
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _build_tz_cache(
    lats: pd.Series,
    lons: pd.Series,
    tz_finder: TimezoneFinder,
) -> dict[tuple[float, float], int]:
    """Build a (lat, lon) -> UTC-offset-hours cache for unique coordinate pairs.

    With ~30 distinct NFL stadium locations across the full history, this
    reduces TimezoneFinder calls from O(n_rows) to O(n_unique_locations).

    Args:
        lats: Series of latitude values (may contain duplicates).
        lons: Series of longitude values (may contain duplicates).
        tz_finder: Shared TimezoneFinder instance.

    Returns:
        Dict mapping (lat_rounded, lon_rounded) to integer UTC offset hours.
    """
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    cache: dict[tuple[float, float], int] = {}
    when = datetime.now(UTC)

    unique_pairs = set(zip(lats.round(4), lons.round(4), strict=False))
    for lat, lon in unique_pairs:
        if math.isnan(lat) or math.isnan(lon):
            continue
        tz_name = tz_finder.certain_timezone_at(lat=float(lat), lng=float(lon))
        if tz_name is None:
            cache[(lat, lon)] = 0
            continue
        local_dt = when.astimezone(ZoneInfo(tz_name))
        offset = local_dt.utcoffset()
        cache[(lat, lon)] = int(offset.total_seconds() // 3600) if offset else 0

    return cache


def add_travel_timezone_altitude(
    modeling: pd.DataFrame,
    games: pd.DataFrame,
    stadiums: pd.DataFrame,
) -> pd.DataFrame:
    """Add travel distance, timezone shift, and altitude features.

    Uses the ACTUAL game stadium coordinates for travel computation.
    Distance is computed via vectorised haversine; timezone offsets are
    cached by unique location to avoid repeated TimezoneFinder lookups.

    Required modeling columns:
        TEAM_A, TEAM_B, YEAR, WEEK_NUM, HOME_FIELD

    Required games columns:
        YEAR, WEEK_NUM, WINNER, LOSER, STADIUM

    Required stadiums columns:
        HOME_TEAM, YEAR, LATITUDE, LONGITUDE, ALTITUDE, STADIUM

    Output columns added:
        LATITUDE_A, LONGITUDE_A         TEAM_A home coords
        LATITUDE_SITE, LONGITUDE_SITE   Actual game stadium coords
        TEAM_A_KM_TRAVELED              Great-circle km from home to game
        TEAM_A_TZ_TRAVELED              Timezone hours difference
        ALTITUDE                        Altitude at game site (metres)

    Args:
        modeling: Base modeling DataFrame.
        games: Canonical games DataFrame.
        stadiums: Stadium reference DataFrame.

    Returns:
        Modeling DataFrame with travel features appended.
    """
    df: pd.DataFrame = modeling.copy()
    if "HOME_FIELD" not in df.columns:
        raise ValueError(
            "HOME_FIELD missing. Ensure home-field feature runs before travel features.",
        )

    base = df.loc[:, [*_KEY_COLS, "HOME_FIELD"]].copy()

    # --- Attach game stadium name ---
    g = games.loc[:, _GAMES_COLS].copy()
    g1 = g.rename(columns={"WINNER": "TEAM_A", "LOSER": "TEAM_B", "STADIUM": "STADIUM_GAME"})
    g2 = g.rename(columns={"WINNER": "TEAM_B", "LOSER": "TEAM_A", "STADIUM": "STADIUM_GAME"})

    base = base.merge(
        g1[["YEAR", "WEEK_NUM", "TEAM_A", "TEAM_B", "STADIUM_GAME"]],
        on=_KEY_COLS,
        how="left",
    )
    base = base.merge(
        g2[["YEAR", "WEEK_NUM", "TEAM_A", "TEAM_B", "STADIUM_GAME"]].rename(
            columns={"STADIUM_GAME": "STADIUM_GAME_REV"}
        ),
        on=_KEY_COLS,
        how="left",
    )
    base["STADIUM_GAME"] = base["STADIUM_GAME"].combine_first(base["STADIUM_GAME_REV"])
    base = base.drop(columns=["STADIUM_GAME_REV"])

    # --- Attach TEAM_A home coords ---
    st_team = stadiums.loc[:, _STADIUMS_TEAM_YEAR_COLS].copy()
    base = (
        base.merge(st_team, how="left", left_on=["TEAM_A", "YEAR"], right_on=["HOME_TEAM", "YEAR"])
        .drop(columns=["HOME_TEAM"])
        .rename(
            columns={
                "LATITUDE": "LATITUDE_A",
                "LONGITUDE": "LONGITUDE_A",
                "ALTITUDE": "ALTITUDE_A_HOME",
                "STADIUM": "STADIUM_A_HOME",
            }
        )
    )

    # --- Attach game site coords by stadium name ---
    st_by_name = stadiums.loc[:, _STADIUMS_BY_NAME_COLS].drop_duplicates(subset=["STADIUM"]).copy()
    base = base.merge(
        st_by_name.rename(
            columns={
                "LATITUDE": "LATITUDE_SITE",
                "LONGITUDE": "LONGITUDE_SITE",
                "ALTITUDE": "ALTITUDE_SITE",
            }
        ),
        how="left",
        left_on="STADIUM_GAME",
        right_on="STADIUM",
    ).drop(columns=["STADIUM"])

    # --- Attach TEAM_B home coords for fallback ---
    st_team_b = (
        stadiums.loc[:, _STADIUMS_TEAM_YEAR_COLS]
        .copy()
        .rename(
            columns={
                "LATITUDE": "LATITUDE_B_HOME",
                "LONGITUDE": "LONGITUDE_B_HOME",
                "ALTITUDE": "ALTITUDE_B_HOME",
                "STADIUM": "STADIUM_B_HOME",
            }
        )
    )
    base = base.merge(
        st_team_b, how="left", left_on=["TEAM_B", "YEAR"], right_on=["HOME_TEAM", "YEAR"]
    ).drop(columns=["HOME_TEAM"])

    # Fill missing site coords
    base["LATITUDE_SITE"] = base["LATITUDE_SITE"].where(
        base["LATITUDE_SITE"].notna(),
        pd.Series(
            np.where(base["HOME_FIELD"] == 1, base["LATITUDE_A"], base["LATITUDE_B_HOME"]),
            index=base.index,
        ),
    )
    base["LONGITUDE_SITE"] = base["LONGITUDE_SITE"].where(
        base["LONGITUDE_SITE"].notna(),
        pd.Series(
            np.where(base["HOME_FIELD"] == 1, base["LONGITUDE_A"], base["LONGITUDE_B_HOME"]),
            index=base.index,
        ),
    )
    base["ALTITUDE"] = base["ALTITUDE_SITE"].where(
        base["ALTITUDE_SITE"].notna(),
        pd.Series(
            np.where(base["HOME_FIELD"] == 1, base["ALTITUDE_A_HOME"], base["ALTITUDE_B_HOME"]),
            index=base.index,
        ),
    )

    # --- Determine travel mask ---
    # Compare on stadium coordinates rather than stadium names: name
    # spellings can drift between data sources (e.g. "Lambeau Field" vs
    # "Lambeau Field, Green Bay"), silently falsifying string-equality
    # comparisons even for actual home games. See audit_2026_06_18.md
    # travel/C1. Coordinates are sourced from the same stadiums.csv for
    # both home and game-site lookups, so coordinate equality is robust.

    lat_match = (base["LATITUDE_SITE"] - base["LATITUDE_A"]).abs() < _COORD_TOL_DEG
    lon_match = (base["LONGITUDE_SITE"] - base["LONGITUDE_A"]).abs() < _COORD_TOL_DEG

    is_true_home = (
        (base["HOME_FIELD"] == 1)
        & base["LATITUDE_SITE"].notna()
        & base["LATITUDE_A"].notna()
        & lat_match
        & lon_match
    )
    travel_mask = ~is_true_home

    # Validate coords for traveling rows
    missing = (
        base.loc[travel_mask, ["LATITUDE_A", "LONGITUDE_A", "LATITUDE_SITE", "LONGITUDE_SITE"]]
        .isna()
        .any(axis=1)
    )
    if missing.any():
        sample = base.loc[travel_mask].loc[missing, [*_KEY_COLS, "STADIUM_GAME"]].head(10)
        raise ValueError(
            "Missing coordinates for travel computation. "
            f"Check stadium coverage. Sample rows:\n{sample.to_string(index=False)}"
        )

    # --- Vectorised distance (haversine) ---
    base["TEAM_A_KM_TRAVELED"] = 0.0
    base["TEAM_A_TZ_TRAVELED"] = 0.0

    if travel_mask.any():
        traveling = base.loc[travel_mask]

        # Distance: fully vectorised
        km_vals = _haversine_km(
            traveling["LATITUDE_A"].to_numpy(dtype=float),
            traveling["LONGITUDE_A"].to_numpy(dtype=float),
            traveling["LATITUDE_SITE"].to_numpy(dtype=float),
            traveling["LONGITUDE_SITE"].to_numpy(dtype=float),
        )
        base.loc[travel_mask, "TEAM_A_KM_TRAVELED"] = km_vals

        # Timezone: cached by unique location
        tz_finder = TimezoneFinder()
        all_lats = pd.concat([traveling["LATITUDE_A"], traveling["LATITUDE_SITE"]])
        all_lons = pd.concat([traveling["LONGITUDE_A"], traveling["LONGITUDE_SITE"]])
        tz_cache = _build_tz_cache(all_lats, all_lons, tz_finder)

        tz_a = traveling.apply(
            lambda r: tz_cache.get(
                (round(float(r["LATITUDE_A"]), 4), round(float(r["LONGITUDE_A"]), 4)), 0
            ),
            axis=1,
        )
        tz_site = traveling.apply(
            lambda r: tz_cache.get(
                (round(float(r["LATITUDE_SITE"]), 4), round(float(r["LONGITUDE_SITE"]), 4)), 0
            ),
            axis=1,
        )
        base.loc[travel_mask, "TEAM_A_TZ_TRAVELED"] = (tz_a - tz_site).values

    # --- Merge features back ---
    feature_cols: list[str] = [
        "LATITUDE_A",
        "LONGITUDE_A",
        "LATITUDE_SITE",
        "LONGITUDE_SITE",
        "TEAM_A_KM_TRAVELED",
        "TEAM_A_TZ_TRAVELED",
        "ALTITUDE",
    ]
    feats = base.loc[:, _KEY_COLS + feature_cols].copy()
    out = df.merge(feats, how="left", on=_KEY_COLS)
    return out.drop_duplicates().reset_index(drop=True)
