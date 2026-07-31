# src/gridiron_edge/features/team/travel.py

"""Travel, timezone shift, and neutral-site features.

This module replaces the previous travel feature with an extended version
that adds three new columns alongside the existing travel distance and
timezone features:

    Existing (preserved, names unchanged):
        LATITUDE_A          float   TEAM_A home latitude
        LONGITUDE_A         float   TEAM_A home longitude
        LATITUDE_SITE       float   Game stadium latitude
        LONGITUDE_SITE      float   Game stadium longitude
        TEAM_A_KM_TRAVELED  float   Great-circle distance from TEAM_A home to game site
        TEAM_A_TZ_TRAVELED  float   Signed timezone-hour difference for TEAM_A
        ALTITUDE            float   Game site altitude (metres)

    Extended features:
        TEAM_B_KM_TRAVELED  float   Great-circle distance from TEAM_B home to game site.
                                    Symmetric counterpart to TEAM_A_KM_TRAVELED.
                                    In the two-row design, each row already has the
                                    correct TEAM_A perspective, but the model benefits
                                    from knowing the opponent's travel burden too.

        TEAM_A_TZ_SHIFT     int     Signed whole-hour timezone shift for TEAM_A:
                                    negative = eastward travel (circadian advantage),
                                    positive = westward (circadian cost).
                                    Same direction as TEAM_A_TZ_TRAVELED but stored
                                    as an integer for cleaner model interpretation.
                                    Aliased from TEAM_A_TZ_TRAVELED (already computed).

        TEAM_B_TZ_SHIFT     int     Signed whole-hour timezone shift for TEAM_B.
                                    Requires computing TEAM_B home → game site distance
                                    using the same geo helpers as TEAM_A.

        IS_NEUTRAL_SITE     int     1 if neither team is at their home stadium.
                                    Identified by GAME_LOCATION == "N" in the games
                                    schema.  Covers London, Mexico City, São Paulo, and
                                    other international/neutral venues.
                                    When IS_NEUTRAL_SITE = 1, HOME_FIELD = 0 for both
                                    perspectives and travel is non-zero for both teams.

Design notes:
    - TEAM_A_TZ_SHIFT and TEAM_B_TZ_SHIFT are integer-rounded versions of
      the float timezone offsets.  Rounding to whole hours is appropriate
      since US timezones are all whole-hour offsets from UTC (no 30-minute
      offsets like India or some Australian zones).
    - The circadian asymmetry is east-vs-west, not absolute distance.
      A team flying from LA to NY crosses 3 timezone hours eastward
      (conventionally advantageous for evening/night games); NY to LA is
      westward (circadian cost).  See Recht et al. (1995) for the
      original sports-circadian literature.
    - IS_NEUTRAL_SITE is independent of travel - a London game has
      IS_NEUTRAL_SITE=1 for both teams, but TEAM_A_KM_TRAVELED may be
      small (east-coast team) or large (west-coast team).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from timezonefinder import TimezoneFinder

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.metrics.travel import add_travel_timezone_altitude
from gridiron_edge.metrics.travel.geo import (
    calculate_timezone_difference,
    measure_distance,
)
from gridiron_edge.metrics.travel.travel import _build_tz_cache, _haversine_km

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

_KEY_COLS: Final[list[str]] = ["TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM"]
_GAMES_COLS: Final[list[str]] = ["YEAR", "WEEK_NUM", "WINNER", "LOSER", "STADIUM", "GAME_LOCATION"]
_STADIUMS_TEAM_YEAR_COLS: Final[list[str]] = [
    "HOME_TEAM",
    "YEAR",
    "LATITUDE",
    "LONGITUDE",
    "ALTITUDE",
    "STADIUM",
]
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


@FeatureRegistry.register("travel")
class TravelFeature:
    """Extended travel, timezone, and neutral-site features.

    Wraps ``add_travel_timezone_altitude`` to produce TEAM_A travel metrics,
    then adds TEAM_B travel/timezone columns and the IS_NEUTRAL_SITE flag.

    Dependencies (run order in pipeline.py):
        home_field must run before travel (travel reads HOME_FIELD).
        travel must run before any feature that uses travel columns.
    """

    spec = FeatureSpec(
        name="travel",
        produces=[
            "LATITUDE_A",
            "LONGITUDE_A",
            "LATITUDE_SITE",
            "LONGITUDE_SITE",
            "TEAM_A_KM_TRAVELED",
            "TEAM_A_TZ_TRAVELED",
            "ALTITUDE",
            "TEAM_B_KM_TRAVELED",
            "TEAM_A_TZ_SHIFT",
            "TEAM_B_TZ_SHIFT",
            "IS_NEUTRAL_SITE",
        ],
        depends_on=("home_field",),
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute travel and venue features and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame; must contain HOME_FIELD (from the
                home_field feature which runs first in the pipeline).
            datasets: Provides ``games()`` and ``stadiums()``.

        Returns:
            Input DataFrame with eleven travel/venue columns appended.
        """
        games = datasets.games()
        stadiums = datasets.stadiums()

        # ── Existing TEAM_A travel (unchanged logic) ──────────────────────
        df = add_travel_timezone_altitude(df, games, stadiums)

        # ── TEAM_B travel and timezone shift ─────────────────────────────
        df = self._add_team_b_travel(df, games, stadiums)

        # ── Integer timezone shifts (round the float offsets) ─────────────
        df["TEAM_A_TZ_SHIFT"] = df["TEAM_A_TZ_TRAVELED"].round().astype("Int64")
        df["TEAM_B_TZ_SHIFT"] = df["_TEAM_B_TZ_TRAVELED"].round().astype("Int64")
        df = df.drop(columns=["_TEAM_B_TZ_TRAVELED"], errors="ignore")

        # ── Neutral site flag ─────────────────────────────────────────────
        df = self._add_neutral_site(df, games)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_team_b_travel(
        self,
        df: pd.DataFrame,
        games: pd.DataFrame,
        stadiums: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute TEAM_B's km travelled and timezone shift to the game site.

        Mirrors the TEAM_A computation in add_travel_timezone_altitude but
        for the opposing team.  Attaches _TEAM_B_TZ_TRAVELED (float) and
        TEAM_B_KM_TRAVELED to df.

        Args:
            df: Modeling DataFrame already containing LATITUDE_SITE,
                LONGITUDE_SITE from the TEAM_A travel computation.
            games: Cleaned games DataFrame.
            stadiums: Stadium reference DataFrame.

        Returns:
            df with TEAM_B_KM_TRAVELED and _TEAM_B_TZ_TRAVELED appended.
        """
        # Attach TEAM_B home coordinates
        st_team = stadiums[_STADIUMS_TEAM_YEAR_COLS].copy()
        base = df.copy()
        base = base.merge(
            st_team.rename(
                columns={
                    "HOME_TEAM": "TEAM_B",
                    "LATITUDE": "_LAT_B_HOME",
                    "LONGITUDE": "_LON_B_HOME",
                    "ALTITUDE": "_ALT_B_HOME",
                    "STADIUM": "_STADIUM_B_HOME",
                }
            ),
            how="left",
            on=["TEAM_B", "YEAR"],
        )

        # Where game site coords exist, compute B's travel distance
        tz_finder = TimezoneFinder()
        km_b: list[float] = []
        tz_b: list[float] = []

        site_lat = base["LATITUDE_SITE"]
        site_lon = base["LONGITUDE_SITE"]
        home_lat = base["_LAT_B_HOME"]
        home_lon = base["_LON_B_HOME"]

        for i in range(len(base)):
            lat_b = home_lat.iat[i]
            lon_b = home_lon.iat[i]
            lat_s = site_lat.iat[i]
            lon_s = site_lon.iat[i]

            if any(pd.isna(v) for v in (lat_b, lon_b, lat_s, lon_s)):
                km_b.append(float("nan"))
                tz_b.append(float("nan"))
                continue

            # Zero travel if TEAM_B is the home team playing at their own stadium
            is_team_b_home = base["HOME_FIELD"].iat[i] == 0  # HOME_FIELD=1 means TEAM_A is home
            if is_team_b_home and base.get("_STADIUM_B_HOME") is not None:
                stadium_game = base.get("STADIUM_GAME", pd.Series(dtype=str))
                if i < len(stadium_game):
                    b_home_stadium = base["_STADIUM_B_HOME"].iat[i]
                    game_stadium = stadium_game.iat[i] if hasattr(stadium_game, "iat") else None
                    same_stadium = b_home_stadium == game_stadium
                    if pd.notna(b_home_stadium) and pd.notna(game_stadium) and same_stadium:
                        km_b.append(0.0)
                        tz_b.append(0.0)
                        continue

            km_b.append(float(measure_distance((lat_b, lon_b), (lat_s, lon_s))))
            tz_b.append(
                float(
                    calculate_timezone_difference(
                        lat_b, lon_b, lat_s, lon_s, tz_find=tz_finder, when=None
                    )
                )
            )

        base["TEAM_B_KM_TRAVELED"] = km_b
        base["_TEAM_B_TZ_TRAVELED"] = tz_b

        # Drop the temporary home coord columns - only keep the feature outputs
        drop_cols = [
            c
            for c in ["_LAT_B_HOME", "_LON_B_HOME", "_ALT_B_HOME", "_STADIUM_B_HOME"]
            if c in base.columns
        ]
        return base.drop(columns=drop_cols)

    def _add_neutral_site(self, df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        """Attach IS_NEUTRAL_SITE flag from the GAME_LOCATION column.

        GAME_LOCATION == "N" in the canonical games schema marks international
        and other neutral-site games.

        Args:
            df: Modeling DataFrame.
            games: Cleaned games DataFrame.

        Returns:
            df with IS_NEUTRAL_SITE column appended.
        """
        neutral = games[["GAME_ID", "GAME_LOCATION"]].drop_duplicates("GAME_ID").copy()
        neutral["IS_NEUTRAL_SITE"] = (neutral["GAME_LOCATION"] == "N").astype(int)
        return df.merge(
            neutral[["GAME_ID", "IS_NEUTRAL_SITE"]],
            how="left",
            on="GAME_ID",
        )


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
