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
    - IS_NEUTRAL_SITE is independent of travel — a London game has
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

        # Drop the temporary home coord columns — only keep the feature outputs
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
