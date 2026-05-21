# src/gridiron_edge/metrics/travel.py

from typing import Final

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from gridiron_edge.metrics.travel.geo import (
    calculate_timezone_difference,
    measure_distance,
)

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


def add_travel_timezone_altitude(
    modeling: pd.DataFrame,
    games: pd.DataFrame,
    stadiums: pd.DataFrame,
) -> pd.DataFrame:
    """Add travel + timezone + altitude features using the ACTUAL game stadium.

    This version uses your cleaned games schema:
      - games.STADIUM = stadium name where the game was played
      - games.GAME_LOCATION is '@' for away and 'NULL_VALUE' for home (not needed if HOME_FIELD exists)

    Modeling required columns:
      - TEAM_A, TEAM_B, YEAR, WEEK_NUM
      - HOME_FIELD (1 if TEAM_A is home, 0 if TEAM_A is away)

    Stadiums required columns:
      - TEAM mapping: HOME_TEAM, YEAR, LATITUDE, LONGITUDE, ALTITUDE, STADIUM (team home stadium name)
      - Stadium lookup: STADIUM, LATITUDE, LONGITUDE, ALTITUDE (same table, used as lookup by name)

    Outputs:
      - LATITUDE_A, LONGITUDE_A (TEAM_A home coords)
      - LATITUDE_SITE, LONGITUDE_SITE (actual game stadium coords)
      - TEAM_A_KM_TRAVELED (TEAM_A home -> game site)
      - TEAM_A_TZ_TRAVELED (timezone offset TEAM_A home -> game site)
      - ALTITUDE (altitude at game site)
    """
    df: pd.DataFrame = modeling.copy()
    if "HOME_FIELD" not in df.columns:
        raise ValueError(
            "HOME_FIELD missing. Ensure home-field feature runs before travel features.",
        )

    # Base join keys
    base = df.loc[:, [*_KEY_COLS, "HOME_FIELD"]].copy()

    # --- Attach game stadium name from cleaned games ---
    g = games.loc[:, _GAMES_COLS].copy()

    # We need to match a modeling row to a game row regardless of home/away orientation.
    # Cleaned games store WINNER/LOSER, not TEAM_A/TEAM_B. We'll join both directions and coalesce.
    g1 = g.rename(
        columns={"WINNER": "TEAM_A", "LOSER": "TEAM_B", "STADIUM": "STADIUM_GAME"},
    )
    g2 = g.rename(
        columns={"WINNER": "TEAM_B", "LOSER": "TEAM_A", "STADIUM": "STADIUM_GAME"},
    )

    base = base.merge(
        g1[["YEAR", "WEEK_NUM", "TEAM_A", "TEAM_B", "STADIUM_GAME"]],
        on=_KEY_COLS,
        how="left",
    )
    base = base.merge(
        g2[["YEAR", "WEEK_NUM", "TEAM_A", "TEAM_B", "STADIUM_GAME"]].rename(
            columns={"STADIUM_GAME": "STADIUM_GAME_REV"},
        ),
        on=_KEY_COLS,
        how="left",
    )
    base["STADIUM_GAME"] = base["STADIUM_GAME"].combine_first(base["STADIUM_GAME_REV"])
    base = base.drop(columns=["STADIUM_GAME_REV"])

    # --- Attach TEAM_A home coords + home stadium name ---
    st_team = stadiums.loc[:, _STADIUMS_TEAM_YEAR_COLS].copy()
    base = (
        base.merge(
            st_team,
            how="left",
            left_on=["TEAM_A", "YEAR"],
            right_on=["HOME_TEAM", "YEAR"],
        )
        .drop(columns=["HOME_TEAM"])
        .rename(
            columns={
                "LATITUDE": "LATITUDE_A",
                "LONGITUDE": "LONGITUDE_A",
                "ALTITUDE": "ALTITUDE_A_HOME",
                "STADIUM": "STADIUM_A_HOME",
            },
        )
    )

    # --- Attach stadium coords for the ACTUAL game site by stadium name ---
    st_by_name = stadiums.loc[:, _STADIUMS_BY_NAME_COLS].drop_duplicates(subset=["STADIUM"]).copy()
    base = base.merge(
        st_by_name.rename(
            columns={
                "LATITUDE": "LATITUDE_SITE",
                "LONGITUDE": "LONGITUDE_SITE",
                "ALTITUDE": "ALTITUDE_SITE",
            },
        ),
        how="left",
        left_on="STADIUM_GAME",
        right_on="STADIUM",
    ).drop(columns=["STADIUM"])

    # If we couldn't find game stadium coords (bad stadium name mapping), fall back to home stadium logic:
    # - if TEAM_A is home -> use TEAM_A home coords
    # - else -> we could use TEAM_B home coords, but we only need site coords for travel distance.
    # For completeness, attach TEAM_B home coords to use as fallback for site when stadium mapping fails.
    st_team_b = (
        stadiums.loc[:, _STADIUMS_TEAM_YEAR_COLS]
        .copy()
        .rename(
            columns={
                "LATITUDE": "LATITUDE_B_HOME",
                "LONGITUDE": "LONGITUDE_B_HOME",
                "ALTITUDE": "ALTITUDE_B_HOME",
                "STADIUM": "STADIUM_B_HOME",
            },
        )
    )
    base = base.merge(
        st_team_b,
        how="left",
        left_on=["TEAM_B", "YEAR"],
        right_on=["HOME_TEAM", "YEAR"],
    ).drop(columns=["HOME_TEAM"])

    # Fill missing site coords (stadium lookup failed) using home team coords.
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
            np.where(
                base["HOME_FIELD"] == 1,
                base["LONGITUDE_A"],
                base["LONGITUDE_B_HOME"],
            ),
            index=base.index,
        ),
    )
    base["ALTITUDE"] = base["ALTITUDE_SITE"].where(
        base["ALTITUDE_SITE"].notna(),
        pd.Series(
            np.where(
                base["HOME_FIELD"] == 1,
                base["ALTITUDE_A_HOME"],
                base["ALTITUDE_B_HOME"],
            ),
            index=base.index,
        ),
    )

    # --- Compute travel ---
    # Travel is 0 only if the game is truly at TEAM_A's home stadium.
    # We check both HOME_FIELD==1 and stadium name equals TEAM_A home stadium name.
    is_true_home = (
        (base["HOME_FIELD"] == 1)
        & (base["STADIUM_GAME"].notna())
        & (base["STADIUM_GAME"] == base["STADIUM_A_HOME"])
    )

    travel_mask = ~is_true_home

    # Validate needed coords for rows we will compute
    missing = (
        base.loc[
            travel_mask,
            ["LATITUDE_A", "LONGITUDE_A", "LATITUDE_SITE", "LONGITUDE_SITE"],
        ]
        .isna()
        .any(axis=1)
    )
    if missing.any():
        sample = base.loc[travel_mask].loc[missing, [*_KEY_COLS, "STADIUM_GAME"]].head(10)
        msg: str = (
            "Missing coordinates needed for travel computation (TEAM_A home or game site). "
            "Check stadium coverage and stadium name mapping. Sample rows:\n"
            f"{sample.to_string(index=False)}"
        )
        raise ValueError(msg)

    tz_finder = TimezoneFinder()
    base["TEAM_A_KM_TRAVELED"] = 0.0
    base["TEAM_A_TZ_TRAVELED"] = 0.0

    idx = base.index[travel_mask].to_list()
    if idx:
        km_vals: list[float] = []
        tz_vals: list[float] = []

        for _, r in base.loc[idx].iterrows():
            km_vals.append(
                float(
                    measure_distance(
                        (r["LATITUDE_A"], r["LONGITUDE_A"]),
                        (r["LATITUDE_SITE"], r["LONGITUDE_SITE"]),
                    ),
                ),
            )
            tz_vals.append(
                float(
                    calculate_timezone_difference(
                        r["LATITUDE_A"],
                        r["LONGITUDE_A"],
                        r["LATITUDE_SITE"],
                        r["LONGITUDE_SITE"],
                        tz_find=tz_finder,
                        when=None,
                    ),
                ),
            )

        base.loc[idx, "TEAM_A_KM_TRAVELED"] = km_vals
        base.loc[idx, "TEAM_A_TZ_TRAVELED"] = tz_vals

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
    feats: pd.DataFrame = base.loc[:, _KEY_COLS + feature_cols].copy()

    out: pd.DataFrame = df.merge(feats, how="left", on=_KEY_COLS)
    return out.drop_duplicates().reset_index(drop=True)
