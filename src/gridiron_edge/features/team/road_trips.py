# src/gridiron_edge/features/team/road_trips.py

"""Canonical Away/Home back-to-back road-game flags.

A side is on the road when it is the designated Away team or when the game
is at a neutral site (both sides travel). The flag is 1 when the side is on
the road, its immediately preceding scheduled game in the same season was
also on the road, and that game was fewer than ``_BYE_GAP_DAYS`` days
earlier (a bye resets the fatigue argument).

Unlike the completed-history rest features, this reads the full schedule
(played and unplayed games): opponent and site are known before kickoff for
every scheduled game, so the flag is exact at any lookahead. A team's first
game of a season has no preceding game and is 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Same threshold as the post-bye flag in ``rest.py``: 13+ days between games
# is a bye week.
_BYE_GAP_DAYS: Final[int] = 13

_OUTPUT_COLUMNS: Final[tuple[str, str]] = (
    "AWAY_BACK_TO_BACK_ROAD",
    "HOME_BACK_TO_BACK_ROAD",
)

_INPUT_COLUMNS: Final[tuple[str, ...]] = ("GAME_ID", "AWAY_TEAM", "HOME_TEAM")


def _team_schedule(schedule: DataFrame) -> DataFrame:
    """Expand one row per game into one row per (game, team) with road state."""
    neutral = pd.to_numeric(schedule["IS_NEUTRAL_SITE"], errors="coerce").fillna(0).astype(int)
    common = schedule.assign(_NEUTRAL=neutral)
    away = common.assign(TEAM=common["AWAY_TEAM"], ON_ROAD=1)
    home = common.assign(TEAM=common["HOME_TEAM"], ON_ROAD=common["_NEUTRAL"])
    rows = pd.concat([away, home], ignore_index=True).loc[
        :, ["GAME_ID", "TEAM", "YEAR", "GAME_DATE", "ON_ROAD"]
    ]
    rows["GAME_DATE"] = pd.to_datetime(rows["GAME_DATE"], errors="coerce")
    return rows.sort_values(["TEAM", "GAME_DATE", "GAME_ID"], kind="stable", ignore_index=True)


def _back_to_back_flags(team_schedule: DataFrame) -> DataFrame:
    """Return (GAME_ID, TEAM, flag) for every scheduled team-game."""
    grouped = team_schedule.groupby(["TEAM", "YEAR"], sort=False)
    previous_on_road = grouped["ON_ROAD"].shift(1)
    # pyrefly: ignore [missing-attribute]
    gap_days = (
        team_schedule["GAME_DATE"] - grouped["GAME_DATE"].shift(1)
    ).dt.total_seconds() / 86400.0

    has_previous = previous_on_road.notna()
    dates_known = team_schedule["GAME_DATE"].notna() & (gap_days.notna() | ~has_previous)

    flag = (
        (team_schedule["ON_ROAD"] == 1) & (previous_on_road == 1) & (gap_days < _BYE_GAP_DAYS)
    ).astype("Int64")
    result = team_schedule.loc[:, ["GAME_ID", "TEAM"]].copy()
    result["FLAG"] = flag.mask(~dates_known)
    return result


@FeatureRegistry.register("home_away_road_trips")
class HomeAwayRoadTripsFeature:
    """Attach Away and Home back-to-back road flags by canonical game."""

    spec = FeatureSpec(
        name="home_away_road_trips",
        produces=list(_OUTPUT_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach schedule-complete back-to-back road flags."""
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
                "YEAR": "YEAR",
                "GAME_DATE": "GAME_DATE",
                "AWAY_TEAM": "AWAY_TEAM",
                "HOME_TEAM": "HOME_TEAM",
                "IS_NEUTRAL_SITE": "IS_NEUTRAL_SITE",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "season": "YEAR",
                "game_date": "GAME_DATE",
                "away_team": "AWAY_TEAM",
                "home_team": "HOME_TEAM",
                "neutral_site": "IS_NEUTRAL_SITE",
            },
        )
        flags = _back_to_back_flags(_team_schedule(schedule))

        for side, column in (("AWAY", _OUTPUT_COLUMNS[0]), ("HOME", _OUTPUT_COLUMNS[1])):
            lookup = flags.rename(columns={"TEAM": f"{side}_TEAM", "FLAG": column})
            source = source.merge(
                lookup,
                how="left",
                on=["GAME_ID", f"{side}_TEAM"],
                sort=False,
                validate="many_to_one",
            )
            source[column] = source[column].astype("Int64")

        return source.reset_index(drop=True)
