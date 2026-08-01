# src/gridiron_edge/features/team/primetime.py

"""Canonical nullable primetime game feature.

Determines primetime state from schedule day and kickoff metadata for one
canonical game row. Historical and upcoming metadata share the same
``IS_PRIMETIME`` output contract.

Monday games are primetime regardless of kickoff time. Sunday, Thursday,
and Saturday games are primetime when kickoff is at or after 20:00.
Unavailable or invalid schedule metadata produces an explicit null state.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Kickoff hour (24-hour) at or above which an evening game is primetime.
# 20:00 covers 8:00 PM kickoffs - the standard SNF/TNF/MNF slot.
_PRIMETIME_HOUR: Final[int] = 20

_PRIMETIME_DAYS_ANY_TIME: Final[frozenset[str]] = frozenset({"Monday"})
_PRIMETIME_DAYS_EVENING: Final[frozenset[str]] = frozenset({"Sunday", "Thursday", "Saturday"})

_KNOWN_GAME_DAYS: Final[frozenset[str]] = frozenset(
    {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
)


def _is_missing_metadata_value(value: object) -> bool:
    """Return whether a scalar metadata value is unavailable."""
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _canonical_primetime_value(
    day_value: object,
    time_value: object,
) -> object:
    """Return nullable primetime state from known schedule metadata."""
    if _is_missing_metadata_value(day_value):
        return pd.NA

    day: str = str(day_value).strip()
    if day not in _KNOWN_GAME_DAYS:
        return pd.NA

    result: object = 0

    if day in _PRIMETIME_DAYS_ANY_TIME:
        result = 1
    elif day in _PRIMETIME_DAYS_EVENING:
        if _is_missing_metadata_value(time_value):
            result = pd.NA
        else:
            gametime: str = str(time_value).strip()
            try:
                hour = int(gametime.split(":")[0])
            except (ValueError, IndexError):
                result = pd.NA
            else:
                result = int(hour >= _PRIMETIME_HOUR) if 0 <= hour <= 23 else pd.NA

    return result


@FeatureRegistry.register("home_away_primetime")
class HomeAwayPrimetimeFeature:
    """Attach schedule-complete nullable primetime identity."""

    spec = FeatureSpec(
        name="home_away_primetime",
        produces=["IS_PRIMETIME"],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach primetime state by canonical game identity."""
        if "GAME_ID" not in df.columns:
            raise ValueError("Home/away game frame is missing required columns: GAME_ID")

        source = df.copy().drop(
            columns=["IS_PRIMETIME"],
            errors="ignore",
        )
        source["_INPUT_ORDER"] = range(len(source))

        lookup = build_game_metadata_lookup(
            historical=datasets.games(),
            upcoming=load_optional_upcoming_metadata(datasets),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "GAME_DAY_OF_WEEK": "GAME_DAY_OF_WEEK",
                "GAMETIME": "GAMETIME",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "game_day_of_week": "GAME_DAY_OF_WEEK",
                "game_time": "GAMETIME",
            },
        )
        lookup["IS_PRIMETIME"] = pd.Series(
            [
                _canonical_primetime_value(day, gametime)
                for day, gametime in zip(
                    lookup["GAME_DAY_OF_WEEK"],
                    lookup["GAMETIME"],
                    strict=True,
                )
            ],
            index=lookup.index,
            dtype="Int64",
        )

        result = source.merge(
            lookup[["GAME_ID", "IS_PRIMETIME"]],
            how="left",
            on="GAME_ID",
            sort=False,
            validate="many_to_one",
        )
        return (
            result.sort_values("_INPUT_ORDER", kind="stable")
            .drop(columns=["_INPUT_ORDER"])
            .reset_index(drop=True)
        )
