# src/gridiron_edge/features/team/divisional.py

"""Divisional game flag feature.

Marks whether a matchup is between two teams in the same NFL division.
Divisional games are historically tighter contests - teams know each other
well, play twice per season, and the stakes (division standings) are higher
than neutral-conference matchups.

nflverse provides ``div_game`` directly on the schedule, so this feature
requires no computation beyond reading the canonical games CSV where
``DIV_GAME`` has already been persisted by ``clean_nflverse_games``.

Produces:

    IS_DIV_GAME     int     1 if both teams are in the same NFL division,
                            0 otherwise.  Postseason games between division
                            rivals are also flagged (nflverse sets div_game=1
                            for those).

Design notes:
    - The two-row-per-game design means IS_DIV_GAME is identical in both
      rows for a given game, which is correct: divisional rivalry is a
      property of the matchup, not of which team is TEAM_A.
    - Postseason games carry the correct div_game value from nflverse.
      A Wild Card matchup between two NFC East teams will be flagged 1;
      a cross-conference Super Bowl will be 0.
    - Games ingested before DIV_GAME was added to the canonical schema
      will have NaN here.  ``_prepare_data`` in all model training paths
      excludes NaN-feature rows, so those games are automatically withheld
      from training without special-casing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

_GAMES_COLS: Final[list[str]] = ["GAME_ID", "DIV_GAME"]


@FeatureRegistry.register("divisional")
class DivisionalFeature:
    """Divisional game flag: IS_DIV_GAME.

    Reads DIV_GAME from the canonical games CSV (populated from nflverse's
    ``div_game`` field in ``clean_nflverse_games``) and attaches it to the
    modeling DataFrame as IS_DIV_GAME.
    """

    spec = FeatureSpec(
        name="divisional",
        produces=["IS_DIV_GAME"],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute the divisional game flag and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with at least a GAME_ID column.
            datasets: Provides ``games()`` for the DIV_GAME column.

        Returns:
            Input DataFrame with IS_DIV_GAME appended.  Values are 0 or 1;
            rows whose GAME_ID is not found in the games CSV receive NaN.
        """
        games: pd.DataFrame = datasets.games()

        if "DIV_GAME" not in games.columns:
            # Games CSV predates the DIV_GAME column - fill with NaN so
            # _prepare_data excludes these rows rather than training on zeros.
            df = df.copy()
            df["IS_DIV_GAME"] = float("nan")
            return df

        div_lookup: pd.DataFrame = (
            games[_GAMES_COLS]
            # pyrefly: ignore [no-matching-overload]
            .drop_duplicates("GAME_ID")
            .rename(columns={"DIV_GAME": "IS_DIV_GAME"})
        )

        df = df.merge(div_lookup, on="GAME_ID", how="left")
        return df


def _canonical_divisional_values(values: pd.Series) -> pd.Series:
    """Validate nullable divisional-game metadata."""
    # pyrefly: ignore [bad-assignment]
    numeric: pd.Series = pd.to_numeric(values, errors="coerce")
    invalid: Series[bool] = values.notna() & numeric.isna()
    invalid |= numeric.notna() & ~numeric.isin([0, 1])
    if invalid.any():
        raise ValueError("IS_DIV_GAME metadata must contain only 0, 1, or null.")
    return numeric.astype("Int64")


@FeatureRegistry.register("home_away_divisional")
class HomeAwayDivisionalFeature:
    """Attach schedule-complete divisional-game identity."""

    spec = FeatureSpec(
        name="home_away_divisional",
        produces=["IS_DIV_GAME"],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach nullable divisional state by canonical game identity."""
        if "GAME_ID" not in df.columns:
            raise ValueError("Home/away game frame is missing required columns: GAME_ID")

        source: DataFrame = df.copy().drop(
            columns=["IS_DIV_GAME"],
            errors="ignore",
        )
        source["_INPUT_ORDER"] = range(len(source))

        lookup: DataFrame = build_game_metadata_lookup(
            historical=datasets.games(),
            upcoming=load_optional_upcoming_metadata(datasets),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "DIV_GAME": "IS_DIV_GAME",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "divisional": "IS_DIV_GAME",
            },
        )
        lookup["IS_DIV_GAME"] = _canonical_divisional_values(lookup["IS_DIV_GAME"])

        result: DataFrame = source.merge(
            lookup,
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
