# src/gridiron_edge/features/team/home_field.py

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


@FeatureRegistry.register("home_field")
class HomeFieldFeature:
    """Feature that encodes home-field advantage for each team in a matchup.

    Produces ``HOME_FIELD`` as a binary integer: ``1`` if the team is playing
    at home, ``0`` if away or neutral. Derived from the ``GAME_LOCATION``
    column in the games dataset (``NULL_VALUE`` indicates a standard home game).
    """

    spec = FeatureSpec(name="home_field", produces=["HOME_FIELD"])

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute home-field advantage indicator and join onto the modeling DataFrame.

        Builds a symmetric long-form table with one row per team per game,
        then merges onto ``df`` on ``GAME_ID``, ``TEAM_A``, ``TEAM_B``,
        ``YEAR``, and ``WEEK_NUM``.

        Args:
            df: Modeling DataFrame containing matchup identifiers.
            datasets: Accessor providing the games dataset via
                ``datasets.games()``.

        Returns:
            Input DataFrame with ``HOME_FIELD`` column appended.
        """
        games = datasets.games()

        g = games.loc[
            :,
            ["GAME_ID", "WINNER", "LOSER", "YEAR", "WEEK_NUM", "GAME_LOCATION"],
        ].copy()
        g["HOME_FIELD"] = (g["GAME_LOCATION"] == "NULL_VALUE").astype(int)

        g1 = g.rename(columns={"WINNER": "TEAM_A", "LOSER": "TEAM_B"})[
            ["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM", "HOME_FIELD"]
        ]

        g2 = g.rename(columns={"LOSER": "TEAM_A", "WINNER": "TEAM_B"})[
            ["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM", "HOME_FIELD"]
        ]
        g2["HOME_FIELD"] = 1 - g2["HOME_FIELD"]

        home = pd.concat([g1, g2], ignore_index=True).drop_duplicates()
        return df.merge(
            home,
            how="left",
            on=["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM"],
        )
