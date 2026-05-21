# src/gridiron_edge/features/team/elo.py

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from pandas import DataFrame

    from gridiron_edge.datasets.accessor import DatasetAccessor


@FeatureRegistry.register("team_elo")
class TeamEloFeature:
    """Feature that joins current Elo ratings for both teams in each matchup.

    Produces ``TEAM_A_ELO`` and ``TEAM_B_ELO`` by merging the Elo state
    table against the modeling DataFrame on team name, season year, and
    week number.
    """

    spec = FeatureSpec(
        name="team_elo",
        produces=["TEAM_A_ELO", "TEAM_B_ELO"],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Join Elo ratings for TEAM_A and TEAM_B onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame containing ``TEAM_A``, ``TEAM_B``,
                ``YEAR``, and ``WEEK_NUM`` columns.
            datasets: Accessor providing the Elo state table via
                ``datasets.elo_state()``.

        Returns:
            Input DataFrame with ``TEAM_A_ELO`` and ``TEAM_B_ELO`` columns
            appended.
        """
        elo: DataFrame = datasets.elo_state().copy()

        out: pd.DataFrame = df.merge(
            elo,
            how="left",
            left_on=["TEAM_A", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        ).drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        out = out.rename(columns={"ELO": "TEAM_A_ELO"})

        out = out.merge(
            elo,
            how="left",
            left_on=["TEAM_B", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        ).drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        return out.rename(columns={"ELO": "TEAM_B_ELO"})
