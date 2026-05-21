# src/gridiron_edge/features/team/travel.py

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.metrics.travel import add_travel_timezone_altitude

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


@FeatureRegistry.register("travel")
class TravelFeature:
    """Feature that computes travel distance, timezone shift, and altitude for TEAM_A.

    Produces ``TEAM_A_KM_TRAVELED``, ``TEAM_A_TZ_TRAVELED``, and ``ALTITUDE``
    by comparing each team's home stadium coordinates to the game venue
    coordinates using the travel metrics library.
    """

    spec = FeatureSpec(
        name="travel",
        produces=["TEAM_A_KM_TRAVELED", "TEAM_A_TZ_TRAVELED", "ALTITUDE"],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute travel and altitude features and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame containing matchup and team identifiers.
            datasets: Accessor providing games and stadiums datasets via
                ``datasets.games()`` and ``datasets.stadiums()``.

        Returns:
            Input DataFrame with ``TEAM_A_KM_TRAVELED``, ``TEAM_A_TZ_TRAVELED``,
            and ``ALTITUDE`` columns appended.
        """
        return add_travel_timezone_altitude(
            df,
            datasets.games(),
            datasets.stadiums(),
        )
