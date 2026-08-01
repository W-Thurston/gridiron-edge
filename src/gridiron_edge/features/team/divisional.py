# src/gridiron_edge/features/team/divisional.py

"""Canonical divisional-game feature.

Attaches schedule-provided divisional identity to one canonical game row.
Historical and upcoming metadata use the same nullable ``IS_DIV_GAME``
output contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
