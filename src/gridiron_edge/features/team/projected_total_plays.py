# src/gridiron_edge/features/team/projected_total_plays.py

"""Projected total plays: both teams' pace combined (market-blind variant).

Domain 13. The market-blind variant sums each team's own rolling
plays-per-game (``AWAY_OFF_PLAYS``/``HOME_OFF_PLAYS``, already produced by
``home_away_epa``) rather than adjusting for expected game script from the
spread - that adjustment needs market data and belongs to Unit 6's
market-aware variant, a separate model identity from this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

PROJECTED_TOTAL_PLAYS: Final[str] = "PROJECTED_TOTAL_PLAYS"

_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_OFF_PLAYS",
    "HOME_OFF_PLAYS",
)


def _require_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require an established input schema."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


@FeatureRegistry.register("projected_total_plays")
class ProjectedTotalPlaysFeature:
    """Market-blind projected total plays: both teams' own pace, summed."""

    spec = FeatureSpec(
        name="projected_total_plays",
        produces=[PROJECTED_TOTAL_PLAYS],
        depends_on=["home_away_epa"],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach the combined projected-plays column.

        Pure arithmetic on already-rolled pace columns; reads no external
        dataset. A game missing either team's rolling plays-per-game (for
        example week 1) produces a null projection, the same as every
        upstream dependency.
        """
        _require_columns(
            df,
            _REQUIRED_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        result = df.copy()
        result[PROJECTED_TOTAL_PLAYS] = result["AWAY_OFF_PLAYS"] + result["HOME_OFF_PLAYS"]
        return result
