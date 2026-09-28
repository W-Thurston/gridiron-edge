# src/gridiron_edge/features/team/unit_efficiency.py

"""Unit efficiency differentials: pass/rush offense against the opponent faced.

Domain 13 ("Unit Matchups & Interactions") features are built from both
teams at once, unlike every other canonical feature, which computes each
side independently and only differences them afterward. This one combines
the opponent-adjusted EPA columns ``home_away_opponent_adjusted_epa``
produces - Home's offense against Away's defense, and Away's offense
against Home's defense - rather than either team's efficiency in
isolation. Requires that feature to have already run (``depends_on``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.models.game_prediction.game_schema import (
    away_feature_name,
    home_feature_name,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Suffixes for _columns.py to build AWAY_/HOME_/_DIFF names from, mirroring
# how ADJUSTED_EPA_METRICS feeds the same mechanism for opponent-adjusted EPA.
UNIT_EFFICIENCY_SUFFIXES: Final[tuple[str, ...]] = (
    "PASS_MATCHUP_EPA",
    "RUSH_MATCHUP_EPA",
)

_REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_OFF_PASS_EPA_ADJ",
    "HOME_OFF_PASS_EPA_ADJ",
    "AWAY_DEF_PASS_EPA_ADJ",
    "HOME_DEF_PASS_EPA_ADJ",
    "AWAY_OFF_RUSH_EPA_ADJ",
    "HOME_OFF_RUSH_EPA_ADJ",
    "AWAY_DEF_RUSH_EPA_ADJ",
    "HOME_DEF_RUSH_EPA_ADJ",
)

AWAY_PASS_MATCHUP_EPA: Final[str] = away_feature_name("PASS_MATCHUP_EPA")
HOME_PASS_MATCHUP_EPA: Final[str] = home_feature_name("PASS_MATCHUP_EPA")
AWAY_RUSH_MATCHUP_EPA: Final[str] = away_feature_name("RUSH_MATCHUP_EPA")
HOME_RUSH_MATCHUP_EPA: Final[str] = home_feature_name("RUSH_MATCHUP_EPA")


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


@FeatureRegistry.register("unit_efficiency_differentials")
class UnitEfficiencyDifferentialsFeature:
    """Compute each side's pass/rush offense against the defense it faces."""

    spec = FeatureSpec(
        name="unit_efficiency_differentials",
        produces=[
            AWAY_PASS_MATCHUP_EPA,
            HOME_PASS_MATCHUP_EPA,
            AWAY_RUSH_MATCHUP_EPA,
            HOME_RUSH_MATCHUP_EPA,
        ],
        depends_on=["home_away_opponent_adjusted_epa"],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach unit-matchup EPA columns.

        Pure arithmetic on already-computed opponent-adjusted EPA columns;
        reads no external dataset. A game missing an input (for example a
        week-1 opponent-adjusted EPA that is itself null) produces null
        matchup values, the same as every upstream dependency.

        Produces only the raw Away/Home matchup values, not their diff -
        matching every other canonical feature's convention (Elo,
        opponent-adjusted EPA), where ``_DIFF`` columns are derived later,
        on demand, by ``_make_diff_features`` rather than persisted here.
        """
        _require_columns(
            df,
            _REQUIRED_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        result = df.copy()
        result[AWAY_PASS_MATCHUP_EPA] = (
            result["AWAY_OFF_PASS_EPA_ADJ"] - result["HOME_DEF_PASS_EPA_ADJ"]
        )
        result[HOME_PASS_MATCHUP_EPA] = (
            result["HOME_OFF_PASS_EPA_ADJ"] - result["AWAY_DEF_PASS_EPA_ADJ"]
        )
        result[AWAY_RUSH_MATCHUP_EPA] = (
            result["AWAY_OFF_RUSH_EPA_ADJ"] - result["HOME_DEF_RUSH_EPA_ADJ"]
        )
        result[HOME_RUSH_MATCHUP_EPA] = (
            result["HOME_OFF_RUSH_EPA_ADJ"] - result["AWAY_DEF_RUSH_EPA_ADJ"]
        )

        return result
