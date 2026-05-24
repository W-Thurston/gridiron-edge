# src/gridiron_edge/features/team/epa.py

"""Rolling EPA (Expected Points Added) features for both teams in a matchup.

Computes pre-game rolling EPA statistics for TEAM_A and TEAM_B using
the previous N games as the rolling window. The window size is a
tunable parameter — see ``evaluate tune --epa`` (Phase 19) for the
grid search that identifies the optimal window.

Rolling window design:
    For a game in season Y, week W, the rolling window uses the N most
    recent games from:
      1. Weeks 1 through W-1 of season Y (current season, prior weeks)
      2. If N > W-1, the tail of season Y-1 fills the remaining slots

    This ensures strict temporal integrity — no future information leaks
    into the feature values.

Features produced (per team, for both TEAM_A and TEAM_B):
    {PREFIX}_OFF_EPA_PER_PLAY    Offensive EPA/play (pass + rush)
    {PREFIX}_OFF_PASS_EPA        Offensive passing EPA/play
    {PREFIX}_OFF_RUSH_EPA        Offensive rushing EPA/play
    {PREFIX}_OFF_SUCCESS_RATE    Fraction of offensive plays with EPA > 0
    {PREFIX}_DEF_EPA_PER_PLAY    Defensive EPA/play allowed
    {PREFIX}_DEF_PASS_EPA        Defensive passing EPA/play allowed
    {PREFIX}_DEF_RUSH_EPA        Defensive rushing EPA/play allowed
    {PREFIX}_DEF_SUCCESS_RATE    Fraction of opponent plays with EPA > 0

Where PREFIX is ``TEAM_A`` or ``TEAM_B``.

EPA reliability note:
    nflfastR EPA model is reliable from 2006 onward. Games before 2006
    will produce NaN EPA features, which downstream models should handle
    via imputation or by filtering training data to 2006+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Default rolling window — will be tuned in Phase 19 evaluation.
# 4 games is a reasonable NFL default: enough to reduce noise,
# short enough to capture current-season form.
DEFAULT_ROLLING_WINDOW: Final[int] = 4

# EPA columns from epa_by_game.parquet that we roll over
_EPA_COLS: Final[list[str]] = [
    "off_epa_per_play",
    "off_pass_epa",
    "off_rush_epa",
    "off_success_rate",
    "def_epa_per_play",
    "def_pass_epa",
    "def_rush_epa",
    "def_success_rate",
]

# Canonical column names produced for TEAM_A and TEAM_B
_TEAM_A_COLS: Final[list[str]] = [f"TEAM_A_{c.upper()}" for c in _EPA_COLS]
_TEAM_B_COLS: Final[list[str]] = [f"TEAM_B_{c.upper()}" for c in _EPA_COLS]


def _build_rolling_epa(
    epa_by_game: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Compute rolling EPA features for every (team, season, week) triple.

    For each game a team plays in week W of season Y, computes the rolling
    mean of each EPA metric over the previous ``window`` games played by
    that team. Games are ordered chronologically by (season, week).

    Pre-game only: the rolling window at week W uses games through week W-1,
    ensuring no leakage.

    Args:
        epa_by_game: Game-level EPA aggregation with columns
            ``game_id``, ``season``, ``week``, ``team``, plus EPA metric cols.
        window: Number of prior games to include in each rolling window.

    Returns:
        DataFrame with columns ``season``, ``week``, ``team``,
        plus rolled EPA columns prefixed with ``rolling_``.
        One row per (season, week, team) matching the input.
    """
    df: DataFrame = epa_by_game.copy()

    # Sort chronologically within each team
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Compute rolling means grouped by team
    # shift(1) ensures we use prior games only (no current game leakage)
    rolled_parts: list[DataFrame] = []
    for _team, group in df.groupby("team", sort=False):
        sorted_group: DataFrame = group.sort_values(["season", "week"]).copy()
        for col in _EPA_COLS:
            if col in sorted_group.columns:
                sorted_group[f"rolling_{col}"] = (
                    sorted_group[col].shift(1).rolling(window=window, min_periods=1).mean()
                )
            else:
                sorted_group[f"rolling_{col}"] = float("nan")
        rolled_parts.append(sorted_group)

    rolled: DataFrame = pd.concat(rolled_parts, ignore_index=True)

    # Keep only the keys and rolled columns
    rolling_cols: list[str] = [f"rolling_{c}" for c in _EPA_COLS]
    return rolled.loc[:, ["game_id", "season", "week", "team", *rolling_cols]].copy()


@FeatureRegistry.register("epa")
class TeamEpaFeature:
    """Rolling EPA features for both teams in each matchup.

    Joins pre-game rolling EPA statistics onto the modeling DataFrame
    for both TEAM_A and TEAM_B. The rolling window defaults to
    ``DEFAULT_ROLLING_WINDOW`` games but can be overridden for tuning.
    """

    spec = FeatureSpec(
        name="epa",
        produces=_TEAM_A_COLS + _TEAM_B_COLS,
    )

    def __init__(self, window: int = DEFAULT_ROLLING_WINDOW) -> None:
        """Initialise with a rolling window size.

        Args:
            window: Number of prior games to use in each rolling window.
                Defaults to ``DEFAULT_ROLLING_WINDOW`` (4 games).
        """
        self.window = window

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Join rolling EPA features for TEAM_A and TEAM_B.

        Args:
            df: Modeling DataFrame with ``TEAM_A``, ``TEAM_B``,
                ``YEAR``, ``WEEK_NUM`` columns. ``YEAR`` is the
                canonical season label (e.g. ``"2025-2026"``).
            datasets: Accessor providing ``epa_by_game()`` — the
                pre-aggregated game-level EPA table.

        Returns:
            Input DataFrame with TEAM_A and TEAM_B EPA columns appended.
            Games before 2006 (or missing from PBP cache) will have
            NaN values in all EPA columns.
        """
        epa_raw: DataFrame = datasets.epa_by_game()

        if epa_raw.empty:
            # No EPA data available — add NaN columns and return
            for col in _TEAM_A_COLS + _TEAM_B_COLS:
                df[col] = float("nan")
            return df

        rolled: DataFrame = _build_rolling_epa(epa_raw, window=self.window)

        # The modeling DataFrame uses long season labels ("2025-2026") and
        # WEEK_NUM, but EPA data uses season int (2025) and week int.
        # Map season label to int for the join.
        year_to_int = (
            df[["YEAR"]].drop_duplicates().assign(season=lambda d: d["YEAR"].str[:4].astype(int))
        )

        df_with_season: DataFrame = df.merge(year_to_int, on="YEAR", how="left")

        # Join TEAM_A
        team_a_epa: DataFrame = rolled.rename(
            columns={f"rolling_{c}": f"TEAM_A_{c.upper()}" for c in _EPA_COLS}
        )
        out: DataFrame = df_with_season.merge(
            team_a_epa[["season", "week", "team"] + [f"TEAM_A_{c.upper()}" for c in _EPA_COLS]],
            how="left",
            left_on=["season", "WEEK_NUM", "TEAM_A"],
            right_on=["season", "week", "team"],
        ).drop(columns=["week", "team"], errors="ignore")

        # Join TEAM_B
        team_b_epa: DataFrame = rolled.rename(
            columns={f"rolling_{c}": f"TEAM_B_{c.upper()}" for c in _EPA_COLS}
        )
        out = out.merge(
            team_b_epa[["season", "week", "team"] + [f"TEAM_B_{c.upper()}" for c in _EPA_COLS]],
            how="left",
            left_on=["season", "WEEK_NUM", "TEAM_B"],
            right_on=["season", "week", "team"],
        ).drop(columns=["week", "team", "season"], errors="ignore")

        return out.reset_index(drop=True)
