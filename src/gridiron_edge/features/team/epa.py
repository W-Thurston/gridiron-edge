# src/gridiron_edge/features/team/epa.py

"""Rolling EPA (Expected Points Added) features for both teams in a matchup.

Computes pre-game rolling EPA statistics for TEAM_A and TEAM_B using
the previous N games as the rolling window. The window size is a
tunable parameter - see ``evaluate tune --epa`` for the
grid search that identifies the optimal window.

Rolling window design:
    For a game in season Y, week W, the rolling window uses the N most
    recent games from:
      1. Weeks 1 through W-1 of season Y (current season, prior weeks)
      2. If N > W-1, the tail of season Y-1 fills the remaining slots

    This ensures strict temporal integrity - no future information leaks
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
    {PREFIX}_OFF_EXPLOSIVE_RATE   Fraction of offensive plays that are explosive
    {PREFIX}_DEF_EXPLOSIVE_RATE   Fraction of opponent explosive plays allowed

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

# Default rolling window - will be tuned in evaluation.
# 4 games is a reasonable NFL default: enough to reduce noise,
# short enough to capture current-season form.
DEFAULT_ROLLING_WINDOW: Final[int] = 4

# EPA columns from epa_by_game.parquet that we roll over.
# This is the single source of truth for EPA metric names - both the model
# layer (_shared.py) and tree tuning (tree.py) derive their column lists
# from this constant rather than maintaining independent copies.
EPA_COLS: Final[list[str]] = [
    # --- Offensive ---
    "off_epa_per_play",
    "off_pass_epa",
    "off_rush_epa",
    "off_success_rate",
    "off_pass_success_rate",
    "off_rush_success_rate",
    "off_explosive_rate",
    "off_third_down_pct",
    "off_redzone_td_pct",
    "off_turnover_rate",
    "off_sack_rate",
    "off_plays",
    "off_yards_per_play",
    "off_redzone_attempts",
    "off_int_rate",
    "off_penalty_rate",
    "off_avg_score_diff",
    "off_close_game_pct",
    # --- Defensive ---
    "def_epa_per_play",
    "def_pass_epa",
    "def_rush_epa",
    "def_success_rate",
    "def_pass_success_rate",
    "def_rush_success_rate",
    "def_explosive_rate",
    "def_third_down_pct",
    "def_redzone_td_pct",
    "def_turnover_rate",
    "def_sack_rate",
    "def_plays",
    "def_yards_per_play",
    "def_redzone_attempts",
    "def_int_rate",
    "def_penalty_rate",
    "def_avg_score_diff",
    "def_close_game_pct",
]

# Private alias kept for internal use within this module
_EPA_COLS: list[str] = EPA_COLS

# Canonical column names produced for TEAM_A and TEAM_B
_TEAM_A_COLS: Final[list[str]] = [f"TEAM_A_{c.upper()}" for c in EPA_COLS]
_TEAM_B_COLS: Final[list[str]] = [f"TEAM_B_{c.upper()}" for c in EPA_COLS]


# Maximum regular-season week. Used by ``_build_rolling_epa`` to
# optionally exclude prior-season playoff games from the rolling
# window (epa/C1).
_MAX_REG_SEASON_WEEK: Final[int] = 18


def _build_rolling_epa(
    epa_by_game: pd.DataFrame,
    *,
    window: int,
    exclude_playoffs: bool = True,
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
        exclude_playoffs: If ``True`` (default), playoff games are
            excluded from the rolling window source. Early-season
            features therefore reflect only prior regular-season form,
            not the structurally different playoff slate. Set to
            ``False`` to include all completed games in the rolling
            window. See ``epa/C1``.

    Returns:
        DataFrame with columns ``season``, ``week``, ``team``,
        plus rolled EPA columns prefixed with ``rolling_``.
        One row per (season, week, team) matching the input.
    """
    df: DataFrame = epa_by_game.copy()

    # Optionally drop playoff games so they do not contribute to the
    # rolling window of any subsequent (regular- or post-season) game.
    if exclude_playoffs:
        df = df.loc[df["week"] <= _MAX_REG_SEASON_WEEK, :].copy()

    # Sort chronologically within each team
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Compute rolling means grouped by team.
    # shift(1) ensures we use prior games only (no current game leakage).
    rolled_parts: list[DataFrame] = []
    for _team, group in df.groupby("team", sort=False):
        sorted_group: DataFrame = group.sort_values(["season", "week"]).copy()
        available_cols: list[str] = [c for c in _EPA_COLS if c in sorted_group.columns]
        missing_cols: list[str] = [c for c in _EPA_COLS if c not in sorted_group.columns]

        if available_cols:
            rolled_vals = (
                sorted_group[available_cols].shift(1).rolling(window=window, min_periods=1).mean()
            )
            sorted_group[[f"rolling_{c}" for c in available_cols]] = rolled_vals

        for col in missing_cols:
            sorted_group[f"rolling_{col}"] = float("nan")

        rolled_parts.append(sorted_group)

    rolled: DataFrame = pd.concat(rolled_parts, ignore_index=True)

    # Keep only the keys and rolled columns
    rolling_cols: list[str] = [f"rolling_{c}" for c in _EPA_COLS]
    return rolled.loc[:, ["game_id", "season", "week", "team", *rolling_cols]].copy()


def _join_team_epa(
    df: DataFrame,
    rolled: DataFrame,
    *,
    prefix: str,
) -> DataFrame:
    """Join pre-game rolling EPA columns for one team perspective onto the modeling DataFrame.

    Args:
        df: Modeling DataFrame containing ``season``, ``WEEK_NUM``, and
            the team column identified by ``prefix`` (``"TEAM_A"`` or ``"TEAM_B"``).
        rolled: Rolling EPA DataFrame from ``_build_rolling_epa`` with
            columns ``season``, ``week``, ``team``, and ``rolling_*`` columns.
        prefix: Column prefix - either ``"TEAM_A"`` or ``"TEAM_B"``.

    Returns:
        Input DataFrame with ``{prefix}_{EPA_COL}`` columns appended.
    """
    renamed: DataFrame = rolled.rename(
        columns={f"rolling_{c}": f"{prefix}_{c.upper()}" for c in _EPA_COLS}
    )
    epa_cols: list[str] = [f"{prefix}_{c.upper()}" for c in _EPA_COLS]
    return df.merge(
        renamed[["season", "week", "team", *epa_cols]],
        how="left",
        left_on=["season", "WEEK_NUM", prefix],
        right_on=["season", "week", "team"],
    ).drop(columns=["week", "team"], errors="ignore")


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

    def __init__(
        self,
        window: int = DEFAULT_ROLLING_WINDOW,
        *,
        exclude_playoffs: bool = True,
    ) -> None:
        """Initialise with a rolling window size and playoff-exclusion flag.

        Args:
            window: Number of prior games to use in each rolling window.
                Defaults to ``DEFAULT_ROLLING_WINDOW`` (4 games).
            exclude_playoffs: If ``True`` (default), prior-season playoff
                games are excluded from the rolling window source. See
                ``epa/C1``.
        """
        self.window = window
        self.exclude_playoffs = exclude_playoffs

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
            datasets: Accessor providing ``epa_by_game()`` - the
                pre-aggregated game-level EPA table.

        Returns:
            Input DataFrame with TEAM_A and TEAM_B EPA columns appended.
            Games before 2006 (or missing from PBP cache) will have
            NaN values in all EPA columns.
        """
        epa_raw: DataFrame = datasets.epa_by_game()

        if epa_raw.empty:
            # No EPA data available - add NaN columns and return.
            # Use assign() to avoid mutating the caller's DataFrame.
            return df.assign(**{col: float("nan") for col in _TEAM_A_COLS + _TEAM_B_COLS})

        rolled: DataFrame = _build_rolling_epa(
            epa_raw,
            window=self.window,
            exclude_playoffs=self.exclude_playoffs,
        )

        # The modeling DataFrame uses long season labels ("2025-2026") and
        # WEEK_NUM, but EPA data uses season int (2025) and week int.
        # Map season label to int for the join.
        year_to_int = (
            df[["YEAR"]].drop_duplicates().assign(season=lambda d: d["YEAR"].str[:4].astype(int))
        )

        df_with_season: DataFrame = df.merge(year_to_int, on="YEAR", how="left")

        out: DataFrame = _join_team_epa(df_with_season, rolled, prefix="TEAM_A")
        out = _join_team_epa(out, rolled, prefix="TEAM_B")
        out = out.drop(columns=["season"], errors="ignore")

        return out.reset_index(drop=True)
