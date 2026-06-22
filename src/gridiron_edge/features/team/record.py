"""Season win/loss record and current streak features.

Captures team momentum and form signals that Elo absorbs only slowly.
A team's win percentage and current streak encode information about
recent trajectory that aggregate Elo ratings smooth over - a 6-2 team
on a 4-game winning streak is meaningfully different from a 6-2 team
that has lost its last 2 games.

Produces (per team, computed from completed games prior to each matchup):

    TEAM_A_WINS          int     Wins in the current season to date
    TEAM_A_LOSSES        int     Losses in the current season to date
    TEAM_A_WIN_PCT       float   Wins / games played (NaN in week 1)
    TEAM_A_WIN_STREAK    int     Current consecutive wins (0 if not winning)
    TEAM_A_LOSS_STREAK   int     Current consecutive losses (0 if not losing)

    TEAM_B_*             Same five features for TEAM_B

Design notes:
    - All features are computed from games *prior to* the current matchup.
      The current game is never included in any aggregate - no leakage.
    - Ties (WIN_OR_TIE == 0.5) count as 0.5 wins and 0.5 losses, matching
      NFL standings convention. They reset both win and loss streaks to 0.
    - Week 1 games have no prior history: WINS=0, LOSSES=0, WIN_PCT=NaN,
      WIN_STREAK=0, LOSS_STREAK=0. _prepare_data excludes NaN rows, so
      week 1 is withheld from training via WIN_PCT.
    - WIN_STREAK and LOSS_STREAK are two separate non-negative columns
      rather than a single signed streak.
    - Neutral-site games (GAME_LOCATION == "N") are included in record
      computation because they affect standings equally.
    - Postseason games from prior seasons are excluded - features are
      reset each season (grouped by YEAR).

Implementation note (record/H1, record/H2):
    Counts are computed via cumsum + shift (vectorized over all teams).
    Streaks are computed via the standard "streak break / group / count"
    pattern: identify rows where the result type changes, take cumsum to
    label streak groups, then cumcount within each group. The result is
    shifted by one row per team-season so the value reflects the streak
    entering the current game.
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

logger: Logger = logging.getLogger(__name__)

_PRODUCES: Final[list[str]] = [
    "TEAM_A_WINS",
    "TEAM_A_LOSSES",
    "TEAM_A_WIN_PCT",
    "TEAM_A_WIN_STREAK",
    "TEAM_A_LOSS_STREAK",
    "TEAM_B_WINS",
    "TEAM_B_LOSSES",
    "TEAM_B_WIN_PCT",
    "TEAM_B_WIN_STREAK",
    "TEAM_B_LOSS_STREAK",
]

# Columns produced by _build_record_table for join-side use.
_STAT_COLS: Final[list[str]] = [
    "WINS",
    "LOSSES",
    "WIN_PCT",
    "WIN_STREAK",
    "LOSS_STREAK",
]


@FeatureRegistry.register("record")
class RecordFeature:
    """Season win/loss record and streak features for both teams.

    Computes cumulative wins, losses, win percentage, win streak, and
    loss streak for each team entering each game, using only results
    from prior games in the same season.
    """

    spec = FeatureSpec(name="record", produces=_PRODUCES)

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute season record and streak features and join onto df.

        Args:
            df: Modeling DataFrame with GAME_ID, TEAM_A, TEAM_B, YEAR,
                WEEK_NUM columns.
            datasets: Provides games() for historical results.

        Returns:
            Input DataFrame with 10 record/streak columns appended.
        """
        games: pd.DataFrame = datasets.games()
        record_table: DataFrame = _build_record_table(games)

        df = df.copy()
        df["WEEK_NUM"] = df["WEEK_NUM"].astype(int)

        if record_table.empty:
            # No completed games to learn from - produce default values.
            for col in _PRODUCES:
                if col.endswith("_WIN_PCT"):
                    df[col] = float("nan")
                else:
                    df[col] = 0
            return df

        # Vectorized join for both team perspectives. Two merges replace
        # 10 row-wise apply() calls (record/H1).
        for prefix, team_col in (("TEAM_A", "TEAM_A"), ("TEAM_B", "TEAM_B")):
            renamed: DataFrame = record_table.rename(
                columns={
                    "TEAM": team_col,
                    **{c: f"{prefix}_{c}" for c in _STAT_COLS},
                }
            )
            df = df.merge(
                renamed,
                how="left",
                on=[team_col, "YEAR", "WEEK_NUM"],
            )

        # Fill defaults for unmatched rows (week 1 entries / new teams /
        # rows that fall outside the completed-games window).
        for col in _PRODUCES:
            if col.endswith("_WIN_PCT"):
                # WIN_PCT NaN is the documented signal for "no prior games".
                continue
            df[col] = df[col].fillna(0)

        return df


# ---------------------------------------------------------------------------
# Record computation (vectorized)
# ---------------------------------------------------------------------------


def _build_record_table(games: pd.DataFrame) -> pd.DataFrame:
    """Build a vectorized record lookup table.

    Returns a DataFrame with one row per (TEAM, YEAR, WEEK_NUM) that
    captures the team's record *entering* that week. Empty DataFrame
    when input is empty or missing required columns.

    Columns:
        TEAM, YEAR, WEEK_NUM, WINS, LOSSES, WIN_PCT, WIN_STREAK, LOSS_STREAK
    """
    required: set[str] = {"WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"}
    if games.empty or not required.issubset(games.columns):
        return pd.DataFrame(columns=["TEAM", "YEAR", "WEEK_NUM", *_STAT_COLS])

    completed: DataFrame = games.loc[games["WIN_OR_TIE"].notna(), :].copy()
    completed["WEEK_NUM"] = completed["WEEK_NUM"].astype(int)

    # Long format: one row per team per game with the team's result.
    winner_rows = completed.loc[:, ["WINNER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].rename(
        columns={"WINNER": "TEAM", "WIN_OR_TIE": "RESULT"}
    )
    loser_rows = completed.loc[:, ["LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].copy()
    loser_rows["RESULT"] = 1.0 - loser_rows["WIN_OR_TIE"]
    loser_rows = loser_rows.rename(columns={"LOSER": "TEAM"}).drop(columns=["WIN_OR_TIE"])

    long: DataFrame = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long = long.sort_values(["TEAM", "YEAR", "WEEK_NUM"], ignore_index=True)

    # ── Count outcomes (record/H2) ────────────────────────────────────
    # Vectorized via cumsum + shift. shift(1) ensures the value at each
    # game reflects the team's record *entering* that game, not after it.
    long["IS_WIN"] = (long["RESULT"] >= 1.0).astype(int)
    long["IS_LOSS"] = (long["RESULT"] <= 0.0).astype(int)
    long["IS_TIE"] = (long["RESULT"] == 0.5).astype(int)

    grouped = long.groupby(["TEAM", "YEAR"], sort=False)

    cumulative_wins = (
        grouped["IS_WIN"].cumsum().groupby([long["TEAM"], long["YEAR"]]).shift(1).fillna(0)
    )
    cumulative_losses = (
        grouped["IS_LOSS"].cumsum().groupby([long["TEAM"], long["YEAR"]]).shift(1).fillna(0)
    )
    cumulative_ties = (
        grouped["IS_TIE"].cumsum().groupby([long["TEAM"], long["YEAR"]]).shift(1).fillna(0)
    )

    # NFL convention: each tie counts as 0.5 W and 0.5 L.
    long["WINS"] = cumulative_wins.astype(float) + 0.5 * cumulative_ties.astype(float)
    long["LOSSES"] = cumulative_losses.astype(float) + 0.5 * cumulative_ties.astype(float)

    n_played = cumulative_wins + cumulative_losses + cumulative_ties
    long["WIN_PCT"] = np.where(
        n_played > 0,
        long["WINS"] / n_played.replace(0, np.nan),
        np.nan,
    )

    # ── Streaks (record/H2) ───────────────────────────────────────────
    # Approach: label each row with a "streak type" (win / loss / tie),
    # detect streak boundaries (where the type changes), assign streak
    # group IDs via cumsum, then cumcount within each group. shift(1)
    # by team-season produces the streak entering the current game.
    long["STREAK_TYPE"] = np.where(
        long["IS_WIN"] == 1,
        "W",
        np.where(long["IS_LOSS"] == 1, "L", "T"),
    )

    # A streak break occurs whenever the streak type changes OR the
    # team-season changes (first game of a new team-season is always
    # a fresh streak).
    prev_streak_type = grouped["STREAK_TYPE"].shift(1)
    same_team_season = long["TEAM"].eq(long["TEAM"].shift(1)) & long["YEAR"].eq(
        long["YEAR"].shift(1)
    )
    streak_break = (long["STREAK_TYPE"] != prev_streak_type) | (~same_team_season)
    long["STREAK_GROUP"] = streak_break.cumsum()

    # cumcount within each streak group; the value at row i is the
    # streak length *including* the current game's result.
    long["STREAK_LEN_AFTER"] = long.groupby("STREAK_GROUP").cumcount() + 1

    # Shift by 1 within team-season so the value reflects the streak
    # entering the current game (the same shift we used for counts).
    long["STREAK_LEN_BEFORE"] = grouped["STREAK_LEN_AFTER"].shift(1).fillna(0).astype(int)
    long["STREAK_TYPE_BEFORE"] = grouped["STREAK_TYPE"].shift(1)

    # Split into separate WIN_STREAK and LOSS_STREAK columns.
    # Ties reset both streaks to 0 (matching the original behavior).
    long["WIN_STREAK"] = np.where(
        long["STREAK_TYPE_BEFORE"] == "W",
        long["STREAK_LEN_BEFORE"],
        0,
    ).astype(int)
    long["LOSS_STREAK"] = np.where(
        long["STREAK_TYPE_BEFORE"] == "L",
        long["STREAK_LEN_BEFORE"],
        0,
    ).astype(int)

    # Return only the columns needed for the join.
    return long.loc[:, ["TEAM", "YEAR", "WEEK_NUM", *_STAT_COLS]].reset_index(drop=True)
