# src/gridiron_edge/features/team/record.py

"""Season win/loss record and current streak features.

Captures team momentum and form signals that Elo absorbs only slowly.
A team's win percentage and current streak encode information about
recent trajectory that aggregate Elo ratings smooth over — a 6-2 team
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
      The current game is never included in any aggregate — no leakage.
    - Ties (WIN_OR_TIE == 0.5) count as 0.5 wins and 0.5 losses, matching
      NFL standings convention. They reset both win and loss streaks to 0.
    - Week 1 games have no prior history: WINS=0, LOSSES=0, WIN_PCT=NaN,
      WIN_STREAK=0, LOSS_STREAK=0. _prepare_data excludes NaN rows, so
      week 1 is withheld from training via WIN_PCT. Alternatively, impute
      WIN_PCT=0.5 for week 1 if you want those rows included — a design
      choice tracked in the backlog.
    - WIN_STREAK and LOSS_STREAK are two separate non-negative columns
      rather than a single signed streak. This lets the model split on
      WIN_STREAK > 3 independently of LOSS_STREAK > 2 without sign
      conflation — cleaner for tree-based models.
    - Neutral-site games (GAME_LOCATION == "N") are included in record
      computation because they affect standings equally.
    - Postseason games from prior seasons are excluded — features are
      reset each season (grouped by YEAR).
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final, Literal

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
        record_map: dict[tuple[str, str, int], dict[str, float]] = _build_record_map(games)

        df = df.copy()
        for prefix, team_col in [("TEAM_A", "TEAM_A"), ("TEAM_B", "TEAM_B")]:
            for stat in ["WINS", "LOSSES", "WIN_PCT", "WIN_STREAK", "LOSS_STREAK"]:
                col: str = f"{prefix}_{stat}"
                df[col] = df.apply(
                    lambda row, tc=team_col, s=stat: record_map.get(
                        (row[tc], row["YEAR"], int(row["WEEK_NUM"])), {}
                    ).get(s, float("nan") if s == "WIN_PCT" else 0),
                    axis=1,
                )

        return df


# ---------------------------------------------------------------------------
# Record computation
# ---------------------------------------------------------------------------


def _build_record_map(
    games: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, float]]:
    """Build a lookup: (team, year, week) → record stats entering that week.

    For each team/season/week combination, computes the team's record
    (wins, losses, win_pct, win_streak, loss_streak) based solely on
    completed games in the same season with WEEK_NUM < current week.

    Args:
        games: Canonical games DataFrame with WINNER, LOSER, YEAR,
            WEEK_NUM, and WIN_OR_TIE columns.

    Returns:
        Dict mapping (team, year, week_num) to a dict of stat values.
        Week 1 entries have zeros for all counts and NaN for WIN_PCT.
    """
    required: set[str] = {"WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"}
    if games.empty or not required.issubset(games.columns):
        return {}

    # Work only with completed games (WIN_OR_TIE is non-null)
    completed: DataFrame = games.loc[games["WIN_OR_TIE"].notna(), :].copy()
    completed["WEEK_NUM"] = completed["WEEK_NUM"].astype(int)

    # Build a long-form table: one row per team per game
    # Each row: team, year, week, result (1=win, 0.5=tie, 0=loss)
    winner_rows = completed.loc[:, ["WINNER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].rename(
        columns={"WINNER": "TEAM", "WIN_OR_TIE": "RESULT"}
    )
    loser_rows = completed.loc[:, ["LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].copy()
    loser_rows["RESULT"] = 1.0 - loser_rows["WIN_OR_TIE"]
    loser_rows = loser_rows.rename(columns={"LOSER": "TEAM"}).drop(columns=["WIN_OR_TIE"])

    long = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long = long.sort_values(["TEAM", "YEAR", "WEEK_NUM"], ignore_index=True)

    record_map: dict[tuple[str, str, int], dict[str, float]] = {}

    for (team, year), group in long.groupby(["TEAM", "YEAR"], sort=False):
        sorted_group = group.sort_values("WEEK_NUM")
        results: list[float] = sorted_group["RESULT"].tolist()
        weeks: list[int] = sorted_group["WEEK_NUM"].tolist()

        # For each game week, compute record from all prior games this season
        for i, week in enumerate(weeks):
            prior: list[float] = results[:i]  # games before this week only

            wins: Literal[0] | float = sum(r for r in prior if r >= 1.0)
            losses: Literal[0] | float = sum(1 - r for r in prior if r <= 0.0)
            # Ties: result == 0.5 — count 0.5 toward wins and losses
            ties_contrib: Literal[0] | float = sum(0.5 for r in prior if r == 0.5)
            wins += ties_contrib
            losses += ties_contrib

            n_played: int = len(prior)
            win_pct: float = wins / n_played if n_played > 0 else float("nan")

            # Streak: count consecutive wins/losses from the most recent game
            win_streak = 0
            loss_streak = 0
            for r in reversed(prior):
                if r >= 1.0:
                    if loss_streak == 0:
                        win_streak += 1
                    else:
                        break
                elif r <= 0.0:
                    if win_streak == 0:
                        loss_streak += 1
                    else:
                        break
                else:
                    # Tie — resets both streaks
                    break

            record_map[(str(team), str(year), week)] = {
                "WINS": wins,
                "LOSSES": losses,
                "WIN_PCT": win_pct,
                "WIN_STREAK": float(win_streak),
                "LOSS_STREAK": float(loss_streak),
            }

    return record_map
