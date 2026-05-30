# src/gridiron_edge/features/team/schedule_strength.py

"""Strength of schedule and strength of victory features.

Measures opponent quality to contextualise a team's record. A team's raw
win-loss record does not distinguish between a schedule of weak opponents
and one of strong opponents — these features provide that context.

Both features use opponent Elo as the quality proxy. Elo is already
computed, trusted, and captures opponent strength in a single number
without introducing additional multicollinearity with the EPA features.
EPA-based opponent quality measures are a potential future extension
tracked in the backlog.

Produces (per team, computed from completed games prior to each matchup):

    TEAM_A_SOS    float   Average pre-game Elo of all opponents faced to
                          date this season. NaN if no games played yet (week 1).

    TEAM_A_SOV    float   Average pre-game Elo of opponents that the team
                          has beaten to date this season. NaN if no wins yet.

    TEAM_B_SOS    float   Same for TEAM_B.
    TEAM_B_SOV    float   Same for TEAM_B.

Design notes:
    - "Pre-game Elo" means the opponent's Elo *entering* the game they
      played against this team — not the opponent's current Elo. This
      avoids future leakage: an opponent that improved dramatically after
      the game does not retroactively make that game look harder.
    - The Elo state table (datasets.elo_state()) provides pre-game Elo
      keyed by (NFL_TEAM, NFL_YEAR, NFL_WEEK). This is the same table
      used by the team_elo feature.
    - Week 1: no prior opponents → SOS = NaN, SOV = NaN. These rows are
      excluded from training by _prepare_data's NaN filter.
    - Teams with wins but no losses (e.g. 4-0) still get a valid SOS
      (average of all 4 opponents' Elos) and SOV (average of those same
      4 opponents, since all games were wins).
    - Ties (WIN_OR_TIE == 0.5) are excluded from SOV — only outright
      wins count toward strength of victory, consistent with NFL
      tiebreaker convention.
    - Postseason games are included in opponent Elo lookups but the
      features reset each season (YEAR grouping).
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

logger: Logger = logging.getLogger(__name__)

_PRODUCES: Final[list[str]] = [
    "TEAM_A_SOS",
    "TEAM_A_SOV",
    "TEAM_B_SOS",
    "TEAM_B_SOV",
]


@FeatureRegistry.register("schedule_strength")
class ScheduleStrengthFeature:
    """Strength of schedule and strength of victory via opponent Elo.

    Computes the average pre-game Elo of all opponents faced (SOS) and
    of opponents beaten (SOV) for each team entering each game, using
    only results from prior games in the same season.
    """

    spec = FeatureSpec(name="schedule_strength", produces=_PRODUCES, depends_on=("team_elo",))

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute SOS and SOV features and join onto df.

        Args:
            df: Modeling DataFrame with GAME_ID, TEAM_A, TEAM_B, YEAR,
                WEEK_NUM columns.
            datasets: Provides games() and elo_state() datasets.

        Returns:
            Input DataFrame with TEAM_A_SOS, TEAM_A_SOV, TEAM_B_SOS,
            TEAM_B_SOV columns appended.
        """
        games: pd.DataFrame = datasets.games()
        elo: pd.DataFrame = datasets.elo_state()

        sos_sov_map: dict[tuple[str, str, int], dict[str, float]] = _build_sos_sov_map(games, elo)

        df = df.copy()
        for prefix, team_col in [("TEAM_A", "TEAM_A"), ("TEAM_B", "TEAM_B")]:
            for stat in ["SOS", "SOV"]:
                col: str = f"{prefix}_{stat}"
                df[col] = df.apply(
                    lambda row, tc=team_col, s=stat: sos_sov_map.get(
                        (row[tc], row["YEAR"], int(row["WEEK_NUM"])), {}
                    ).get(s, float("nan")),
                    axis=1,
                )

        return df


# ---------------------------------------------------------------------------
# SOS / SOV computation
# ---------------------------------------------------------------------------


def _build_sos_sov_map(
    games: pd.DataFrame,
    elo: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, float]]:
    """Build a lookup: (team, year, week) → {SOS, SOV} entering that week.

    Args:
        games: Canonical games DataFrame with WINNER, LOSER, YEAR,
            WEEK_NUM, WIN_OR_TIE columns.
        elo: Elo state table with NFL_TEAM, NFL_YEAR, NFL_WEEK, ELO columns.

    Returns:
        Dict mapping (team, year, week_num) to {"SOS": float, "SOV": float}.
        SOS or SOV are NaN when there are no applicable prior games.
    """
    required_games: set[str] = {"WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"}
    required_elo: set[str] = {"NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"}
    if (
        games.empty
        or elo.empty
        or not required_games.issubset(games.columns)
        or not required_elo.issubset(elo.columns)
    ):
        return {}

    completed: DataFrame = games.loc[games["WIN_OR_TIE"].notna(), :].copy()
    completed["WEEK_NUM"] = completed["WEEK_NUM"].astype(int)

    # Build long-form: one row per team per game with opponent name and result
    # Winner perspective: opponent = LOSER, result = WIN_OR_TIE (1.0 or 0.5)
    winner_rows = completed.loc[:, ["WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].rename(
        columns={"WINNER": "TEAM", "LOSER": "OPPONENT", "WIN_OR_TIE": "RESULT"}
    )
    # Loser perspective: opponent = WINNER, result = 1 - WIN_OR_TIE
    loser_rows = completed.loc[:, ["LOSER", "WINNER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].copy()
    loser_rows["RESULT"] = 1.0 - loser_rows["WIN_OR_TIE"]
    loser_rows = loser_rows.rename(columns={"LOSER": "TEAM", "WINNER": "OPPONENT"}).drop(
        columns=["WIN_OR_TIE"]
    )

    long = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long = long.sort_values(["TEAM", "YEAR", "WEEK_NUM"], ignore_index=True)

    # Build Elo lookup: (team, year, week) → pre-game Elo
    elo_lookup: dict[tuple[str, str, int], float] = {
        (str(row["NFL_TEAM"]), str(row["NFL_YEAR"]), int(row["NFL_WEEK"])): float(row["ELO"])
        for _, row in elo.iterrows()
    }

    sos_sov_map: dict[tuple[str, str, int], dict[str, float]] = {}

    for (team, year), group in long.groupby(["TEAM", "YEAR"], sort=False):
        sorted_group = group.sort_values("WEEK_NUM").reset_index(drop=True)
        weeks: list[int] = sorted_group["WEEK_NUM"].tolist()
        opponents: list[str] = sorted_group["OPPONENT"].astype(str).tolist()
        results: list[float] = sorted_group["RESULT"].tolist()

        for i, week in enumerate(weeks):
            # Only games prior to this week in the same season
            prior_opps: list[str] = opponents[:i]
            prior_results: list[float] = results[:i]
            prior_weeks: list[int] = weeks[:i]

            opp_elos: list[float] = []
            win_elos: list[float] = []

            for j, opp in enumerate(prior_opps):
                opp_week: int = prior_weeks[j]
                opp_elo: float = elo_lookup.get((opp, str(year), opp_week), float("nan"))
                if not pd.isna(opp_elo):
                    opp_elos.append(opp_elo)
                    if prior_results[j] >= 1.0:  # outright win only for SOV
                        win_elos.append(opp_elo)

            sos: float = sum(opp_elos) / len(opp_elos) if opp_elos else float("nan")
            sov: float = sum(win_elos) / len(win_elos) if win_elos else float("nan")

            sos_sov_map[(str(team), str(year), week)] = {
                "SOS": sos,
                "SOV": sov,
            }

    return sos_sov_map
