# src/gridiron_edge/evaluation/elo.py

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.core import elo_win_probability


def evaluate_elo(
    *,
    time_period: str = "YEAR",
    ranking_system: Callable[[float, float], tuple[float, float]] = elo_win_probability,
    repo: Path | None = None,
) -> None:
    """Print Elo pick accuracy aggregated by year or week.

    Args:
        time_period: Aggregation level — ``"YEAR"`` or ``"WEEK"``.
        ranking_system: Callable that accepts two Elo ratings and returns
            ``(win_probability, loss_probability)``. Defaults to the
            standard Elo win probability function.
        repo: Absolute path to the repository root. Defaults to the
            value returned by ``repo_root()``.
    """
    resolved_repo: Path = repo or repo_root()
    games_path = dataset_path(resolved_repo, "games")
    elo_path = dataset_path(resolved_repo, "elo_state")

    df = pd.read_csv(games_path)
    df = df.loc[:, ["WEEK_NUM", "WINNER", "LOSER", "YEAR", "WIN_OR_TIE"]]
    df_elo = pd.read_csv(elo_path)
    elo_prob: list[float] = []

    for row in tqdm(df.itertuples(), total=df.shape[0]):
        winning_team_name = row.WINNER
        losing_team_name = row.LOSER
        year = row.YEAR
        week = row.WEEK_NUM

        winner_elo = df_elo.loc[
            (df_elo["NFL_TEAM"] == winning_team_name)
            & (df_elo["NFL_YEAR"] == year)
            & (df_elo["NFL_WEEK"] == week),
            "ELO",
        ].values[0]
        loser_elo = df_elo.loc[
            (df_elo["NFL_TEAM"] == losing_team_name)
            & (df_elo["NFL_YEAR"] == year)
            & (df_elo["NFL_WEEK"] == week),
            "ELO",
        ].values[0]
        elo_prob.append(ranking_system(winner_elo, loser_elo)[0])

    elo_prob_series: pd.Series = pd.Series(elo_prob, index=df.index)
    df["ELO_PROB"] = elo_prob_series
    df.loc[(df["WIN_OR_TIE"] == 1) & (df["ELO_PROB"] > 0.5), "CORRECT"] = 1
    df["CORRECT"] = df["CORRECT"].fillna(0)

    if time_period == "YEAR":
        for i in df["YEAR"].unique():
            subset = df.loc[df["YEAR"] == i, :]
            print(
                f"{i}: {subset['CORRECT'].sum() / subset.shape[0]:.0%} correct on the season",
            )
    elif time_period == "WEEK":
        for i in df["WEEK_NUM"].unique():
            subset = df.loc[df["WEEK_NUM"] == i, :]
            print(
                f"{i:02}: {subset['CORRECT'].sum() / subset.shape[0]:.0%} "
                "correct for this week in season",
            )

    print(f"Overall: {df['CORRECT'].sum() / df.shape[0]:.0%}")
    print()
