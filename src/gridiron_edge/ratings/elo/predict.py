# src/gridiron_edge/ratings/elo/predict.py

from pathlib import Path

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.core import elo_win_probability
from gridiron_edge.viz.excel import write_predictions_sheet


def predict_elo_for_week(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Merge Elo onto upcoming schedule for a week and compute win probabilities.

    Returns the schedule dataframe (probabilities as formatted strings).
    """
    repo = repo or repo_root()
    elo_path = dataset_path(repo, "elo_state")
    schedule_path = dataset_path(repo, "schedule_upcoming")

    df_elo = pd.read_csv(elo_path)
    df_schedule = pd.read_csv(schedule_path)
    df_schedule = df_schedule.loc[
        (df_schedule["YEAR"] == year) & (df_schedule["WEEK_NUM"] == week),
        :,
    ]

    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["AWAY_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"], axis=1)
        .rename(
            columns={"ELO": "AWAY_TEAM_ELO"},
        )
    )
    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["HOME_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"], axis=1)
        .rename(
            columns={"ELO": "HOME_TEAM_ELO"},
        )
    )

    df_schedule.dropna(inplace=True)
    df_schedule[["AWAY_TEAM_WIN_PROB", "HOME_TEAM_WIN_PROB"]] = df_schedule.apply(
        lambda x: elo_win_probability(x.AWAY_TEAM_ELO, x.HOME_TEAM_ELO),
        axis=1,
        result_type="expand",
    )
    df_schedule["AWAY_TEAM_WIN_PROB"] = df_schedule["AWAY_TEAM_WIN_PROB"].apply(
        lambda x: f"{x * 100:.1f} %",
    )
    df_schedule["HOME_TEAM_WIN_PROB"] = df_schedule["HOME_TEAM_WIN_PROB"].apply(
        lambda x: f"{x * 100:.1f} %",
    )
    df_schedule.drop(["YEAR"], axis=1, inplace=True)
    return df_schedule


def predict_elo_only(*, year: str, week: int, repo: Path | None = None) -> None:
    """Write upcoming-week Elo predictions to the ranks Excel workbook."""
    settings = get_settings()
    df = predict_elo_for_week(year=year, week=week, repo=repo)
    excel_path = settings.ranks_excel
    write_predictions_sheet(df, excel_path=excel_path)
    print(f"> Saving Elo based predictions to: {excel_path}")
