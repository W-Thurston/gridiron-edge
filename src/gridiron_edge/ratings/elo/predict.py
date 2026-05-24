# src/gridiron_edge/ratings/elo/predict.py

"""Elo-based game predictions for upcoming schedule."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.core import elo_win_probability


def predict_elo_for_week(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Merge Elo onto the upcoming schedule and compute win probabilities.

    Args:
        year: NFL season label (e.g. ``"2026-2027"``).
        week: Week number to predict.
        repo: Repository root path. Defaults to ``repo_root()``.

    Returns:
        DataFrame with Elo ratings and win probability columns added,
        filtered to the requested year and week.
    """
    resolved_repo: Path = repo or repo_root()
    elo_path: Path = dataset_path(resolved_repo, "elo_state")
    schedule_path: Path = dataset_path(resolved_repo, "schedule_upcoming")

    df_elo: DataFrame = pd.read_csv(elo_path)
    df_schedule: DataFrame = pd.read_csv(schedule_path)
    df_schedule = df_schedule.loc[
        (df_schedule["YEAR"] == year) & (df_schedule["WEEK_NUM"] == week), :
    ].copy()

    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["AWAY_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "AWAY_TEAM_ELO"})
    )
    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["HOME_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "HOME_TEAM_ELO"})
    )

    df_schedule = df_schedule.dropna(subset=["AWAY_TEAM_ELO", "HOME_TEAM_ELO"])

    df_schedule[["AWAY_TEAM_WIN_PROB", "HOME_TEAM_WIN_PROB"]] = df_schedule.apply(
        lambda x: elo_win_probability(x["AWAY_TEAM_ELO"], x["HOME_TEAM_ELO"]),
        axis=1,
        result_type="expand",
    )
    df_schedule["AWAY_TEAM_WIN_PROB"] = df_schedule["AWAY_TEAM_WIN_PROB"].map(
        lambda x: f"{x * 100:.1f} %"
    )
    df_schedule["HOME_TEAM_WIN_PROB"] = df_schedule["HOME_TEAM_WIN_PROB"].map(
        lambda x: f"{x * 100:.1f} %"
    )
    return df_schedule.drop(columns=["YEAR"])


def predict_elo_only(*, year: str, week: int, repo: Path | None = None) -> Path:
    """Compute Elo predictions and write to a versioned CSV.

    Replaces the legacy Excel write. Writes to
    ``data/output/predictions/{year[:4]}/week_{week:02d}_predictions.csv``.

    Args:
        year: NFL season label (e.g. ``"2026-2027"``).
        week: Week number to predict.
        repo: Repository root path.

    Returns:
        Path to the written predictions CSV.
    """
    resolved_repo: Path = repo or repo_root()
    df: DataFrame = predict_elo_for_week(year=year, week=week, repo=resolved_repo)

    out_dir: Path = resolved_repo / "data" / "output" / "predictions" / year[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"week_{week:02d}_predictions.csv"
    df.to_csv(out_path, index=False)
    return out_path
