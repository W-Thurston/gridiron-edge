# src/gridiron_edge/evaluation/team_cohort_splits.py

"""Per-team cohort splits computation and persistence.

Computes per-(team, cohort) aggregations of 8 metrics from
epa_by_game.parquet, across 4 cohorts (season, l4, home, away),
for the current season.

Produces DataFrame with columns:
    team_abbr, cohort,
    off_epa_per_play, off_pass_epa, off_rush_epa,
    def_epa_per_play, def_rush_epa,
    off_third_down_pct, off_redzone_td_pct,
    turnover_diff, sample_size,
    rank_off_epa_per_play, rank_off_pass_epa, rank_off_rush_epa,
    rank_def_epa_per_play, rank_def_rush_epa,
    rank_off_third_down_pct, rank_off_redzone_td_pct,
    rank_turnover_diff

Ranks: 1 = best in cohort for that metric. Off metrics and
turnover_diff: rank 1 = highest. Def metrics: rank 1 = lowest
(stingiest).

Persisted at data/output/rankings/team_cohort_splits.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

TEAM_COHORT_SPLITS_PATH: Final[str] = "data/output/rankings/team_cohort_splits.parquet"

COHORTS: Final[list[str]] = ["season", "l4", "home", "away"]

# Metrics with their aggregation source column and rank direction.
# rank_direction=asc: rank 1 = lowest (stingiest for defense).
# rank_direction=desc: rank 1 = highest (best for offense).
METRICS: Final[dict[str, str]] = {
    "off_epa_per_play": "desc",
    "off_pass_epa": "desc",
    "off_rush_epa": "desc",
    "def_epa_per_play": "asc",
    "def_rush_epa": "asc",
    "off_third_down_pct": "desc",
    "off_redzone_td_pct": "desc",
    "turnover_diff": "desc",
}


def compute_team_cohort_splits(
    epa_df: DataFrame,
    long_to_short: dict[str, str],
) -> DataFrame:
    """Compute per-team cohort splits for the current season.

    Args:
        epa_df: DataFrame with EPA per game per team. Long team names.
        long_to_short: Mapping from long team names to short codes.

    Returns:
        DataFrame with team_abbr, cohort, 8 metric columns, sample_size,
        and 8 rank columns. Empty if input is empty or missing required
        columns.
    """
    if epa_df.empty:
        return _empty_splits_df()

    required_metrics = {
        "off_epa_per_play",
        "off_pass_epa",
        "off_rush_epa",
        "def_epa_per_play",
        "def_rush_epa",
        "off_third_down_pct",
        "off_redzone_td_pct",
        "off_turnover_rate",
        "def_turnover_rate",
    }
    required_meta = {"game_id", "season", "week", "team"}

    if not required_metrics.issubset(epa_df.columns):
        return _empty_splits_df()
    if not required_meta.issubset(epa_df.columns):
        return _empty_splits_df()

    # Filter to latest season only.
    latest_season = int(epa_df["season"].max())
    scope = epa_df.loc[epa_df["season"] == latest_season, :].copy()

    if scope.empty:
        return _empty_splits_df()

    # Convert long team names to short codes.
    scope["team_abbr"] = scope["team"].map(long_to_short).fillna(scope["team"])

    # Compute turnover_diff = off_turnover_rate - def_turnover_rate.
    scope["turnover_diff"] = scope["off_turnover_rate"] - scope["def_turnover_rate"]

    # Determine home/away for each row from game_id parsing.
    # game_id format: "YYYY_WW_AWAY_HOME" → team is home if team_abbr matches HOME part.
    scope["is_home"] = scope.apply(_is_home_team, axis=1)

    # Sort within team by (season, week) for l4 slicing later.
    scope = scope.sort_values(["team_abbr", "season", "week"]).reset_index(drop=True)

    # Compute per-cohort aggregates.
    all_rows: list[DataFrame] = []

    for cohort in COHORTS:
        subset = _select_cohort(scope, cohort)
        if subset.empty:
            continue

        agg = _aggregate_metrics(subset)
        agg["cohort"] = cohort
        all_rows.append(agg)

    if not all_rows:
        return _empty_splits_df()

    df = pd.concat(all_rows, ignore_index=True)

    # Add rank columns per (cohort, metric).
    df = _add_rank_columns(df)

    return df


def write_team_cohort_splits(df: DataFrame, repo: Path) -> Path:
    """Persist team cohort splits DataFrame to Parquet.

    Args:
        df: DataFrame returned by ``compute_team_cohort_splits``.
        repo: Repository root.

    Returns:
        Absolute path to the written artifact.
    """
    path = repo / TEAM_COHORT_SPLITS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_team_cohort_splits(repo: Path) -> DataFrame:
    """Load the team cohort splits artifact.

    Returns:
        DataFrame with the splits schema, or empty DataFrame if no
        artifact exists.
    """
    path = repo / TEAM_COHORT_SPLITS_PATH
    if not path.exists():
        return _empty_splits_df()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_home_team(row: pd.Series) -> bool:
    """Determine if a row's team was the home team for its game.

    game_id format: "YYYY_WW_AWAY_HOME". Team is home if team_abbr
    matches the HOME part.
    """
    game_id = str(row["game_id"])
    team_abbr = str(row["team_abbr"])
    parts = game_id.split("_")
    if len(parts) != 4:
        return False
    _, _, _, home_team = parts
    return team_abbr == home_team


def _select_cohort(df: DataFrame, cohort: str) -> DataFrame:
    """Filter the DataFrame to rows matching the given cohort."""
    if cohort == "season":
        return df
    elif cohort == "home":
        return df.loc[df["is_home"], :]
    elif cohort == "away":
        return df.loc[~df["is_home"], :]
    elif cohort == "l4":
        # Last 4 games per team, ordered by (season, week).
        return (
            df.sort_values(["team_abbr", "season", "week"])
            .groupby("team_abbr", group_keys=False)
            .tail(4)
        )
    else:
        # pyrefly: ignore [bad-return]
        return df.iloc[:0]  # empty


def _aggregate_metrics(df: DataFrame) -> DataFrame:
    """Aggregate metrics per team from a cohort subset."""
    # Compute means for each metric, plus a count for sample_size.
    grouped = df.groupby("team_abbr")

    agg = grouped[list(METRICS.keys())].mean().reset_index()

    # Sample size = count of games in cohort per team.
    sample_size = grouped.size().rename("sample_size").reset_index()
    agg = agg.merge(sample_size, on="team_abbr", how="left")

    return agg.loc[:, ["team_abbr", *list(METRICS.keys()), "sample_size"]]


def _add_rank_columns(df: DataFrame) -> DataFrame:
    """Add rank columns per (cohort, metric).

    Off metrics + turnover_diff: rank 1 = highest.
    Def metrics: rank 1 = lowest.
    """
    df = df.copy()

    for metric, direction in METRICS.items():
        rank_col = f"rank_{metric}"
        # Rank within each cohort separately.
        ascending = direction == "asc"
        df[rank_col] = (
            df.groupby("cohort")[metric].rank(method="min", ascending=ascending).astype("Int64")
        )

    return df


def _empty_splits_df() -> DataFrame:
    """Empty DataFrame with the team cohort splits schema."""
    metric_cols = list(METRICS.keys())
    rank_cols = [f"rank_{m}" for m in metric_cols]

    return pd.DataFrame(
        columns=[
            "team_abbr",
            "cohort",
            *metric_cols,
            "sample_size",
            *rank_cols,
        ]
    )
