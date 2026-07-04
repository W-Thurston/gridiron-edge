# src/gridiron_edge/evaluation/percentiles.py

"""Team percentile ranking computation.

Computes per-team percentile rankings for four stats (Elo rating,
avg_wins, make_playoffs, win_sb) at the current (season, week).
Percentiles are 0-1 scale, higher is better.

Consumed by the /teams, /teams/{abbr}, and /compare/teams API endpoints
via ``load_latest_team_percentiles``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

PERCENTILES_SUBDIR: Final[str] = "data/output/rankings/percentiles"


def compute_team_percentiles(
    elo_state: DataFrame,
    projections: DataFrame,
    long_to_short: dict[str, str],
) -> DataFrame:
    """Compute per-team percentile rankings for the latest (season, week).

    Rankings are descending — the team with the highest value on a stat
    gets the highest percentile. Percentile formula: ``(count - rank) / count``.
    Teams missing a stat value get NaN percentile for that stat.

    The Elo state stores long team names; projections use short codes.
    Both are converted to short codes for join and output.

    Args:
        elo_state: DataFrame with columns NFL_TEAM (long), NFL_YEAR,
            NFL_WEEK, ELO.
        projections: DataFrame with columns TEAM (short), AVG_WINS,
            P_MAKE_PLAYOFFS, P_WIN_SB, etc.
        long_to_short: Mapping from long team names to short codes.

    Returns:
        DataFrame with columns: team_abbr, season, week, rating_pct,
        avg_wins_pct, make_playoffs_pct, win_sb_pct. One row per team
        present in either input. Empty if both inputs are empty.
    """
    if elo_state.empty and projections.empty:
        return _empty_percentiles_df()

    # Determine latest (season, week) from Elo state.
    if elo_state.empty:
        # Projections-only: no season/week resolvable, return empty.
        return _empty_percentiles_df()

    latest_year = str(elo_state["NFL_YEAR"].max())
    year_rows = elo_state.loc[elo_state["NFL_YEAR"] == latest_year, :]
    latest_week = int(year_rows["NFL_WEEK"].max())

    # Current-week Elo, converted to short codes.
    current_elo = year_rows.loc[
        year_rows["NFL_WEEK"] == latest_week,
        ["NFL_TEAM", "ELO"],
    ].copy()
    current_elo["team_abbr"] = (
        current_elo["NFL_TEAM"].map(long_to_short).fillna(current_elo["NFL_TEAM"])
    )
    current_elo = current_elo[["team_abbr", "ELO"]].rename(columns={"ELO": "rating"})

    # Prepare projections with short codes as team_abbr.
    if projections.empty:
        proj = pd.DataFrame(columns=["team_abbr"])
    else:
        proj = projections.rename(columns={"TEAM": "team_abbr"})

    # Outer merge — keep any team present in either source.
    merged = current_elo.merge(proj, on="team_abbr", how="outer")

    # Compute percentiles per stat. Higher raw value → higher percentile.
    merged["rating_pct"] = _percentile_rank(merged["rating"])
    merged["avg_wins_pct"] = _percentile_rank(merged.get("AVG_WINS"))
    merged["make_playoffs_pct"] = _percentile_rank(merged.get("P_MAKE_PLAYOFFS"))
    merged["win_sb_pct"] = _percentile_rank(merged.get("P_WIN_SB"))

    merged["season"] = latest_year
    merged["week"] = latest_week

    return merged.loc[
        :,
        [
            "team_abbr",
            "season",
            "week",
            "rating_pct",
            "avg_wins_pct",
            "make_playoffs_pct",
            "win_sb_pct",
        ],
    ].copy()


def write_team_percentiles(
    df: DataFrame,
    *,
    season: str,
    week: int,
    repo: Path,
) -> Path:
    """Persist a percentile DataFrame to the versioned artifact path.

    Filename: ``percentiles_{season}_wk{NN}.parquet``. Same (season, week)
    overwrites on repeat — natural dedup by week.

    Args:
        df: DataFrame returned by ``compute_team_percentiles``.
        season: Season label, e.g. "2026-2027".
        week: NFL week number.
        repo: Repository root.

    Returns:
        Absolute path to the written artifact.
    """
    out_dir = repo / PERCENTILES_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"percentiles_{season}_wk{week:02d}.parquet"
    path = out_dir / filename
    df.to_parquet(path, index=False)
    return path


def load_latest_team_percentiles(repo: Path) -> DataFrame:
    """Load the most recent percentile artifact.

    Returns empty DataFrame with the percentile schema if none exist.

    Args:
        repo: Repository root.

    Returns:
        DataFrame with the percentile schema, or empty DataFrame if no
        artifact exists yet.
    """
    percentiles_dir = repo / PERCENTILES_SUBDIR
    if not percentiles_dir.exists():
        return _empty_percentiles_df()

    files = sorted(percentiles_dir.glob("percentiles_*.parquet"))
    if not files:
        return _empty_percentiles_df()

    return pd.read_parquet(files[-1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile_rank(series: pd.Series | None) -> pd.Series:
    """Compute descending percentile rank as ``(count - rank) / count``.

    NaN values are excluded from ranking and remain NaN in output.
    A missing series (None) returns a Series of all NaN.
    """
    if series is None:
        return pd.Series(dtype=float)

    ranks = series.rank(ascending=False, method="min", na_option="keep")
    count = series.notna().sum()
    if count == 0:
        return pd.Series([float("nan")] * len(series), index=series.index)
    return (count - ranks) / count


def _empty_percentiles_df() -> DataFrame:
    """Empty DataFrame with the percentile artifact schema."""
    return pd.DataFrame(
        columns=[
            "team_abbr",
            "season",
            "week",
            "rating_pct",
            "avg_wins_pct",
            "make_playoffs_pct",
            "win_sb_pct",
        ]
    )
