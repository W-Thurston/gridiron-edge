# src/gridiron_edge/evaluation/opponent_allowed.py

"""Per-defense per-position aggregation of stats allowed.

Computes per-(opponent_team, position, stat_type, cohort) aggregates:
mean stat allowed, sample size, and rank within the (position,
stat_type, cohort) group.

Attribution: for each game, sum stat from all offensive players with
matching position. Then average across games to get per-defense value.

Cohorts (all within the latest NFL season):
    - season: all games
    - l4: defense's last 4 games (season-ordered)
    - home: games where the defense was home
    - away: games where the defense was away

Note: home/away is the DEFENSE's home/away. Since player_game_logs
`is_home` is the offensive player's perspective, defense home/away is
its inverse.

Persisted at data/output/props/opponent_allowed.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.evaluation.situational_splits import STAT_COLUMN_MAP

OPPONENT_ALLOWED_PATH: Final[str] = "data/output/props/opponent_allowed.parquet"

# Maps stat_type → position that the stat_type applies to.
STAT_POSITION_MAP: Final[dict[str, str]] = {
    "qb_pass_yards": "QB",
    "qb_rush_yards": "QB",
    "rb_rush_yards": "RB",
    "wr_rec_yards": "WR",
    "te_rec_yards": "TE",
}

COHORTS: Final[list[str]] = ["season", "l4", "home", "away"]


def compute_opponent_allowed(player_game_logs: DataFrame) -> DataFrame:
    """Compute per-defense per-position stat aggregations across 4 cohorts.

    For each (opponent_team, position, stat_type), computes cohort
    snapshots: season, l4 (last 4 games), home, away. Ranks within
    (position, stat_type, cohort) — 1 = stingiest.

    Home/away is the defense's perspective: derived by inverting the
    offensive player's ``is_home`` (a player home game is a defense
    away game).

    Args:
        player_game_logs: DataFrame with columns player_id, team,
            opponent_team, position, season, week, game_id, is_home,
            and stat columns (passing_yards, rushing_yards,
            receiving_yards).

    Returns:
        DataFrame: opponent_team, position, stat_type, cohort,
        avg_allowed, sample_size, rank_against_position. Empty if input
        is empty or missing required columns.
    """
    if player_game_logs.empty:
        return _empty_opponent_allowed_df()

    required_cols = {
        "player_id",
        "team",
        "opponent_team",
        "position",
        "season",
        "week",
        "game_id",
        "is_home",
    }
    if not required_cols.issubset(player_game_logs.columns):
        return _empty_opponent_allowed_df()

    latest_season = int(player_game_logs["season"].max())
    scope = player_game_logs.loc[player_game_logs["season"] == latest_season, :].copy()
    if scope.empty:
        return _empty_opponent_allowed_df()

    all_rows: list[DataFrame] = []

    for stat_type, stat_col in STAT_COLUMN_MAP.items():
        if stat_col not in scope.columns:
            continue

        position = STAT_POSITION_MAP.get(stat_type)
        if position is None:
            continue

        stat_data = scope.loc[scope["position"] == position, :]
        if stat_data.empty:
            continue

        # Sum per game: all players of `position` on the offense vs
        # this defense. is_home is constant within (opponent_team,
        # game_id) since the offense is fixed — carry it through.
        per_game = (
            stat_data.groupby(["opponent_team", "game_id", "week", "is_home"])[stat_col]
            .sum()
            .reset_index()
        )
        # Defense home/away = inverse of the offensive player's is_home.
        per_game["defense_is_home"] = ~per_game["is_home"].astype(bool)

        # season: all games.
        all_rows.append(
            _compute_cohort(
                per_game,
                stat_col=stat_col,
                cohort="season",
                stat_type=stat_type,
                position=position,
            )
        )

        # l4: defense's last 4 games by week.
        l4_input = (
            per_game.sort_values(["opponent_team", "week"])
            .groupby("opponent_team", group_keys=False)
            .tail(4)
        )
        all_rows.append(
            _compute_cohort(
                l4_input,
                stat_col=stat_col,
                cohort="l4",
                stat_type=stat_type,
                position=position,
            )
        )

        # home: defense was home.
        home_input = per_game.loc[per_game["defense_is_home"], :]
        all_rows.append(
            _compute_cohort(
                home_input,
                stat_col=stat_col,
                cohort="home",
                stat_type=stat_type,
                position=position,
            )
        )

        # away: defense was away.
        away_input = per_game.loc[~per_game["defense_is_home"], :]
        all_rows.append(
            _compute_cohort(
                away_input,
                stat_col=stat_col,
                cohort="away",
                stat_type=stat_type,
                position=position,
            )
        )

    if not all_rows:
        return _empty_opponent_allowed_df()

    return pd.concat(all_rows, ignore_index=True)


def write_opponent_allowed(df: DataFrame, repo: Path) -> Path:
    """Persist opponent-allowed DataFrame to Parquet.

    Args:
        df: DataFrame returned by ``compute_opponent_allowed``.
        repo: Repository root.

    Returns:
        Absolute path to the written artifact.
    """
    path = repo / OPPONENT_ALLOWED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_opponent_allowed(repo: Path) -> DataFrame:
    """Load the opponent-allowed artifact.

    Returns:
        DataFrame with the opponent-allowed schema, or empty DataFrame
        if no artifact exists.
    """
    path = repo / OPPONENT_ALLOWED_PATH
    if not path.exists():
        return _empty_opponent_allowed_df()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_cohort(
    per_game: DataFrame,
    *,
    stat_col: str,
    cohort: str,
    stat_type: str,
    position: str,
) -> DataFrame:
    """Compute per-defense average and rank for a cohort.

    Args:
        per_game: DataFrame with opponent_team, game_id, week, stat_col.
        stat_col: Column to average.
        cohort: 'season', 'l4', 'home', or 'away' — for output labeling.
        stat_type: Stat type — for output labeling.
        position: Position — for output labeling.

    Returns:
        DataFrame with opponent_team, position, stat_type, cohort,
        avg_allowed, sample_size, rank_against_position.
    """
    if per_game.empty:
        return _empty_opponent_allowed_df()

    agg = (
        per_game.groupby("opponent_team")[stat_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_allowed", "count": "sample_size"})
    )

    # Rank: 1 = stingiest (lowest avg_allowed).
    agg["rank_against_position"] = agg["avg_allowed"].rank(method="min", ascending=True).astype(int)

    agg["position"] = position
    agg["stat_type"] = stat_type
    agg["cohort"] = cohort

    return agg.loc[
        :,
        [
            "opponent_team",
            "position",
            "stat_type",
            "cohort",
            "avg_allowed",
            "sample_size",
            "rank_against_position",
        ],
    ].copy()


def _empty_opponent_allowed_df() -> DataFrame:
    """Empty DataFrame with the opponent-allowed schema."""
    return pd.DataFrame(
        columns=[
            "opponent_team",
            "position",
            "stat_type",
            "cohort",
            "avg_allowed",
            "sample_size",
            "rank_against_position",
        ]
    )
