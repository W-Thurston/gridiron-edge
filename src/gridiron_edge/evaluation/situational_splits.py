# src/gridiron_edge/evaluation/situational_splits.py

"""Per-player situational splits computation and persistence.

Computes per-(player_id, stat_type, cohort) splits by joining player
game logs to the games CSV on game_id, then partitioning by cohort:

    - season: all games (full sample)
    - home / away: based on GAME_LOCATION
    - favored / underdog: based on FAVORITED
    - indoor / outdoor: based on ROOF
    - l4: last 4 games (season/week ordered)

Produces DataFrame with columns:
    player_id, cohort, sample_size, mean_value

Persisted per-stat-type at
data/output/props/situational_splits/{stat_type}.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

SITUATIONAL_SPLITS_SUBDIR: Final[str] = "data/output/props/situational_splits"

# Maps stat_type → column in player_game_logs to aggregate.
STAT_COLUMN_MAP: Final[dict[str, str]] = {
    "qb_pass_yards": "passing_yards",
    "qb_rush_yards": "rushing_yards",
    "rb_rush_yards": "rushing_yards",
    "wr_rec_yards": "receiving_yards",
    "te_rec_yards": "receiving_yards",
}

COHORTS: Final[list[str]] = [
    "season",
    "home",
    "away",
    "favored",
    "underdog",
    "indoor",
    "outdoor",
    "l4",
]


def compute_player_situational_splits(
    player_game_logs: DataFrame,
    games: DataFrame,
    long_to_short: dict[str, str],
    stat_type: str,
) -> DataFrame:
    """Compute per-player situational splits for a stat_type.

    Joins player_game_logs to games on game_id to attach context flags,
    partitions by cohort, computes sample_size + mean_value per
    (player_id, cohort).

    Args:
        player_game_logs: DataFrame with columns player_id, team, game_id,
            season, week, and the stat column (e.g. passing_yards).
        games: DataFrame with columns GAME_ID, GAME_LOCATION, WINNER,
            LOSER, ROOF, VEGAS_LINE, FAVORITED.
        long_to_short: Mapping from long team names to short codes.
        stat_type: Which stat family (must be in STAT_COLUMN_MAP).

    Returns:
        DataFrame with columns player_id, cohort, sample_size, mean_value.
        Empty if inputs are empty or stat_type is unrecognized.
    """
    if stat_type not in STAT_COLUMN_MAP:
        return _empty_splits_df()

    stat_col: str = STAT_COLUMN_MAP[stat_type]

    if player_game_logs.empty or games.empty:
        return _empty_splits_df()

    if stat_col not in player_game_logs.columns:
        return _empty_splits_df()

    # Join player game logs to games on game_id.
    joined = player_game_logs.merge(
        games[["GAME_ID", "GAME_LOCATION", "WINNER", "LOSER", "ROOF", "FAVORITED"]],
        left_on="game_id",
        right_on="GAME_ID",
        how="inner",
    )

    if joined.empty:
        return _empty_splits_df()

    # Attach cohort membership as boolean columns.
    joined = _attach_cohort_flags(joined, long_to_short)

    # Aggregate per (player_id, cohort). One pass — build a long-format
    # DataFrame with cohort labels.
    all_rows: list[DataFrame] = []
    for cohort in COHORTS:
        if cohort == "season":
            subset = joined
        elif cohort == "l4":
            # Last 4 games per player, sorted by (season, week).
            subset = (
                joined.sort_values(["player_id", "season", "week"])
                .groupby("player_id", group_keys=False)
                .tail(4)
            )
        else:
            flag_col = f"is_{cohort}"
            if flag_col not in joined.columns:
                continue
            subset = joined.loc[joined[flag_col], :]

        if subset.empty:
            continue

        agg = (
            subset.groupby("player_id")[stat_col]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "sample_size", "mean": "mean_value"})
        )
        agg["cohort"] = cohort
        # pyrefly: ignore [bad-argument-type]
        all_rows.append(agg[["player_id", "cohort", "sample_size", "mean_value"]])

    if not all_rows:
        return _empty_splits_df()

    return pd.concat(all_rows, ignore_index=True)


def write_situational_splits(
    df: DataFrame,
    stat_type: str,
    repo: Path,
) -> Path:
    """Persist a splits DataFrame to per-stat-type Parquet.

    Filename: ``{stat_type}.parquet``. Same stat_type overwrites on
    repeat — natural dedup by stat.

    Args:
        df: DataFrame returned by ``compute_player_situational_splits``.
        stat_type: Stat family (e.g. ``"qb_pass_yards"``).
        repo: Repository root.

    Returns:
        Absolute path to the written artifact.
    """
    out_dir = repo / SITUATIONAL_SPLITS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{stat_type}.parquet"
    path = out_dir / filename
    df.to_parquet(path, index=False)
    return path


def load_situational_splits(stat_type: str, repo: Path) -> DataFrame:
    """Load the situational splits artifact for a stat_type.

    Returns:
        DataFrame with the splits schema, or empty DataFrame if no
        artifact exists for the stat_type.
    """
    path = repo / SITUATIONAL_SPLITS_SUBDIR / f"{stat_type}.parquet"
    if not path.exists():
        return _empty_splits_df()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_cohort_flags(
    df: DataFrame,
    long_to_short: dict[str, str],
) -> DataFrame:
    """Attach boolean cohort flags per row.

    Uses the joined DataFrame's columns to compute:
        - is_home: player's team was home (GAME_LOCATION == 'H' + team matches)
        - is_away: opposite of is_home
        - is_favored: player's team is in FAVORITED
        - is_underdog: player's team is NOT in FAVORITED (but game had a favorite)
        - is_indoor: ROOF is dome-like
        - is_outdoor: ROOF is outdoor-like
    """
    df = df.copy()

    # Derive player_team_long from short code.
    short_to_long = {v: k for k, v in long_to_short.items()}
    df["player_team_long"] = df["team"].map(short_to_long).fillna(df["team"])

    # is_home / is_away.
    # GAME_LOCATION: "H" = winner played at home. "@" = winner played on road.
    # We need to determine whether the player's team was home for THIS game.
    # Home team = WINNER if GAME_LOCATION == "H", else LOSER.
    home_team_long = df["WINNER"].where(df["GAME_LOCATION"] == "H", df["LOSER"])
    df["is_home"] = df["player_team_long"] == home_team_long
    df["is_away"] = ~df["is_home"]

    # is_favored / is_underdog.
    # FAVORITED = long team name of who was favored. NaN if no clear favorite.
    df["is_favored"] = df["FAVORITED"].notna() & (df["player_team_long"] == df["FAVORITED"])
    df["is_underdog"] = df["FAVORITED"].notna() & (df["player_team_long"] != df["FAVORITED"])

    # is_indoor / is_outdoor.
    # ROOF values: 'outdoors', 'dome', 'closed', 'open'.
    # Indoor if ROOF is 'dome' or 'closed'.
    df["is_indoor"] = df["ROOF"].isin(["dome", "closed"])
    df["is_outdoor"] = df["ROOF"].isin(["outdoors", "open"])

    return df


def _empty_splits_df() -> DataFrame:
    """Empty DataFrame with the situational splits schema."""
    return pd.DataFrame(columns=["player_id", "cohort", "sample_size", "mean_value"])
