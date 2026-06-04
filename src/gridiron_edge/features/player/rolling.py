# src/gridiron_edge/features/player/rolling.py

"""Per-player rolling statistics for prop model features.

Computes rolling mean and standard deviation over configurable windows
(default L3 and L6 games) for key stat columns. All rolling computations
use ``shift(1)`` to prevent lookahead leakage — a player's rolling stats
for week N reflect only games through week N-1.

Rolling windows operate on game count (not week number), so bye weeks
are handled naturally. Windows do NOT cross season boundaries by default.

Usage::

    from gridiron_edge.features.player.rolling import build_player_rolling_features

    df = build_player_rolling_features()
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Default rolling windows: short-term form and medium-term baseline.
DEFAULT_WINDOWS: Final[list[int]] = [3, 6]

# Stats to compute rolling features for, grouped by position relevance.
# All stats get both rolling mean and rolling std dev.

# Passing stats (QB)
_PASSING_STATS: Final[list[str]] = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "attempts",
    "completions",
    "passing_air_yards",
    "passing_epa",
    "passing_cpoe",
    "sacks_suffered",
]

# Rushing stats (RB, QB)
_RUSHING_STATS: Final[list[str]] = [
    "rushing_yards",
    "rushing_tds",
    "carries",
    "rushing_epa",
    "rushing_fumbles",
]

# Receiving stats (WR, TE, RB)
_RECEIVING_STATS: Final[list[str]] = [
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "targets",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_epa",
    "target_share",
    "air_yards_share",
]

# All stat columns that get rolling features.
ROLLING_STAT_COLS: Final[list[str]] = _PASSING_STATS + _RUSHING_STATS + _RECEIVING_STATS


def _compute_rolling(
    df: DataFrame,
    *,
    windows: list[int],
    cross_season: bool = False,
) -> DataFrame:
    """Compute shifted rolling mean and std for each player.

    Args:
        df: Player game logs sorted by (player_id, season, week).
        windows: List of rolling window sizes (e.g. [3, 6]).
        cross_season: If ``False`` (default), rolling windows reset
            at season boundaries. If ``True``, windows span across
            seasons (useful for capturing career-level trends).

    Returns:
        DataFrame with original columns plus rolling feature columns
        named ``{stat}_L{window}_mean`` and ``{stat}_L{window}_std``.
    """
    group_cols: list[str] = ["player_id"] if cross_season else ["player_id", "season"]

    # Sort within groups
    df = df.sort_values(["player_id", "season", "week"]).copy()

    # Only compute for stat columns that exist in the data
    available_stats: list[str] = [c for c in ROLLING_STAT_COLS if c in df.columns]
    missing_stats: list[str] = [c for c in ROLLING_STAT_COLS if c not in df.columns]
    if missing_stats:
        logger.debug("Stat columns not in data (skipped): %s", missing_stats)

    for window in windows:
        for stat in available_stats:
            # Simpler approach: use the shifted series directly with groupby
            mean_col: str = f"{stat}_L{window}_mean"
            std_col: str = f"{stat}_L{window}_std"

            # Compute rolling within groups using transform
            grouped = df.groupby(group_cols)

            df[mean_col] = grouped[stat].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
            )
            df[std_col] = grouped[stat].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).std()
            )

    n_new_cols: int = len(available_stats) * len(windows) * 2
    logger.info(
        "Computed %d rolling features (%d stats x %d windows x 2 aggs)",
        n_new_cols,
        len(available_stats),
        len(windows),
    )
    return df


def build_player_rolling_features(
    *,
    windows: list[int] | None = None,
    cross_season: bool = False,
    repo: Path | None = None,
) -> DataFrame:
    """Load cleaned player game logs and compute rolling features.

    Args:
        windows: Rolling window sizes. Defaults to ``[3, 6]``.
        cross_season: Whether rolling windows span season boundaries.
        repo: Repository root.

    Returns:
        DataFrame with original columns plus rolling feature columns.

    Raises:
        FileNotFoundError: If cleaned player game logs don't exist.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    resolved_windows: list[int] = windows or DEFAULT_WINDOWS

    path: Path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"
    if not path.exists():
        msg: str = (
            f"Cleaned player game logs not found at {path}. "
            "Run: gridiron run-data-pipeline --only clean-player-stats"
        )
        raise FileNotFoundError(msg)

    df: DataFrame = pd.read_parquet(path)
    logger.info("Loaded %d player-game rows for rolling features", len(df))

    # Filter to skill positions — rolling features are only meaningful
    # for players who accumulate stats regularly
    skill_df: DataFrame = df.loc[df["is_skill"], :].copy()
    logger.info("Filtered to %d skill-position rows", len(skill_df))

    result: DataFrame = _compute_rolling(
        skill_df,
        windows=resolved_windows,
        cross_season=cross_season,
    )

    return result
