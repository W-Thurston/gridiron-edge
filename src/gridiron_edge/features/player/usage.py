# src/gridiron_edge/features/player/usage.py
"""Per-player usage share features for prop models.

Computes team-level totals per game, derives each player's share of
targets, carries, and touches, then applies rolling windows.  All rolling
computations use shift(1) to prevent lookahead leakage — a player's
usage share for week N reflects only games through week N-1.

Features produced (per window W):
- usage_target_share_LW  — player targets / team total targets
- usage_carry_share_LW   — player carries / team total carries
- usage_touch_share_LW   — player touches / team total touches

Usage::

    from gridiron_edge.features.player.usage import build_usage_features

    df = build_usage_features()
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

# Default rolling windows — consistent with rolling.py
DEFAULT_WINDOWS: Final[list[int]] = [3, 6]

# Per-game share columns (intermediate — dropped before return)
_SHARE_COLS: Final[list[str]] = [
    "usage_target_share",
    "usage_carry_share",
    "usage_touch_share",
]


def _compute_team_totals(df: DataFrame) -> DataFrame:
    """Compute team-level totals for targets, carries, and touches per game.

    Returns a DataFrame with one row per (season, week, team, game_id)
    and columns: team_total_targets, team_total_carries, team_total_touches.
    """
    team_totals = df.groupby(["season", "week", "team", "game_id"], as_index=False).agg(
        team_total_targets=("targets", "sum"),
        team_total_carries=("carries", "sum"),
    )
    team_totals["team_total_touches"] = (
        team_totals["team_total_targets"] + team_totals["team_total_carries"]
    )
    return team_totals


def _compute_per_game_shares(df: DataFrame) -> DataFrame:
    """Compute per-player per-game usage shares.

    Joins team totals back to player rows and computes target, carry, and
    touch shares.  Division by zero produces 0.0 (not NaN).
    """
    team_totals = _compute_team_totals(df)

    merged = df.merge(
        team_totals,
        on=["season", "week", "team", "game_id"],
        how="left",
    )

    targets = merged["targets"].fillna(0)
    carries = merged["carries"].fillna(0)
    touches = targets + carries

    merged["usage_target_share"] = targets.div(merged["team_total_targets"]).fillna(0.0)
    merged["usage_carry_share"] = carries.div(merged["team_total_carries"]).fillna(0.0)
    merged["usage_touch_share"] = touches.div(merged["team_total_touches"]).fillna(0.0)

    return merged.drop(
        columns=["team_total_targets", "team_total_carries", "team_total_touches"],
    )


def _rolling_shares(
    df: DataFrame,
    *,
    windows: list[int],
    cross_season: bool = False,
) -> DataFrame:
    """Compute shifted rolling mean of usage shares per player.

    Uses shift(1) so that a player's rolling usage share for week N
    reflects only games through week N-1.
    """
    df = df.sort_values(["player_id", "season", "week"])

    group_cols = ["player_id"] if cross_season else ["player_id", "season"]

    for window in windows:
        for col in _SHARE_COLS:
            out_col = f"{col}_L{window}"
            df[out_col] = df.groupby(group_cols)[col].transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
            )

    return df


def build_usage_features(
    *,
    windows: list[int] | None = None,
    cross_season: bool = False,
    repo: Path | None = None,
) -> DataFrame:
    """Load cleaned player game logs and compute usage share features.

    Args:
        windows: Rolling window sizes.  Defaults to [3, 6].
        cross_season: If True, rolling windows cross season boundaries.
        repo: Repository root override.

    Returns:
        DataFrame with original columns plus usage share rolling features.
        Per-game share intermediates are dropped; only ``_L{window}``
        columns are included (safe as model features).

    Raises:
        FileNotFoundError: If cleaned player game logs not found.
    """
    resolved_repo = repo or get_settings().repo_root
    logs_path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"

    if not logs_path.exists():
        msg = f"Cleaned player game logs not found: {logs_path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(logs_path)
    logger.info("Loaded %d player game logs for usage features", len(df))

    if windows is None:
        windows = list(DEFAULT_WINDOWS)

    df = _compute_per_game_shares(df)
    df = _rolling_shares(df, windows=windows, cross_season=cross_season)

    # Drop intermediate per-game share columns — only rolling features are
    # safe as model inputs (per-game shares contain current-game info).
    df = df.drop(columns=_SHARE_COLS)

    usage_cols = [c for c in df.columns if c.startswith("usage_") and "_L" in c]
    logger.info(
        "Built %d usage features for %d player-games",
        len(usage_cols),
        len(df),
    )
    for col in usage_cols:
        nan_rate = df[col].isna().mean()
        logger.debug("  %s: %.1f%% NaN", col, nan_rate * 100)

    return df
