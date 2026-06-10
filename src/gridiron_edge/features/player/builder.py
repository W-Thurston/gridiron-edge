# src/gridiron_edge/features/player/builder.py
"""Unified prop feature builder — single entry point for training-ready data.

Orchestrates all player feature modules (rolling, matchup, usage, game
context) into a single DataFrame suitable for prop model training.

Usage::

    from gridiron_edge.features.player.builder import build_prop_features

    df = build_prop_features(position_filter=["QB"])
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.features.player.game_context import build_game_context_features
from gridiron_edge.features.player.matchup import build_matchup_features
from gridiron_edge.features.player.rolling import build_player_rolling_features
from gridiron_edge.features.player.usage import build_usage_features

logger: Logger = logging.getLogger(__name__)

# Columns that must exist before feature building starts.
_REQUIRED_INPUT_COLS: Final[list[str]] = [
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent_team",
    "season",
    "week",
    "game_id",
    "is_skill",
]


def build_prop_features(
    *,
    position_filter: list[str],
    repo: Path | None = None,
) -> DataFrame:
    """Build the complete prop model feature DataFrame.

    Loads player game logs once, chains all feature builders, filters
    by position, drops rows with NaN in feature columns, and returns
    a training-ready DataFrame.

    Args:
        position_filter: Positions to include (e.g., ``["QB"]``,
            ``["RB"]``, ``["WR", "TE"]``).
        repo: Repository root override.

    Returns:
        DataFrame with one row per player-game, containing identity
        columns, target stat columns, and all prop feature columns.

    Raises:
        FileNotFoundError: If cleaned player game logs or games data
            not found.
        ValueError: If required input columns are missing.
    """
    resolved_repo = repo or get_settings().repo_root
    logs_path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"

    if not logs_path.exists():
        msg = f"Cleaned player game logs not found: {logs_path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(logs_path)
    n_raw = len(df)
    logger.info("Loaded %d player game logs", n_raw)

    # Validate required columns
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        msg = f"Missing required columns in player game logs: {missing}"
        raise ValueError(msg)

    # ── Chain feature builders (single load, df passed through) ──────────

    logger.info("Building rolling features...")
    df = build_player_rolling_features(df=df, repo=resolved_repo)

    logger.info("Building matchup features...")
    df = build_matchup_features(df=df, repo=resolved_repo)

    logger.info("Building usage features...")
    df = build_usage_features(df=df, repo=resolved_repo)

    logger.info("Building game context features...")
    df = build_game_context_features(df=df, repo=resolved_repo)

    # ── Filter by position ───────────────────────────────────────────────

    n_before_pos = len(df)
    df = df.loc[df["position"].isin(position_filter), :].copy()
    logger.info(
        "Position filter %s: %d → %d rows",
        position_filter,
        n_before_pos,
        len(df),
    )

    # ── Validate feature columns ─────────────────────────────────────────

    available_features = [c for c in PROP_FEATURE_COLS if c in df.columns]
    missing_features = [c for c in PROP_FEATURE_COLS if c not in df.columns]
    if missing_features:
        logger.warning(
            "%d expected feature columns not found: %s",
            len(missing_features),
            missing_features[:10],
        )

    # NaN handling is deferred to the trainer, which has position context
    # to determine which features have reasonable coverage. Dropping here
    # would be too aggressive (e.g., QBs have ~99% NaN in receiving
    # features, WRs in passing features).

    logger.info(
        "Final prop feature DataFrame: %d rows, %d feature columns, "
        "%d total columns, %d unique players",
        len(df),
        len(available_features),
        len(df.columns),
        df["player_id"].nunique(),
    )

    return df
