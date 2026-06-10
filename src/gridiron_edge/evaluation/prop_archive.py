# src/gridiron_edge/evaluation/prop_archive.py
"""Append-only archive for prop model predictions.

Persists enriched prop predictions to a parquet log for historical
tracking, CLV analysis, and model comparison over time.

Deduplicates on (game_id, player_id, stat_type, model_version) so
re-running predictions for the same player-game replaces the old row.

Usage::

    from gridiron_edge.evaluation.prop_archive import (
        archive_prop_predictions,
        load_prop_archive,
    )

    archive_prop_predictions(predictions_df, repo=repo)
    history = load_prop_archive(repo=repo)
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Deduplication key — last write wins for same player-game-stat-model
_DEDUP_KEYS: Final[list[str]] = [
    "game_id",
    "player_id",
    "stat_type",
    "model_version",
]

# Archive schema — columns in canonical order
_ARCHIVE_COLUMNS: Final[list[str]] = [
    "predicted_at",
    "is_backfilled",
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "position",
    "team",
    "stat_type",
    "model_version",
    "predicted_mean",
    "predicted_std",
    "lo_90",
    "hi_90",
    "line",
    "p_over",
    "lean",
    "confidence_tier",
]

_DEFAULT_SUBDIR: Final[str] = "data/output/props"
_DEFAULT_FILENAME: Final[str] = "prop_predictions_log.parquet"


def _archive_path(repo: Path | None = None) -> Path:
    """Resolve the archive file path."""
    resolved: Path = repo or get_settings().repo_root
    return resolved / _DEFAULT_SUBDIR / _DEFAULT_FILENAME


def archive_prop_predictions(
    df: DataFrame,
    *,
    repo: Path | None = None,
    is_backfilled: bool = False,
    model_version: str = "v1",
) -> Path:
    """Append prop predictions to the archive.

    Adds metadata columns (predicted_at, is_backfilled, model_version),
    deduplicates against existing archive on the dedup key, and writes
    the merged result.

    Args:
        df: Predictions DataFrame.  Must contain at minimum: game_id,
            player_id, stat_type, predicted_mean.
        repo: Repository root override.
        is_backfilled: Whether these predictions are historical backfill
            (True) or live predictions (False).
        model_version: Model version tag for dedup and tracking.

    Returns:
        Path to the written archive file.

    Raises:
        ValueError: If required columns are missing from df.
    """
    required: list[str] = ["game_id", "player_id", "stat_type", "predicted_mean"]
    missing: list[str] = [c for c in required if c not in df.columns]
    if missing:
        msg: str = f"Missing required columns for archive: {missing}"
        raise ValueError(msg)

    result: DataFrame = df.copy()

    # Add metadata
    result["predicted_at"] = datetime.now(UTC).isoformat()
    result["is_backfilled"] = is_backfilled
    if "model_version" not in result.columns:
        result["model_version"] = model_version

    # Ensure all archive columns exist (fill missing with NaN)
    for col in _ARCHIVE_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA

    # Select and order columns
    result = result.loc[:, [c for c in _ARCHIVE_COLUMNS if c in result.columns]]

    # Load existing archive and merge
    path: Path = _archive_path(repo)
    if path.exists():
        existing: DataFrame = pd.read_parquet(path)
        logger.info(
            "Loaded existing archive: %d rows",
            len(existing),
        )
        # Concat new on top, then dedup keeping last (new wins)
        merged: DataFrame = pd.concat(
            [existing, result],
            ignore_index=True,
        ).drop_duplicates(
            subset=_DEDUP_KEYS,
            keep="last",
        )
    else:
        merged = result

    # Write
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    logger.info(
        "Archived %d predictions (%d new) → %s",
        len(merged),
        len(result),
        path,
    )
    return path


def load_prop_archive(
    *,
    repo: Path | None = None,
    stat_type: str | None = None,
    season: int | None = None,
) -> DataFrame:
    """Load the prop predictions archive.

    Args:
        repo: Repository root override.
        stat_type: Optional filter by stat type (e.g., "qb_pass_yards").
        season: Optional filter by season.

    Returns:
        DataFrame of archived predictions, or empty DataFrame with
        archive schema if no archive exists.
    """
    path: Path = _archive_path(repo)
    if not path.exists():
        logger.info("No prop archive found at %s", path)
        return DataFrame(columns=_ARCHIVE_COLUMNS)

    df: DataFrame = pd.read_parquet(path)
    logger.info("Loaded prop archive: %d rows", len(df))

    if stat_type is not None:
        df = df.loc[df["stat_type"] == stat_type, :]
    if season is not None:
        df = df.loc[df["season"] == season, :]

    return df
