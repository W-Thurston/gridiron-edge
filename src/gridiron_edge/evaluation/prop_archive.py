# src/gridiron_edge/evaluation/prop_archive.py
"""Append-only archive for prop model predictions.

Persists enriched prop predictions to a parquet log for historical
tracking, CLV analysis, and model comparison over time.

Deduplicates on (game_id, player_id, stat_type, model_name, model_type)
so re-running predictions for the same player-game and algorithm replaces
the old row without collapsing different algorithms together.

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

# Deduplication key - last write wins for same player-game-stat-model
_DEDUP_KEYS: Final[list[str]] = [
    "game_id",
    "player_id",
    "stat_type",
    "model_name",
    "model_type",
]

# Archive schema - columns in canonical order
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
    "model_name",
    "model_type",
    "predicted_mean",
    "predicted_std",
    "lo_90",
    "hi_90",
    "line",
    "p_over",
    "lean",
    "confidence_tier",
]

_PROP_PREDICTION_COLUMNS: Final[list[str]] = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "position",
    "team",
    "stat_type",
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


def _require_archive_schema(
    df: DataFrame,
    *,
    label: str,
) -> None:
    """Require the exact current persisted prop archive schema."""
    actual: list[str] = df.columns.tolist()
    expected: list[str] = _ARCHIVE_COLUMNS

    missing: list[str] = [column for column in expected if column not in actual]
    extra: list[str] = [column for column in actual if column not in expected]

    problems: list[str] = []

    if missing:
        problems.append("missing columns: " + ", ".join(missing))

    if extra:
        problems.append("extra columns: " + ", ".join(extra))

    if not missing and not extra and actual != expected:
        problems.append("columns are not in canonical order")

    if problems:
        raise ValueError(
            f"{label} does not match the current prop prediction "
            "archive schema: " + "; ".join(problems)
        )


def _archive_path(repo: Path | None = None) -> Path:
    """Resolve the archive file path."""
    resolved: Path = repo or get_settings().repo_root
    return resolved / _DEFAULT_SUBDIR / _DEFAULT_FILENAME


def archive_prop_predictions(
    df: DataFrame,
    *,
    repo: Path | None = None,
    is_backfilled: bool = False,
    model_name: str,
    model_type: str,
) -> Path:
    """Append prop predictions to the archive.

    Adds metadata columns (predicted_at, is_backfilled, model_name, model_type),
    deduplicates against existing archive on the dedup key, and writes
    the merged result.

    Args:
        df: Predictions DataFrame containing the complete current prop
            prediction payload. Columns with unavailable values remain
            present with null values.
        repo: Repository root override.
        is_backfilled: Whether these predictions are historical backfill
            (True) or live predictions (False).
        model_name: Prop model family name (e.g. "qb_pass_yards").
        model_type: Algorithm identifier
            (e.g. "elasticnet", "random_forest", "xgboost").


    Returns:
        Path to the written archive file.

    Raises:
        ValueError: If any current prop prediction payload columns are
            missing from df.
    """
    missing: list[str] = [column for column in _PROP_PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Prop prediction rows are missing required archive columns: " + ", ".join(missing)
        )

    result: DataFrame = df.copy()

    # Add metadata
    result["predicted_at"] = datetime.now(UTC).isoformat()
    result["is_backfilled"] = is_backfilled
    result["model_name"] = model_name
    result["model_type"] = model_type

    # Serialize only the complete current archive schema.
    result = result.loc[:, _ARCHIVE_COLUMNS]

    # Load existing archive and merge
    path: Path = _archive_path(repo)
    if path.exists():
        existing: DataFrame = pd.read_parquet(path)
        _require_archive_schema(
            existing,
            label="Existing prop prediction archive",
        )
        logger.info(
            "Loaded existing archive: %d rows",
            len(existing),
        )

        # Concat new on top, then dedup keeping last (new wins).
        merged: DataFrame = pd.concat(
            [existing, result],
            ignore_index=True,
        ).drop_duplicates(
            subset=_DEDUP_KEYS,
            keep="last",
        )
    else:
        merged = result

    _require_archive_schema(
        merged,
        label="Prop prediction archive write",
    )

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
    _require_archive_schema(
        df,
        label="Prop prediction archive",
    )
    logger.info("Loaded prop archive: %d rows", len(df))

    if stat_type is not None:
        df = df.loc[df["stat_type"] == stat_type, :]
    if season is not None:
        df = df.loc[df["season"] == season, :]

    return df


# ---------------------------------------------------------------------------
# Canonical prop evaluation join
# ---------------------------------------------------------------------------


def build_prop_evaluation_df(
    *,
    model_name: str,
    model_type: str,
    season: int | None = None,
    repo: Path | None = None,
    actuals_df: DataFrame | None = None,
) -> DataFrame:
    """Join archived prop predictions to player game actuals.

    Returns a DataFrame containing one row per player-game prediction
    matched against its actual stat outcome. This is the canonical
    surface for prop evaluation, champion comparison, and CLV-style
    archive analytics.

    Predictions are filtered to a specific ``(model_name, model_type)``
    pair. Actuals are
    obtained either from ``actuals_df`` (preferred for tests and reuse)
    or by building features via ``build_prop_features`` (default for
    normal CLI usage).

    Args:
        model_name: Prop family name (e.g. ``"qb_pass_yards"``).
        model_type: Algorithm identifier
            (e.g. ``"elasticnet"``, ``"random_forest"``).
        season: Optional season filter applied to the archive load.
        repo: Repository root override.
        actuals_df: Optional pre-built actuals DataFrame. Must include
            ``game_id``, ``player_id``, and the trainer's
            ``spec.target_col``. When ``None``, ``build_prop_features``
            is called using the trainer's ``position_filter``.

    Returns:
        DataFrame with one row per matched prediction. The actual stat
        column is renamed to ``actual`` so downstream evaluators stay
        decoupled from per-stat column naming.

    Raises:
        KeyError: If ``model_name`` is not a registered prop model.
        ValueError: If the resolved actuals DataFrame is missing
            required columns.
    """
    # Local import to avoid CLI/registry import cycles at module load.
    from typing import cast

    from gridiron_edge.models.prop_prediction.base import PropTrainer

    # Resolve the trainer and target column.
    # Ensure prop trainer modules are imported so they register with
    # ModelRegistry. Without this, ModelRegistry.get() fails with an
    # empty registry when build_prop_evaluation_df is called from a
    # context that hasn't already imported the prop modules.
    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    model_cls = ModelRegistry.get(model_name)

    trainer = cast(PropTrainer, model_cls())
    if not isinstance(trainer, PropTrainer):
        msg = (
            f"Registered model {model_name!r} is not a PropTrainer; "
            f"cannot build prop evaluation join."
        )
        raise ValueError(msg)

    target_col: str = trainer.spec.target_col

    # Load predictions for the requested composite model identity.
    predictions: DataFrame = load_prop_archive(
        repo=repo,
        stat_type=model_name,
        season=season,
    )
    if predictions.empty:
        return _empty_evaluation_df()

    predictions = predictions.loc[
        (predictions["model_name"] == model_name) & (predictions["model_type"] == model_type),
        :,
    ]
    if predictions.empty:
        return _empty_evaluation_df()

    # Resolve actuals.
    if actuals_df is None:
        from gridiron_edge.features.player.builder import build_prop_features

        actuals_df = build_prop_features(
            position_filter=trainer.spec.position_filter,
            repo=repo,
        )

    required: set[str] = {"game_id", "player_id", target_col}
    missing: list[str] = sorted(required - set(actuals_df.columns))
    if missing:
        msg = f"actuals_df is missing required columns for {model_name!r}: {missing}"
        raise ValueError(msg)

    actuals: DataFrame = actuals_df.dropna(subset=[target_col]).loc[
        :,
        ["game_id", "player_id", target_col],
    ]

    # Inner join where predictions and actuals both exist.
    merged: DataFrame = predictions.merge(
        actuals,
        on=["game_id", "player_id"],
        how="inner",
    )
    if merged.empty:
        return _empty_evaluation_df()

    # Normalize the actual column name.
    merged = merged.rename(columns={target_col: "actual"})

    # Sort stably for deterministic downstream behavior.
    sort_cols: list[str] = [
        c for c in ("season", "week", "game_id", "player_id") if c in merged.columns
    ]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    return merged


def _empty_evaluation_df() -> DataFrame:
    """Schema-consistent empty result for the evaluation join."""
    columns: list[str] = [
        *_ARCHIVE_COLUMNS,
        "actual",
    ]
    return DataFrame(columns=columns)
