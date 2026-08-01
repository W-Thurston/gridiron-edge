# src/gridiron_edge/models/game_prediction/total.py
"""Total-points regression model.

Total models use the canonical one-row-per-game expanded feature set and
train directly against ``ACTUAL_TOTAL`` from the persisted modeling
artifact.

Total prediction is an independent model workflow registered under
``total_random_forest`` and ``total_xgboost``.

Public API:
    _prepare_total_data  Canonical Total training-data preparation.
    TotalTrainer         Total regression model specification.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

from pandas import DataFrame, Index, Series

from gridiron_edge.models.game_prediction._features import (
    FEATURE_SETS,
    HOLDOUT_SEASONS,
    _make_expanded_features,
)
from gridiron_edge.models.game_prediction.base import (
    GameModelSpec,
    GameModelType,
    GamesTrainer,
)
from gridiron_edge.models.game_prediction.game_schema import (
    ACTUAL_TOTAL_TARGET,
)

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def _prepare_total_data(
    repo: Path,
) -> tuple[
    DataFrame,
    Series,
    DataFrame,
    Series,
    list[str],
    list[str],
]:
    """Prepare canonical Total train and holdout data.

    Loads the canonical modeling artifact, uses its persisted
    ``ACTUAL_TOTAL`` target directly, removes rows with unavailable model
    features, and performs the configured chronological holdout split.

    Args:
        repo: Repository root containing the canonical modeling artifact.

    Returns:
        Train features, train target, holdout features, holdout target,
        sorted training seasons, and sorted holdout seasons.

    Raises:
        ValueError: If the canonical Total target is unavailable.
    """
    from gridiron_edge.datasets.loaders import (
        load_modeling_file,
    )

    df: DataFrame = load_modeling_file(repo)

    if ACTUAL_TOTAL_TARGET not in df.columns:
        raise ValueError(
            "Canonical Total modeling data is missing required target "
            f"column: {ACTUAL_TOTAL_TARGET}"
        )

    df = df.dropna(subset=[ACTUAL_TOTAL_TARGET]).copy()

    df = df.sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
        ignore_index=True,
    )

    features: DataFrame = _make_expanded_features(df)
    valid = features.notna().all(axis=1)
    valid_index = features.index[valid]

    df = df.reindex(valid_index).copy()
    features = features.reindex(valid_index).copy()

    y: Series = df[ACTUAL_TOTAL_TARGET].astype(float)

    train_mask = ~df["YEAR"].isin(HOLDOUT_SEASONS)
    hold_mask = df["YEAR"].isin(HOLDOUT_SEASONS)

    logger.info(
        "Total model data: train=%d  holdout=%d  mean_total=%.1f",
        train_mask.sum(),
        hold_mask.sum(),
        y.mean(),
    )

    train_seasons: list[str] = sorted(
        df.loc[
            train_mask,
            "YEAR",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    holdout_seasons: list[str] = sorted(
        df.loc[
            hold_mask,
            "YEAR",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    train_index: Index = df.index[train_mask]
    holdout_index: Index = df.index[hold_mask]

    x_train: DataFrame = features.reindex(train_index)
    y_train: Series = y.reindex(train_index)
    x_holdout: DataFrame = features.reindex(holdout_index)
    y_holdout: Series = y.reindex(holdout_index)

    return (
        x_train,
        y_train,
        x_holdout,
        y_holdout,
        train_seasons,
        holdout_seasons,
    )


# ---------------------------------------------------------------------------
# TotalTrainer - spec-only subclass of GamesTrainer
# ---------------------------------------------------------------------------


class TotalTrainer(GamesTrainer):
    """Train total-points regressors (random_forest / xgboost).

    Logistic is excluded - it is not a regression estimator. Attempting
    ``TotalTrainer().train(df, model_type=GameModelType.LOGISTIC)`` raises
    ``ValueError`` via the spec validation in :meth:`GamesTrainer.train`.
    """

    @property
    def spec(self) -> GameModelSpec:
        """Return the Total regression model specification."""
        return GameModelSpec(
            name="total",
            task="regression",
            target_col=ACTUAL_TOTAL_TARGET,
            feature_set={
                GameModelType.RANDOM_FOREST: FEATURE_SETS["expanded"],
                GameModelType.XGBOOST: FEATURE_SETS["expanded"],
            },
            description="Game total points - regression.",
        )
