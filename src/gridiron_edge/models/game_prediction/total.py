# src/gridiron_edge/models/game_prediction/total.py
"""Total points regression model.

Total-points predictions use the same expanded feature set as the
win-probability models, but target ``actual_total = PTS_WINNER + PTS_LOSER``
instead of ``RESULT``. The model is trained via :class:`TotalTrainer`
and served at predict time through :class:`GamesPredictor` (registered
under composite key ``"total_random_forest"`` and ``"total_xgboost"``).

This is a supporting model — total predictions feed into win_prob
:meth:`GamesPredictor._maybe_predict_totals` to attach ``model_total``
to game-level predictions.

Public API:
    _prepare_total_data  Data preparation helper (used by GamesTrainer).
    TotalTrainer         Spec-only subclass of GamesTrainer.
    DEFAULT_TOTAL_MODEL_NAME / DEFAULT_TOTAL_MODEL_TYPE: identity
        constants used by GamesPredictor when defaulting the total model.

Workstream 2 D2b.3:
    The legacy ``train_total_model`` / ``load_total_model`` /
    ``predict_total`` free functions were deleted. All total-model
    training, loading, and prediction now flow through ``GamesTrainer``
    / ``GamesPredictor``.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

import pandas as pd
from pandas import DataFrame

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

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default ``model_name`` for the total points regression family.
#: Used by :meth:`GamesPredictor._maybe_predict_totals` to identify which
#: total model to attach to win_prob predictions.
DEFAULT_TOTAL_MODEL_NAME: str = "total"

#: Default ``model_type`` for the total points regression family.
DEFAULT_TOTAL_MODEL_TYPE: str = "random_forest"


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def _prepare_total_data(
    repo: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare features and total-points target for training.

    Loads the modeling file, joins game scores to compute
    ``actual_total = PTS_WINNER + PTS_LOSER``, then splits into
    train / holdout using the same ``HOLDOUT_SEASONS`` as the win model.

    Called by :meth:`GamesTrainer._prepare_window` when ``spec.task ==
    "regression"`` (the regression branch always uses the standard
    4-game EPA window — total models don't tune ``epa_window``).

    Returns:
        Tuple of (x_train, y_train, x_hold, y_hold, train_seasons, holdout_seasons).
        Season lists are formatted as ``"YYYY-YYYY"`` strings to match the
        :class:`BaseModelMetadata` convention.
    """
    from gridiron_edge.datasets.loaders import load_games, load_modeling_file

    df: DataFrame = load_modeling_file(repo)
    games: DataFrame = load_games(repo)

    # Build total lookup: GAME_ID → actual_total
    games_lookup: DataFrame = games.dropna(subset=["PTS_WINNER", "PTS_LOSER"]).copy()
    games_lookup = games_lookup.drop_duplicates(subset=["GAME_ID"])
    games_lookup["actual_total"] = games_lookup["PTS_WINNER"] + games_lookup["PTS_LOSER"]

    df = df.merge(
        games_lookup[["GAME_ID", "actual_total"]],
        on="GAME_ID",
        how="inner",
    )

    df = df.dropna(subset=["actual_total"])

    features: DataFrame = _make_expanded_features(df)
    valid = features.notna().all(axis=1)
    df = df.loc[valid, :].copy()
    features = features.loc[valid, :].copy()

    y = df["actual_total"].astype(float)

    # Sort by time so TimeSeriesSplit respects temporal ordering.
    time_order = df[["YEAR", "WEEK_NUM"]].sort_values(["YEAR", "WEEK_NUM"]).index
    df = df.loc[time_order]
    features = features.loc[time_order]
    y = y.loc[time_order]

    train_mask = ~df["YEAR"].isin(HOLDOUT_SEASONS)
    hold_mask = df["YEAR"].isin(HOLDOUT_SEASONS)

    logger.info(
        "Total model data: train=%d  holdout=%d  mean_total=%.1f",
        train_mask.sum(),
        hold_mask.sum(),
        y.mean(),
    )

    train_seasons: list[str] = sorted(df.loc[train_mask, "YEAR"].unique().tolist())
    hold_seasons: list[str] = sorted(df.loc[hold_mask, "YEAR"].unique().tolist())

    # pyrefly: ignore [bad-return]
    return (
        features.loc[train_mask],
        y.loc[train_mask],
        features.loc[hold_mask],
        y.loc[hold_mask],
        train_seasons,
        hold_seasons,
    )


# ---------------------------------------------------------------------------
# TotalTrainer — spec-only subclass of GamesTrainer
# ---------------------------------------------------------------------------


class TotalTrainer(GamesTrainer):
    """Train total-points regressors (random_forest / xgboost).

    Logistic is excluded — it is not a regression estimator. Attempting
    ``TotalTrainer().train(df, model_type=GameModelType.LOGISTIC)`` raises
    ``ValueError`` via the spec validation in :meth:`GamesTrainer.train`.
    """

    @property
    def spec(self) -> GameModelSpec:
        """Return the total-points model specification."""
        return GameModelSpec(
            name="total",
            task="regression",
            target_col="actual_total",
            feature_set={
                GameModelType.RANDOM_FOREST: FEATURE_SETS["expanded"],
                GameModelType.XGBOOST: FEATURE_SETS["expanded"],
            },
            description="Game total points — regression.",
        )
