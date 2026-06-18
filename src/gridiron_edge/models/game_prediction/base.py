# src/gridiron_edge/models/game_prediction/base.py

"""Base infrastructure for game prediction models.

Game models predict game-level outcomes: win probability (classification)
or total points (regression). They share the team/EPA feature pipeline but
have their own training, evaluation, and prediction interfaces.

Architecture (final shape, Workstream 2):
    - ``GameModelType`` — supported algorithms (D2a)
    - ``GameModelSpec`` — metadata describing a game model (D2a)
    - ``GameModelMetadata`` — metadata recorded alongside an artifact (D1a)
    - ``GamesTrainer`` — base class for training game models (D2a)
    - Evaluation uses Brier/ECE/AUC for classification and MAE/RMSE/R² for
      regression, dispatched on ``spec.task``.

This module currently contains only :class:`GameModelMetadata`. The trainer
infrastructure (``GameModelType``, ``GameModelSpec``, ``GamesTrainer``,
module-level ``_create_model`` / ``_get_param_grid``) is added in D2a.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridiron_edge.models.artifact import BaseModelMetadata


@dataclass(kw_only=True)
class GameModelMetadata(BaseModelMetadata):
    """Metadata recorded alongside a trained game model artifact.

    Inherits the shared metadata fields from :class:`BaseModelMetadata`.

    Classification metrics are populated when ``task="classification"``
    (used by ``win_prob``); regression metrics are populated when
    ``task="regression"`` (used by ``total``). Unused metrics remain NaN.

    Attributes:
        holdout_brier: Brier score (classification). Primary win_prob metric.
        holdout_ece: Expected calibration error (classification).
        holdout_auc: Area under ROC curve (classification).
        holdout_log_loss: Log loss (classification).
        holdout_accuracy: Accuracy (classification).
        holdout_mae: MAE (regression). Primary total metric.
        holdout_rmse: RMSE (regression).
        holdout_r2: R² (regression).
    """

    # Classification metrics (NaN for regression)
    holdout_brier: float = float("nan")
    holdout_ece: float = float("nan")
    holdout_auc: float = float("nan")
    holdout_log_loss: float = float("nan")
    holdout_accuracy: float = float("nan")
    # Regression metrics (NaN for classification)
    holdout_mae: float = float("nan")
    holdout_rmse: float = float("nan")
    holdout_r2: float = float("nan")
