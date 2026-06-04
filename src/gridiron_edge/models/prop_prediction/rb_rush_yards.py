# src/gridiron_edge/models/prop_prediction/rb_rush_yards.py

"""RB rushing yards prop model.

Predicts a RB's rushing yards for a given game using player rolling
stats and opponent matchup features. Ridge regression baseline.

Usage::

    from gridiron_edge.models.prop_prediction.rb_rush_yards import RBRushYardsTrainer

    trainer = RBRushYardsTrainer()
    metadata = trainer.train()
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
from sklearn.linear_model import Ridge

# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler

from gridiron_edge.models.prop_prediction.base import (
    PropModelSpec,
    PropTrainer,
)

logger: Logger = logging.getLogger(__name__)

_FEATURE_COLUMNS: list[str] = [
    # --- Player rolling stats ---
    "rushing_yards_L3_mean",
    "rushing_yards_L3_std",
    "rushing_yards_L6_mean",
    "carries_L3_mean",
    "carries_L6_mean",
    "rushing_tds_L3_mean",
    "rushing_epa_L3_mean",
    "rushing_epa_L6_mean",
    "rushing_fumbles_L3_mean",
    # --- Receiving (RBs catch passes too) ---
    "receiving_yards_L3_mean",
    "targets_L3_mean",
    # --- Opponent matchup features ---
    "opp_rush_yards_allowed_L6",
    "opp_rush_yards_allowed_rank_L6",
    "opp_rush_epa_allowed_L6",
    "opp_rush_epa_allowed_rank_L6",
    "opp_rush_tds_allowed_L6",
]


class RBRushYardsTrainer(PropTrainer):
    """RB rushing yards prop model using Ridge regression."""

    _scaler: StandardScaler | None = None
    _model: Ridge | None = None

    @property
    def spec(self) -> PropModelSpec:
        """RB rushing yards model specification."""
        return PropModelSpec(
            name="rb_rush_yards",
            target_col="rushing_yards",
            position_filter=["RB", "FB"],
            description="RB rushing yards — Ridge regression on rolling + matchup features",
        )

    def _feature_columns(self) -> list[str]:
        return _FEATURE_COLUMNS

    def _build_features(self, df: DataFrame) -> DataFrame:
        """Select feature columns and target, drop NaN rows."""
        target: str = self.spec.target_col
        cols: list[str] = [
            *self._feature_columns(),
            target,
            "player_id",
            "season",
            "week",
            "player_name",
            "game_id",
        ]
        available: list[str] = [c for c in cols if c in df.columns]
        return df.loc[:, available].copy()

    def _fit(
        self,
        x_train: DataFrame,
        y_train: pd.Series,
        x_val: DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Fit Ridge regression with feature scaling."""
        self._scaler = StandardScaler()
        x_train_scaled = self._scaler.fit_transform(x_train)
        x_val_scaled = self._scaler.transform(x_val)

        best_alpha = 1.0
        best_mae = float("inf")

        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha)
            model.fit(x_train_scaled, y_train)
            preds = model.predict(x_val_scaled)
            mae = float(np.mean(np.abs(y_val.values - preds)))
            if mae < best_mae:
                best_mae: float = mae
                best_alpha: float = alpha

        self._model = Ridge(alpha=best_alpha)
        self._model.fit(x_train_scaled, y_train)

        coefs: dict = dict(zip(self._feature_columns(), self._model.coef_, strict=False))
        top_features: list = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        logger.info(
            "Best alpha=%.2f, val MAE=%.1f. Top features: %s",
            best_alpha,
            best_mae,
            [(f, f"{c:.2f}") for f, c in top_features],
        )

        return {
            "alpha": best_alpha,
            "val_mae": best_mae,
            "n_features": len(self._feature_columns()),
        }

    def _predict(self, x: DataFrame) -> np.ndarray:
        """Generate predictions from fitted model."""
        if self._model is None or self._scaler is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        x_scaled = self._scaler.transform(x)
        preds = self._model.predict(x_scaled)
        return np.clip(preds, 0, 400)
