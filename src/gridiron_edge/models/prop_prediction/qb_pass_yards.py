# src/gridiron_edge/models/prop_prediction/qb_pass_yards.py

"""QB passing yards prop model.

First concrete prop model. Predicts a QB's passing yards for a given
game using player rolling stats and opponent matchup features.

Uses Ridge regression as the baseline — simple, regularized, interpretable,
and fast to train. Can be swapped for XGBoost/RF once the pipeline is
validated end-to-end.

Usage::

    from gridiron_edge.models.prop_prediction.qb_pass_yards import QBPassYardsTrainer

    trainer = QBPassYardsTrainer()
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

# Feature columns for QB passing yards prediction.
# Curated from the 164 available columns to focus on high-signal,
# low-NaN features directly relevant to passing production.
_FEATURE_COLUMNS: list[str] = [
    # --- Player rolling stats (L3 = recent form) ---
    "passing_yards_L3_mean",
    "passing_yards_L3_std",
    "passing_yards_L6_mean",
    "attempts_L3_mean",
    "attempts_L6_mean",
    "completions_L3_mean",
    "passing_tds_L3_mean",
    "passing_interceptions_L3_mean",
    "passing_air_yards_L3_mean",
    "passing_air_yards_L6_mean",
    "passing_epa_L3_mean",
    "passing_epa_L6_mean",
    "sacks_suffered_L3_mean",
    # --- Opponent matchup features ---
    "opp_pass_yards_allowed_L6",
    "opp_pass_yards_allowed_rank_L6",
    "opp_pass_epa_allowed_L6",
    "opp_pass_epa_allowed_rank_L6",
    "opp_sacks_allowed_L6",
    "opp_sacks_allowed_rank_L6",
    "opp_pass_tds_allowed_L6",
]


class QBPassYardsTrainer(PropTrainer):
    """QB passing yards prop model using Ridge regression."""

    _scaler: StandardScaler | None = None
    _model: Ridge | None = None

    @property
    def spec(self) -> PropModelSpec:
        """QB passing yards model specification."""
        return PropModelSpec(
            name="qb_pass_yards",
            target_col="passing_yards",
            position_filter=["QB"],
            description="QB passing yards — Ridge regression on rolling + matchup features",
        )

    def _feature_columns(self) -> list[str]:
        return _FEATURE_COLUMNS

    def _build_features(self, df: DataFrame) -> DataFrame:
        """Select feature columns and target, drop NaN rows."""
        target = self.spec.target_col
        cols = [
            *self._feature_columns(),
            target,
            "player_id",
            "season",
            "week",
            "player_name",
            "game_id",
        ]
        available = [c for c in cols if c in df.columns]
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

        # Try a few alpha values, pick best on validation
        best_alpha = 1.0
        best_mae = float("inf")

        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha)
            model.fit(x_train_scaled, y_train)
            preds = model.predict(x_val_scaled)
            mae = float(np.mean(np.abs(y_val.values - preds)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        self._model = Ridge(alpha=best_alpha)
        self._model.fit(x_train_scaled, y_train)

        # Log feature importances (coefficients)
        coefs = dict(zip(self._feature_columns(), self._model.coef_, strict=False))
        top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
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

        # Clip to reasonable range (no negative passing yards in practice)
        return np.clip(preds, 0, 600)
