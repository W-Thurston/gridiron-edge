# src/gridiron_edge/models/prop_prediction/wr_rec_yards.py

"""WR receiving yards prop model.

Predicts a WR's receiving yards for a given game using player rolling
stats and opponent matchup features. ElasticNet regression baseline.

Usage::

    from gridiron_edge.models.prop_prediction.wr_rec_yards import WRRecYardsTrainer

    trainer = WRRecYardsTrainer()
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
from sklearn.linear_model import ElasticNet

# pyrefly: ignore [missing-import]
from sklearn.preprocessing import StandardScaler

from gridiron_edge.models.prop_prediction.base import (
    PropModelSpec,
    PropTrainer,
)

logger: Logger = logging.getLogger(__name__)


class WRRecYardsTrainer(PropTrainer):
    """WR receiving yards prop model using ElasticNet regression."""

    _scaler: StandardScaler | None = None
    _model: ElasticNet | None = None

    @property
    def spec(self) -> PropModelSpec:
        """WR receiving yards model specification."""
        return PropModelSpec(
            name="wr_rec_yards",
            target_col="receiving_yards",
            position_filter=["WR"],
            description="WR receiving yards — ElasticNet regression on rolling + matchup features",
        )

    def _fit(
        self,
        x_train: DataFrame,
        y_train: pd.Series,
        x_val: DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Fit ElasticNet with feature scaling and hyperparameter search."""
        self._scaler = StandardScaler()
        x_train_scaled = self._scaler.fit_transform(x_train)
        x_val_scaled = self._scaler.transform(x_val)

        best_alpha = 1.0
        best_l1_ratio = 0.5
        best_mae = float("inf")

        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                model.fit(x_train_scaled, y_train)
                preds = model.predict(x_val_scaled)
                mae = float(np.mean(np.abs(y_val.values - preds)))
                if mae < best_mae:
                    best_mae: float = mae
                    best_alpha: float = alpha
                    best_l1_ratio: float = l1_ratio

        self._model = ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1_ratio,
            max_iter=10000,
        )
        self._model.fit(x_train_scaled, y_train)

        # Log feature importances — ElasticNet zeros out irrelevant features
        coefs: dict[str, float] = dict(
            zip(self._feature_columns(), self._model.coef_, strict=False)
        )
        nonzero: dict[str, float] = {f: c for f, c in coefs.items() if abs(c) > 1e-6}
        top_features: list[tuple[str, float]] = sorted(
            nonzero.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]
        logger.info(
            "Best alpha=%.2f, l1_ratio=%.1f, val MAE=%.1f. Features: %d/%d nonzero. Top: %s",
            best_alpha,
            best_l1_ratio,
            best_mae,
            len(nonzero),
            len(coefs),
            [(f, f"{c:.2f}") for f, c in top_features],
        )

        return {
            "alpha": best_alpha,
            "l1_ratio": best_l1_ratio,
            "val_mae": best_mae,
            "n_features": len(self._feature_columns()),
            "n_nonzero": len(nonzero),
        }

    def _predict(self, x: DataFrame) -> np.ndarray:
        """Generate predictions from fitted model."""
        if self._model is None or self._scaler is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        x_scaled = self._scaler.transform(x)
        preds = self._model.predict(x_scaled)
        return np.clip(preds, 0, 400)
