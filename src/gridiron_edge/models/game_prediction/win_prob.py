# src/gridiron_edge/models/game_prediction/win_prob.py

"""WinProbTrainer — game-winner classification spec subclass.

Spec-only subclass of :class:`GamesTrainer`. All training, evaluation,
and persistence logic lives in :class:`GamesTrainer`. This module declares
which feature sets and target column define the ``win_prob`` family.

Logistic uses the combined feature set; tree-based models use the expanded
feature set — matches the existing trainer behavior.

The trainer contributes only the spec; all training behavior — feature
selection, HP search, fit, eval — lives in the :class:`GamesTrainer`
base class.
"""

from __future__ import annotations

from gridiron_edge.models.game_prediction._features import FEATURE_SETS
from gridiron_edge.models.game_prediction.base import (
    GameModelSpec,
    GameModelType,
    GamesTrainer,
)


class WinProbTrainer(GamesTrainer):
    """Train win-probability classifiers (logistic / random_forest / xgboost)."""

    @property
    def spec(self) -> GameModelSpec:
        """Return the win-probability model specification."""
        return GameModelSpec(
            name="win_prob",
            task="classification",
            target_col="RESULT",
            feature_set={
                GameModelType.LOGISTIC: FEATURE_SETS["combined"],
                GameModelType.RANDOM_FOREST: FEATURE_SETS["expanded"],
                GameModelType.XGBOOST: FEATURE_SETS["expanded"],
            },
            description="Game winner probability — multi-algorithm classifier.",
        )
