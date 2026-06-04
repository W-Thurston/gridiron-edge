# src/gridiron_edge/models/prop_prediction/base.py

"""Base infrastructure for prop prediction models.

Prop models predict continuous player stats (passing yards, rushing yards,
etc.) rather than binary game outcomes. They share the player feature
pipeline (rolling stats + matchup features) but have their own training,
evaluation, and prediction interfaces.

Architecture:
    - ``PropModelSpec`` — metadata describing a prop model
    - ``PropModelResult`` — standardized prediction output
    - ``PropTrainer`` — base class for training prop models
    - Evaluation uses MAE/RMSE/R² instead of Brier/AUC/ECE

Adding a new prop model:
    1. Subclass ``PropTrainer``
    2. Implement ``_feature_columns()``, ``_build_features()``, ``_fit()``
    3. Register in the prop model registry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
from sklearn.model_selection import TimeSeriesSplit

from gridiron_edge.core.settings import get_settings
from gridiron_edge.features.player.matchup import build_matchup_features
from gridiron_edge.features.player.rolling import build_player_rolling_features

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropModelSpec:
    """Metadata describing a prop model's identity.

    Attributes:
        name: Unique key (e.g. ``"qb_pass_yards"``).
        target_col: Column in player game logs that is the prediction target.
        position_filter: Position(s) this model applies to.
        description: Human-readable description.
    """

    name: str
    target_col: str
    position_filter: list[str]
    description: str = ""


@dataclass
class PropModelMetadata:
    """Metadata recorded alongside a trained prop model artifact.

    Attributes:
        model_name: Registered model name.
        trained_at: ISO-format UTC timestamp.
        target_col: Target column name.
        holdout_mae: MAE on holdout set (primary metric).
        holdout_rmse: RMSE on holdout set.
        holdout_r2: R² on holdout set.
        training_seasons: Seasons used for training.
        holdout_seasons: Seasons used for evaluation.
        parameters: Hyperparameters.
        feature_columns: Ordered feature columns the model expects.
        n_train_rows: Number of training rows.
        n_holdout_rows: Number of holdout rows.
        notes: Free-text notes.
    """

    model_name: str
    trained_at: str
    target_col: str
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float
    training_seasons: list[int] = field(default_factory=list)
    holdout_seasons: list[int] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    n_holdout_rows: int = 0
    notes: str = ""


@dataclass
class PropPrediction:
    """A single prop prediction for a player-game.

    Attributes:
        player_id: nflverse player ID.
        player_name: Display name.
        game_id: nflverse game ID.
        season: Season year.
        week: Week number.
        predicted: Model's point estimate.
        actual: Actual value (None for upcoming games).
    """

    player_id: str
    player_name: str
    game_id: str
    season: int
    week: int
    predicted: float
    actual: float | None = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_props(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute prop model evaluation metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with MAE, RMSE, R², and median absolute error.
    """
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    median_ae = float(np.median(np.abs(residuals)))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
    }


# ---------------------------------------------------------------------------
# Minimum attempts filter — prevents modeling garbage-time / low-usage games
# ---------------------------------------------------------------------------

_MIN_ATTEMPTS: Final[dict[str, tuple[str, int]]] = {
    "passing_yards": ("attempts", 10),
    "rushing_yards": ("carries", 5),
    "receiving_yards": ("targets", 2),
}


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class PropTrainer(ABC):
    """Abstract base class for prop model trainers.

    Subclasses implement:
        - ``spec`` — PropModelSpec describing the model
        - ``_feature_columns()`` — which feature columns to use
        - ``_build_features()`` — assemble the feature matrix
        - ``_fit()`` — train the underlying sklearn/xgb model

    The base class handles:
        - Data loading and filtering
        - Train/holdout splitting (TimeSeriesSplit)
        - Evaluation
        - Artifact persistence
    """

    @property
    @abstractmethod
    def spec(self) -> PropModelSpec:
        """Return the model specification."""
        ...

    @abstractmethod
    def _feature_columns(self) -> list[str]:
        """Return the ordered list of feature column names."""
        ...

    @abstractmethod
    def _build_features(self, df: DataFrame) -> DataFrame:
        """Build the feature matrix from enriched player game logs.

        Args:
            df: Player game logs with rolling + matchup features.

        Returns:
            DataFrame with feature columns and target column.
        """
        ...

    @abstractmethod
    def _fit(
        self,
        x_train: DataFrame,
        y_train: pd.Series,
        x_val: DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Fit the model and return hyperparameters used.

        Args:
            x_train: Training features.
            y_train: Training target.
            x_val: Validation features.
            y_val: Validation target.

        Returns:
            Dict of hyperparameters for metadata recording.
        """
        ...

    @abstractmethod
    def _predict(self, x: DataFrame) -> np.ndarray:
        """Generate predictions from fitted model.

        Args:
            x: Feature matrix.

        Returns:
            Array of predicted values.
        """
        ...

    def _load_data(self, *, repo: Path | None = None) -> DataFrame:
        """Load player game logs with rolling + matchup features.

        Chains the rolling and matchup feature builders, then filters
        to the relevant positions and applies minimum attempt thresholds.
        """
        resolved_repo = repo or get_settings().repo_root

        # Build rolling features (includes skill position filter)
        df = build_player_rolling_features(repo=resolved_repo)

        # Build matchup features separately and join
        matchup_df = build_matchup_features(repo=resolved_repo)
        matchup_cols = [c for c in matchup_df.columns if c.startswith("opp_")]
        join_keys = ["player_id", "season", "week"]

        df = df.merge(
            # pyrefly: ignore [no-matching-overload]
            matchup_df[join_keys + matchup_cols].drop_duplicates(subset=join_keys),
            on=join_keys,
            how="left",
        )

        # Filter to relevant positions
        df = df[df["position"].isin(self.spec.position_filter)].copy()

        # Apply minimum attempt threshold
        target = self.spec.target_col
        if target in _MIN_ATTEMPTS:
            volume_col, min_val = _MIN_ATTEMPTS[target]
            if volume_col in df.columns:
                before = len(df)
                df = df[df[volume_col] >= min_val].copy()
                logger.info(
                    "Filtered %s >= %d: %d → %d rows",
                    volume_col,
                    min_val,
                    before,
                    len(df),
                )

        # Drop rows where target is NaN
        # pyrefly: ignore [no-matching-overload]
        df = df.dropna(subset=[target])

        logger.info(
            "Loaded %d rows for %s (%s)",
            len(df),
            self.spec.name,
            self.spec.target_col,
        )
        return df

    def train(self, *, repo: Path | None = None) -> PropModelMetadata:
        """Full training pipeline: load data, split, fit, evaluate.

        Uses TimeSeriesSplit for temporal validation, consistent with
        the game prediction models.

        Returns:
            PropModelMetadata with evaluation metrics.
        """
        df = self._load_data(repo=repo)
        features_df = self._build_features(df)
        feature_cols = self._feature_columns()

        # Ensure chronological order
        features_df = features_df.sort_values(["season", "week"]).reset_index(drop=True)

        # Drop rows with NaN in features or target
        target = self.spec.target_col
        required_cols = [*feature_cols, target]
        features_df = features_df.dropna(subset=required_cols)

        x = features_df[feature_cols]
        y = features_df[target]

        logger.info(
            "Training %s: %d rows, %d features",
            self.spec.name,
            len(x),
            len(feature_cols),
        )

        # TimeSeriesSplit — same approach as game models
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(x))
        train_idx, val_idx = splits[-1]  # Use last split for final eval

        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit
        params = self._fit(x_train, y_train, x_val, y_val)

        # Evaluate on holdout
        y_pred = self._predict(x_val)
        metrics = evaluate_props(np.asarray(y_val), y_pred)

        logger.info(
            "%s holdout: MAE=%.1f, RMSE=%.1f, R²=%.3f (n=%d)",
            self.spec.name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            len(y_val),
        )

        # Determine season ranges
        train_seasons = sorted(features_df.iloc[train_idx]["season"].unique().tolist())
        holdout_seasons = sorted(features_df.iloc[val_idx]["season"].unique().tolist())

        return PropModelMetadata(
            model_name=self.spec.name,
            trained_at=datetime.now(UTC).isoformat(),
            target_col=self.spec.target_col,
            holdout_mae=metrics["mae"],
            holdout_rmse=metrics["rmse"],
            holdout_r2=metrics["r2"],
            training_seasons=train_seasons,
            holdout_seasons=holdout_seasons,
            parameters=params,
            feature_columns=feature_cols,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_val),
        )
