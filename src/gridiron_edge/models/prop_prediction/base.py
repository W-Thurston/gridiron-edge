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
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.features.player.builder import build_prop_features

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
    r2: float = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
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
# Universal feature columns — shared by all prop models.
# Built programmatically so they stay in sync with rolling + matchup modules.
# ElasticNet handles feature selection; no manual per-model curation needed.
# ---------------------------------------------------------------------------


def _build_universal_features() -> list[str]:
    """Build the universal feature column list from rolling + matchup + context."""
    from gridiron_edge.features.player.matchup import _MATCHUP_STATS
    from gridiron_edge.features.player.rolling import DEFAULT_WINDOWS, ROLLING_STAT_COLS

    cols: list[str] = []

    # Rolling features: {stat}_L{WIND_SPEED_MPHow}_{agg}
    for stat in ROLLING_STAT_COLS:
        for w in DEFAULT_WINDOWS:
            cols.append(f"{stat}_L{w}_mean")
            cols.append(f"{stat}_L{w}_std")

    # Matchup features: opp_{name}_allowed_L6 + rank
    for _, _, name in _MATCHUP_STATS:
        cols.append(f"opp_{name}_allowed_L6")
        cols.append(f"opp_{name}_allowed_rank_L6")

    # Game context features
    cols.extend(
        [
            "implied_team_total",
            "spread_line",
            "OVER_UNDER",
            "is_home",
            "roof_dome",
            "surface_turf",
            "TEMP_F",
            "WIND_SPEED_MPH",
            "rest_days",
            "opp_rest_days",
            "rest_diff",
            "DIV_GAME",
        ]
    )

    return cols


UNIVERSAL_FEATURE_COLS: Final[list[str]] = _build_universal_features()

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

    def _feature_columns(self) -> list[str]:
        """Return the prop feature column list.

        Built programmatically from component modules via PROP_FEATURE_COLS.
        ElasticNet handles feature selection — no manual per-model curation
        needed. Subclasses may override if they have a reason to diverge.
        """
        return list(PROP_FEATURE_COLS)

    def _build_features(self, df: DataFrame) -> DataFrame:
        """Build the feature matrix from enriched player game logs.

        Default implementation returns the DataFrame as-is since
        build_prop_features() already handles all feature engineering.
        Subclasses may override to add position-specific derived features.

        Args:
            df: Player game logs with all prop features.

        Returns:
            DataFrame with feature columns and target column.
        """
        return df

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
        """Load player game logs with all prop features.

        Uses the unified feature builder, then applies minimum attempt
        thresholds and drops rows where the target is NaN.

        Args:
            repo: Repository root override.

        Returns:
            DataFrame filtered to relevant positions with all features.
        """
        df: DataFrame = build_prop_features(
            position_filter=self.spec.position_filter,
            repo=repo,
        )

        # Apply minimum attempt threshold
        target: str = self.spec.target_col
        if target in _MIN_ATTEMPTS:
            volume_col, min_val = _MIN_ATTEMPTS[target]
            if volume_col in df.columns:
                before: int = len(df)
                df = df.loc[df[volume_col] >= min_val, :].copy()
                logger.info(
                    "Filtered %s >= %d: %d → %d rows",
                    volume_col,
                    min_val,
                    before,
                    len(df),
                )

        # Drop rows where target is NaN
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

        Uses HOLDOUT_SEASONS for the train/holdout split, consistent
        with the game prediction models.

        Returns:
            PropModelMetadata with evaluation metrics.
        """
        # Parse HOLDOUT_SEASONS ("2023-2024" → 2023) for integer season column
        holdout_ints: set[int] = {int(s.split("-")[0]) for s in HOLDOUT_SEASONS}

        df: DataFrame = self._load_data(repo=repo)
        features_df: DataFrame = self._build_features(df)
        feature_cols: list[str] = self._feature_columns()

        # Ensure chronological order
        features_df = features_df.sort_values(["season", "week"]).reset_index(drop=True)

        # Filter to available feature columns
        available_features: list[str] = [c for c in feature_cols if c in features_df.columns]

        # Position-aware NaN handling: only keep features with reasonable
        # coverage for this position, then drop rows with NaN in those.
        # This avoids losing all QB rows due to receiving features being
        # ~99% NaN, or all WR rows due to passing features.
        target: str = self.spec.target_col
        nan_rates: Series = features_df[available_features].isna().mean()
        usable_features: list[str] = [c for c in available_features if nan_rates[c] < 0.5]
        dropped_features = set(available_features) - set(usable_features)
        if dropped_features:
            logger.info(
                "Dropped %d features with >50%% NaN for %s: %s",
                len(dropped_features),
                self.spec.position_filter,
                sorted(dropped_features)[:10],
            )

        required_cols: list[str] = [*usable_features, target]
        n_before = len(features_df)
        features_df = features_df.dropna(subset=required_cols)
        logger.info(
            "NaN drop: %d → %d rows (%d dropped, %d usable features)",
            n_before,
            len(features_df),
            n_before - len(features_df),
            len(usable_features),
        )
        available_features = usable_features

        # HOLDOUT_SEASONS split — consistent with game models
        train_mask: Series = ~features_df["season"].isin(holdout_ints)
        hold_mask: Series = features_df["season"].isin(holdout_ints)

        train_df: DataFrame = features_df.loc[train_mask, :]
        hold_df: DataFrame = features_df.loc[hold_mask, :]

        x_train: DataFrame = train_df.loc[:, available_features]
        y_train: Series = train_df[target]
        x_hold: DataFrame = hold_df.loc[:, available_features]
        y_hold: Series = hold_df[target]

        logger.info(
            "Training %s: %d train rows (seasons %s), %d holdout rows (seasons %s), %d features",
            self.spec.name,
            len(x_train),
            sorted(train_df["season"].unique()),
            len(x_hold),
            sorted(hold_df["season"].unique()),
            len(available_features),
        )

        if len(x_hold) == 0:
            msg = (
                f"No holdout data for {self.spec.name}. "
                f"HOLDOUT_SEASONS={holdout_ints} produced 0 rows."
            )
            raise ValueError(msg)

        # Fit on training data, validated against holdout
        params: dict[str, Any] = self._fit(x_train, y_train, x_hold, y_hold)

        # Evaluate on holdout
        y_pred: ndarray = self._predict(x_hold)
        metrics: dict[str, float] = evaluate_props(np.asarray(y_hold), y_pred)

        logger.info(
            "%s holdout: MAE=%.1f, RMSE=%.1f, R²=%.3f (n=%d)",
            self.spec.name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            len(y_hold),
        )

        return PropModelMetadata(
            model_name=self.spec.name,
            trained_at=datetime.now(UTC).isoformat(),
            target_col=self.spec.target_col,
            holdout_mae=metrics["mae"],
            holdout_rmse=metrics["rmse"],
            holdout_r2=metrics["r2"],
            training_seasons=sorted(train_df["season"].unique().tolist()),
            holdout_seasons=sorted(hold_df["season"].unique().tolist()),
            parameters=params,
            feature_columns=available_features,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_hold),
        )
