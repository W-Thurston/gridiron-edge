# src/gridiron_edge/models/prop_prediction/base.py

"""Base infrastructure for prop prediction models.

Prop models predict continuous player stats (passing yards, rushing yards,
etc.) rather than binary game outcomes. They share the player feature
pipeline (rolling stats + matchup features) but have their own training,
evaluation, and prediction interfaces.

Architecture:
    - ``PropModelSpec`` - metadata describing a prop model
    - ``PropModelResult`` - standardized prediction output
    - ``PropTrainer`` - base class for training prop models
    - Evaluation uses MAE/RMSE/R² instead of Brier/AUC/ECE

Adding a new prop model:
    1. Subclass ``PropTrainer``
    2. Implement ``_feature_columns()``, ``_build_features()``, ``_fit()``
    3. Register in the prop model registry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import product
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.features.player.builder import build_prop_features
from gridiron_edge.models.artifact import BaseModelMetadata

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PropModelType(StrEnum):
    """Supported prop model algorithm types."""

    ELASTICNET = "elasticnet"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


@dataclass(frozen=True)
class PropModelSpec:
    """Metadata describing a prop model's identity.

    Attributes:
        name: Unique key (e.g. ``"qb_pass_yards"``).
        target_col: Column in player game logs that is the prediction target.
        position_filter: Position(s) this model applies to.
        description: Human-readable description.
        clip_lo: Minimum predicted value (predictions clipped to this floor).
        clip_hi: Maximum predicted value (predictions clipped to this ceiling).
        trainable: Whether this model has an explicit training step.
            Defaults to ``True`` since all prop trainers implement the
            Trainable protocol.
        exclude_feature_prefixes: Feature-name prefixes to strip from
            the feature set before any NaN-rate filtering. Used to
            remove structurally-invalid feature families for the
            position (e.g. ``receiving_*`` for QB models, ``passing_*``
            for non-QB models). Sporadic non-null values in
            structurally-invalid features (trick plays, halfback
            passes) can push a column's NaN rate just under the 50%
            threshold used by ``_filter_and_split``, causing the
            feature to be kept for training but exhibit ~100% NaN in
            the prediction slice, which collapses the holdout. This
            list closes that hole by dropping such families outright,
            independent of NaN rate.
    """

    name: str
    target_col: str
    position_filter: list[str]
    description: str = ""
    clip_lo: float = 0.0
    clip_hi: float = 1000.0
    trainable: bool = True
    exclude_feature_prefixes: tuple[str, ...] = ()


@dataclass(kw_only=True)
class PropModelMetadata(BaseModelMetadata):
    """Metadata recorded alongside a trained prop model artifact.

    Adds the prop-only ``target_col`` field. Holdout metrics live in
    :attr:`BaseModelMetadata.metrics`.
    """

    kind: str = "prop"
    target_col: str


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
# Minimum attempts filter - prevents modeling garbage-time / low-usage games
# ---------------------------------------------------------------------------

_MIN_ATTEMPTS: Final[dict[str, tuple[str, int]]] = {
    "passing_yards": ("attempts", 10),
    "rushing_yards": ("carries", 5),
    "receiving_yards": ("targets", 2),
}

# Number of TimeSeriesSplit folds for inner CV during HP search.
# Matches the GamesTrainer convention (also _CV_FOLDS = 5).
_CV_FOLDS: Final[int] = 5

# ---------------------------------------------------------------------------
# Model factory + hyperparameter grids
# ---------------------------------------------------------------------------


def _create_model(model_type: PropModelType) -> tuple[Any, Any]:
    """Create a model instance and optional scaler.

    Returns:
        Tuple of (model_instance, scaler_or_none).
        ElasticNet gets a StandardScaler; tree models get None.
    """
    if model_type == PropModelType.ELASTICNET:
        from sklearn.linear_model import ElasticNet  # pyrefly: ignore [missing-import]
        from sklearn.preprocessing import StandardScaler  # pyrefly: ignore [missing-import]

        return ElasticNet(max_iter=10_000), StandardScaler()

    if model_type == PropModelType.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestRegressor  # pyrefly: ignore [missing-import]

        return RandomForestRegressor(n_jobs=-1, random_state=42), None

    if model_type == PropModelType.XGBOOST:
        from xgboost import XGBRegressor  # pyrefly: ignore [missing-import]

        return XGBRegressor(n_jobs=-1, random_state=42, verbosity=0), None

    msg: str = f"Unknown model type: {model_type}"
    raise ValueError(msg)


def _get_param_grid(model_type: PropModelType) -> list[dict[str, Any]]:
    """Return the hyperparameter search grid for the given model type."""
    if model_type == PropModelType.ELASTICNET:
        return [
            {"alpha": a, "l1_ratio": r}
            for a, r in product(
                [0.001, 0.01, 0.1, 1.0, 10.0],
                [0.1, 0.3, 0.5, 0.7, 0.9],
            )
        ]

    if model_type == PropModelType.RANDOM_FOREST:
        return [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n, d, leaf in product(
                [100, 300, 500],
                [8, 12, 16, None],
                [5, 10, 20],
            )
        ]

    if model_type == PropModelType.XGBOOST:
        return [
            {
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "subsample": s,
            }
            for n, d, lr, s in product(
                [100, 300, 500],
                [4, 6, 8],
                [0.01, 0.05, 0.1],
                [0.8, 1.0],
            )
        ]

    msg: str = f"Unknown model type: {model_type}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class PropTrainer(ABC):
    """Base class for prop model trainers.

    Subclasses implement only ``spec`` - all training and prediction
    logic lives here. The factory pattern (``_create_model``) handles
    ElasticNet, RandomForest, and XGBoost transparently.
    """

    _model: Any = None
    _scaler: Any = None

    @property
    @abstractmethod
    def spec(self) -> PropModelSpec:
        """Return the model specification."""
        ...

    def _feature_columns(self) -> list[str]:
        """Return the prop feature column list.

        Built programmatically from component modules via PROP_FEATURE_COLS.
        ElasticNet handles feature selection - no manual per-model curation
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

    def _fit(
        self,
        x_train: DataFrame,
        y_train: Series,
        model_type: PropModelType = PropModelType.ELASTICNET,
    ) -> dict[str, Any]:
        """Fit the model via HP search with TimeSeriesSplit inner CV.

        For each hyperparameter combination, trains on TimeSeriesSplit folds
        of the training data and averages fold MAE. Selects the combination
        with the lowest mean fold MAE, then retrains on the full training
        set with those params.

        The holdout set is not touched by this method. The caller
        (``PropTrainer.train``) evaluates the refit model on the holdout
        exactly once for honest metrics.

        Args:
            x_train: Training features (chronologically sorted).
            y_train: Training target (aligned with x_train).
            model_type: Algorithm to use.

        Returns:
            Dict of best hyperparameters plus ``cv_mae`` (mean fold MAE
            of the selected combination) for metadata recording.
        """
        # pyrefly: ignore [missing-import]
        from sklearn.model_selection import TimeSeriesSplit

        # pyrefly: ignore [missing-import, untyped-import]
        from tqdm import tqdm

        grid: list[dict[str, Any]] = _get_param_grid(model_type)
        tscv = TimeSeriesSplit(n_splits=_CV_FOLDS)

        best_cv_mae: float = float("inf")
        best_params: dict[str, Any] = {}

        bar: tqdm[dict[str, Any]] = tqdm(
            grid,
            desc=f"  {self.spec.name} ({model_type})",
            unit="combo",
            ncols=100,
            colour="cyan",
        )
        for params in bar:
            fold_scores: list[float] = []

            for train_idx, val_idx in tscv.split(x_train):
                x_tr_fold = x_train.iloc[train_idx]
                y_tr_fold = y_train.iloc[train_idx]
                x_va_fold = x_train.iloc[val_idx]
                y_va_fold = y_train.iloc[val_idx]

                model, scaler = _create_model(model_type)
                model.set_params(**params)

                if scaler is not None:
                    x_tr_arr = scaler.fit_transform(x_tr_fold)
                    x_va_arr = scaler.transform(x_va_fold)
                else:
                    x_tr_arr = x_tr_fold.values
                    x_va_arr = x_va_fold.values

                model.fit(x_tr_arr, y_tr_fold)
                preds: ndarray = model.predict(x_va_arr)
                fold_mae: float = float(np.mean(np.abs(y_va_fold.values - preds)))
                fold_scores.append(fold_mae)

            if not fold_scores:
                continue

            mean_cv_mae: float = float(np.mean(fold_scores))

            if mean_cv_mae < best_cv_mae:
                best_cv_mae = mean_cv_mae
                best_params = params
                bar.set_postfix_str(f"best CV MAE={best_cv_mae:.1f}")

        if not best_params:
            msg: str = (
                f"{self.spec.name}/{model_type.value}: HP search produced no "
                f"valid CV folds. Training set may be too small for "
                f"{_CV_FOLDS}-fold TimeSeriesSplit."
            )
            raise RuntimeError(msg)

        # Refit best params on the full training set
        model, scaler = _create_model(model_type)
        model.set_params(**best_params)

        x_tr_full = scaler.fit_transform(x_train) if scaler is not None else x_train.values
        model.fit(x_tr_full, y_train)

        self._model = model
        self._scaler = scaler

        # Include CV score in returned params for metadata observability
        return {**best_params, "cv_mae": round(best_cv_mae, 6)}

    def _predict(self, x: DataFrame) -> ndarray:
        """Generate predictions from fitted model with spec-based clipping.

        Args:
            x: Feature matrix (columns must match training features).

        Returns:
            Array of predicted values, clipped to [spec.clip_lo, spec.clip_hi].
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        x_scaled = self._scaler.transform(x) if self._scaler is not None else x.values

        preds: ndarray = self._model.predict(x_scaled)
        return np.clip(preds, self.spec.clip_lo, self.spec.clip_hi)

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

    def _prepare_features(
        self,
        *,
        repo: Path | None = None,
    ) -> tuple[DataFrame, list[str]]:
        """Load raw data, build features, return (features_df, available).

        Shared entry point for train() and train_through(). Returns the
        full features DataFrame (all seasons) sorted chronologically and
        the intersection of self._feature_columns() with columns actually
        present in the DataFrame.

        Args:
            repo: Repository root override.

        Returns:
            Tuple of (features_df, available_feature_columns).
        """
        df: DataFrame = self._load_data(repo=repo)
        features_df: DataFrame = self._build_features(df)
        features_df = features_df.sort_values(["season", "week"]).reset_index(drop=True)
        feature_cols: list[str] = self._feature_columns()

        # Strip structurally-invalid feature families before any NaN-based
        # filtering. See PropModelSpec.exclude_feature_prefixes docstring
        # for rationale.
        excluded_prefixes = self.spec.exclude_feature_prefixes
        if excluded_prefixes:
            before = len(feature_cols)
            feature_cols = [
                c for c in feature_cols if not any(c.startswith(p) for p in excluded_prefixes)
            ]
            n_excluded = before - len(feature_cols)
            if n_excluded:
                logger.info(
                    "%s: excluded %d features by prefix (%s)",
                    self.spec.name,
                    n_excluded,
                    ", ".join(excluded_prefixes),
                )

        available: list[str] = [c for c in feature_cols if c in features_df.columns]
        return features_df, available

    def _filter_and_split(
        self,
        *,
        features_df: DataFrame,
        available_features: list[str],
        train_mask: Series,
        hold_mask: Series,
        context: str,
    ) -> tuple[DataFrame, DataFrame, list[str]]:
        """Apply era-aware NaN policy and split into train/holdout.

        NaN rates are measured on the training slice (rows where
        ``train_mask`` is True) so era-boundary features like
        ``passing_cpoe`` are correctly excluded when the training slice
        predates the feature's existence. Then dropna on the full
        DataFrame and reapply the masks against the surviving index.

        Args:
            features_df: Full features DataFrame (all seasons).
            available_features: Feature columns present in features_df.
            train_mask: Boolean Series identifying training rows.
            hold_mask: Boolean Series identifying holdout rows.
            context: Human-readable description for logs and errors
                (e.g. "cutoff_season=2015").

        Returns:
            Tuple of (train_df, hold_df, usable_features).

        Raises:
            ValueError: If the training slice is empty.
        """
        target = self.spec.target_col
        train_slice = features_df.loc[train_mask, :]
        if train_slice.empty:
            msg: str = f"No training rows for {self.spec.name} ({context})."
            raise ValueError(msg)

        nan_rates: Series = train_slice[available_features].isna().mean()
        usable: list[str] = [c for c in available_features if nan_rates[c] < 0.5]
        dropped: set[str] = set(available_features) - set(usable)
        if dropped:
            logger.info(
                "Dropped %d features with >50%% NaN in train slice for %s (%s): %s",
                len(dropped),
                self.spec.position_filter,
                context,
                sorted(dropped)[:10],
            )

        n_before: int = len(features_df)
        cleaned: DataFrame = features_df.dropna(subset=[*usable, target])
        logger.info(
            "NaN drop: %d → %d rows (%d dropped, %d usable features)",
            n_before,
            len(cleaned),
            n_before - len(cleaned),
            len(usable),
        )

        surviving_train = train_mask.loc[cleaned.index]
        surviving_hold = hold_mask.loc[cleaned.index]
        train_df = cleaned.loc[surviving_train, :]
        hold_df = cleaned.loc[surviving_hold, :]

        return train_df, hold_df, usable

    def _fit_and_build_metadata(
        self,
        *,
        train_df: DataFrame,
        hold_df: DataFrame,
        usable_features: list[str],
        model_type: PropModelType,
        context: str,
    ) -> PropModelMetadata:
        """Fit the model, evaluate on holdout, and return metadata.

        Shared by ``train()`` and ``train_through()``. Both split the data
        differently but the fit-and-evaluate path is identical from here on.

        Args:
            train_df: Training rows (post-split).
            hold_df: Holdout rows (post-split).
            usable_features: Feature columns to fit on. Recorded verbatim
                in the returned metadata's ``feature_columns`` field as
                the fit-time source of truth for prediction.
            model_type: Algorithm to fit.
            context: Human-readable description for logs.

        Returns:
            ``PropModelMetadata`` describing the fitted model.

        Raises:
            ValueError: If the holdout slice is empty.
        """
        target = self.spec.target_col
        x_train = train_df.loc[:, usable_features]
        y_train: Series = train_df[target]
        x_hold = hold_df.loc[:, usable_features]
        y_hold: Series = hold_df[target]

        logger.info(
            "Training %s (%s): %d train rows, %d holdout rows, %d features",
            self.spec.name,
            context,
            len(x_train),
            len(x_hold),
            len(usable_features),
        )

        if len(x_hold) == 0:
            msg: str = f"No holdout rows for {self.spec.name} ({context})."
            raise ValueError(msg)

        params = self._fit(x_train, y_train, model_type=model_type)
        y_pred = self._predict(x_hold)
        metrics: dict[str, float] = evaluate_props(np.asarray(y_hold), y_pred)

        logger.info(
            "%s holdout (%s): MAE=%.1f, RMSE=%.1f, R²=%.3f (n=%d)",
            self.spec.name,
            context,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            len(y_hold),
        )

        return PropModelMetadata(
            model_name=self.spec.name,
            model_type=model_type.value,
            task="regression",
            trained_at=datetime.now(UTC).isoformat(),
            target_col=self.spec.target_col,
            training_seasons=[f"{y}-{y + 1}" for y in sorted(train_df["season"].unique().tolist())],
            holdout_seasons=[f"{y}-{y + 1}" for y in sorted(hold_df["season"].unique().tolist())],
            parameters=params,
            feature_columns=usable_features,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_hold),
            metrics={
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            },
        )

    def predict_with_meta(
        self,
        df: DataFrame,
        meta: PropModelMetadata,
    ) -> tuple[ndarray, DataFrame]:
        """Predict on a slice using the fit-time feature list from meta.

        This is the only sanctioned prediction path outside of the
        internal fit-and-evaluate flow. Callers pass the metadata
        returned by ``train()`` / ``train_through()`` (or loaded via
        ``ArtifactStore.read_metadata``) so the exact feature list the
        model was fit on drives prediction. This closes the drift bug
        where independent NaN filtering at predict time can disagree
        with the fit-time feature set and trigger sklearn's feature-name
        check.

        Rows in ``df`` with NaN in any fit-time feature are dropped
        before prediction. Callers can use the returned DataFrame's
        index to align predictions with their input.

        Args:
            df: Data slice to predict on. Must contain all columns in
                ``meta.feature_columns``.
            meta: Metadata from the model's training run.

        Returns:
            Tuple of (predictions ndarray, df filtered to rows that
            survived NaN-drop and were predicted).

        Raises:
            RuntimeError: If ``meta.feature_columns`` contains a column
                not present in ``df`` — a feature-pipeline regression.
        """
        usable: list[str] = list(meta.feature_columns)
        missing: list[str] = [c for c in usable if c not in df.columns]
        if missing:
            msg: str = (
                f"{meta.model_name}/{meta.model_type}: prediction slice "
                f"missing columns present at fit time: {missing}. Feature "
                f"pipeline changed between train and predict."
            )
            raise RuntimeError(msg)

        clean: DataFrame = df.dropna(subset=usable)
        if clean.empty:
            return np.array([]), clean

        preds = self._predict(clean.loc[:, usable])
        return preds, clean

    def train(
        self,
        *,
        model_type: PropModelType = PropModelType.ELASTICNET,
        repo: Path | None = None,
    ) -> PropModelMetadata:
        """Full training pipeline using the HOLDOUT_SEASONS split."""
        holdout_ints: set[int] = {int(s.split("-")[0]) for s in HOLDOUT_SEASONS}
        features_df, available = self._prepare_features(repo=repo)

        train_mask: Series[bool] = ~features_df["season"].isin(holdout_ints)
        hold_mask: Series[bool] = features_df["season"].isin(holdout_ints)
        context: str = f"HOLDOUT_SEASONS={sorted(holdout_ints)}"

        train_df, hold_df, usable = self._filter_and_split(
            features_df=features_df,
            available_features=available,
            train_mask=train_mask,
            hold_mask=hold_mask,
            context=context,
        )

        return self._fit_and_build_metadata(
            train_df=train_df,
            hold_df=hold_df,
            usable_features=usable,
            model_type=model_type,
            context=context,
        )

    def train_through(
        self,
        *,
        cutoff_season: int,
        model_type: PropModelType = PropModelType.ELASTICNET,
        repo: Path | None = None,
    ) -> PropModelMetadata:
        """Walk-forward training using seasons strictly before ``cutoff_season``.

        ``cutoff_season`` itself becomes the holdout. Used by
        ``gridiron props backfill`` and the composite ``full-retrain``
        prop backfill stage.
        """
        features_df, available = self._prepare_features(repo=repo)

        train_mask: Series[bool] = features_df["season"] < cutoff_season
        hold_mask: Series[bool] = features_df["season"] == cutoff_season
        context: str = f"cutoff_season={cutoff_season}"

        train_df, hold_df, usable = self._filter_and_split(
            features_df=features_df,
            available_features=available,
            train_mask=train_mask,
            hold_mask=hold_mask,
            context=context,
        )

        return self._fit_and_build_metadata(
            train_df=train_df,
            hold_df=hold_df,
            usable_features=usable,
            model_type=model_type,
            context=context,
        )

    def train_and_save(
        self,
        *,
        model_type: PropModelType = PropModelType.ELASTICNET,
        repo: Path | None = None,
    ) -> PropModelMetadata:
        """Train using the standard HOLDOUT_SEASONS split and persist the artifact.

        Convenience wrapper for the prop CLI workflows that need a trained,
        persisted model. Mirrors the game-side artifact persistence pattern:
        one call produces a fitted model, its scaler, and an on-disk
        artifact that downstream commands (``projections_cmd``) can reload
        without retraining.

        Args:
            model_type: Algorithm to train.
            repo: Repository root override. Defaults to
                ``get_settings().repo_root``.

        Returns:
            ``PropModelMetadata`` for the persisted artifact.
        """
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.models.artifact import ArtifactStore

        resolved_repo: Path = repo or get_settings().repo_root

        meta: PropModelMetadata = self.train(model_type=model_type, repo=resolved_repo)

        ArtifactStore(resolved_repo).save(
            metadata=meta,
            model_obj=self._model,
            scaler=self._scaler,
            overwrite=True,
        )

        return meta

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for the default algorithm.

        Mirrors the game-side ``is_trained`` semantics so prop trainers
        satisfy the ``Trainable`` protocol. Defaults to the
        ``PropModelType.ELASTICNET`` artifact since the prop CLI uses
        that algorithm as the canonical baseline.
        """
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.models.artifact import ArtifactStore

        resolved_repo = repo or get_settings().repo_root
        return ArtifactStore(resolved_repo).is_trained(
            self.spec.name,
            PropModelType.ELASTICNET.value,
        )
