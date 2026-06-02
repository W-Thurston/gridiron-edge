# src/gridiron_edge/models/game_prediction/total.py
"""Total points regression model.

Trains Random Forest and XGBoost regressors to predict the combined
score of an NFL game.  Uses the same expanded feature set as the
win-probability models but targets ``actual_total = PTS_WINNER +
PTS_LOSER`` instead of ``RESULT``.

This is a supporting model — it feeds into ``enrich_predictions()``
rather than operating through the ``PredictorRegistry``.

Public API:
    train_total_model    Train and save a total-points regressor
    predict_total        Load model and predict totals for a DataFrame
    load_total_model     Load a trained total model from the artifact store
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestRegressor

# pyrefly: ignore [missing-import]
from sklearn.model_selection import TimeSeriesSplit

# pyrefly: ignore [untyped-import]
from tqdm import tqdm

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import load_games, load_modeling_file
from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
from gridiron_edge.models.artifact import ArtifactStore, ModelMetadata
from gridiron_edge.models.game_prediction._features import (
    HOLDOUT_SEASONS,
    _make_expanded_features,
)

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hyperparameter search space — mirrors tree.py RF pattern but for regression.
_RF_PARAM_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_leaf": [4, 8, 12, 16],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}

_N_ITER: int = 50
_CV_FOLDS: int = 5


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

    Returns:
        Tuple of (x_train, y_train, x_hold, y_hold,
        train_seasons, holdout_seasons).
    """
    df: DataFrame = load_modeling_file(repo)
    games: DataFrame = load_games(repo)

    # Build total lookup: GAME_ID → actual_total
    games_lookup: DataFrame = games.dropna(subset=["PTS_WINNER", "PTS_LOSER"]).copy()
    games_lookup: DataFrame = games_lookup.drop_duplicates(subset=["GAME_ID"])
    games_lookup["actual_total"] = games_lookup["PTS_WINNER"] + games_lookup["PTS_LOSER"]

    # Join to modeling DataFrame
    df = df.merge(
        games_lookup[["GAME_ID", "actual_total"]],
        on="GAME_ID",
        how="inner",
    )

    # Drop rows with missing total (shouldn't happen, but defensive)
    df = df.dropna(subset=["actual_total"])

    # Build features (same as win model)
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

    # pyrefly: ignore [bad-return]
    return (
        features.loc[train_mask],
        y.loc[train_mask],
        features.loc[hold_mask],
        y.loc[hold_mask],
        sorted(df.loc[train_mask, "YEAR"].unique().tolist()),
        sorted(df.loc[hold_mask, "YEAR"].unique().tolist()),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_total_model(
    model_version: str = "total_rf_v1",
    *,
    repo: Path | None = None,
    n_iter: int = _N_ITER,
    cv_folds: int = _CV_FOLDS,
) -> dict[str, Any]:
    """Train a Random Forest regressor for total points prediction.

    Uses randomized hyperparameter search with cross-validated MAE,
    matching the pattern in ``tree.py`` but for regression.

    Args:
        model_version: Artifact name (e.g. ``"total_rf_v1"``).
        repo: Repository root.  Defaults to ``get_settings().repo_root``.
        n_iter: Number of random hyperparameter samples.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with training metadata (best_params, train_mae, holdout_mae,
        n_train, n_holdout, train_seasons, holdout_seasons).
    """
    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    x_train, y_train, x_hold, y_hold, train_szns, hold_szns = _prepare_total_data(resolved_repo)

    # pyrefly: ignore [bad-assignment]
    rng: Generator = np.random.default_rng(42)
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    best_mae: float = float("inf")
    best_params: dict[str, Any] = {}
    best_model: RandomForestRegressor | None = None

    for i in tqdm(range(n_iter), desc="Total model HP search", unit="iter"):
        # Sample hyperparameters
        # pyrefly: ignore [missing-attribute]
        sampled: dict[str, Any] = {k: v[rng.integers(len(v))] for k, v in _RF_PARAM_SPACE.items()}

        # Cross-validated MAE
        fold_maes: list[float] = []
        for train_idx, val_idx in tscv.split(x_train):
            xt, xv = x_train.iloc[train_idx], x_train.iloc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            rf = RandomForestRegressor(random_state=42, n_jobs=-1, **sampled)
            rf.fit(xt, yt)
            preds = rf.predict(xv)
            fold_mae = float(np.mean(np.abs(preds - yv)))
            fold_maes.append(fold_mae)

        mean_mae = float(np.mean(fold_maes))

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = dict(sampled)

            logger.info(
                "train_total_model: iter %d/%d  MAE=%.3f  params=%s",
                i + 1,
                n_iter,
                mean_mae,
                sampled,
            )

    # Retrain best model on full training set
    best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(x_train, y_train)

    # Evaluate on holdout
    train_preds = best_model.predict(x_train)
    hold_preds = best_model.predict(x_hold)
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    hold_mae = float(np.mean(np.abs(hold_preds - y_hold)))
    train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))
    hold_rmse = float(np.sqrt(np.mean((hold_preds - y_hold) ** 2)))

    logger.info(
        "train_total_model: DONE  train_MAE=%.3f  holdout_MAE=%.3f  "
        "train_RMSE=%.3f  holdout_RMSE=%.3f",
        train_mae,
        hold_mae,
        train_rmse,
        hold_rmse,
    )

    # Save
    metadata = ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_szns,
        holdout_seasons=hold_szns,
        holdout_brier=hold_mae,  # repurpose: MAE for regression models
    )

    store.save(model_version, best_model, metadata=metadata)
    logger.info("train_total_model: saved %s", model_version)

    return {
        "best_params": best_params,
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "cv_mae": best_mae,
        "train_mae": train_mae,
        "holdout_mae": hold_mae,
        "train_rmse": train_rmse,
        "holdout_rmse": hold_rmse,
        "n_train": len(x_train),
        "n_holdout": len(x_hold),
        "train_seasons": train_szns,
        "holdout_seasons": hold_szns,
        "n_features": x_train.shape[1],
        "mean_total_train": y_train.mean(),
        "mean_total_holdout": y_hold.mean(),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def load_total_model(
    model_version: str = "total_rf_v1",
    *,
    repo: Path | None = None,
) -> RandomForestRegressor | None:
    """Load a trained total model from the artifact store.

    Returns ``None`` if the model has not been trained yet.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    if not store.is_trained(model_version):
        logger.debug("load_total_model: %s not found", model_version)
        return None

    model = store.load(model_version)
    logger.info("load_total_model: loaded %s", model_version)
    return model


def predict_total(
    df: pd.DataFrame,
    model_version: str = "total_rf_v1",
    *,
    repo: Path | None = None,
) -> pd.Series:
    """Predict total points for each game in a modeling DataFrame.

    Args:
        df: Modeling DataFrame (same format as ``load_modeling_file()``
            output).  Must contain the expanded feature columns.
        model_version: Total model artifact name.
        repo: Repository root.

    Returns:
        Series of predicted totals, indexed like *df*.

    Raises:
        FileNotFoundError: If the total model has not been trained.
    """
    model = load_total_model(model_version, repo=repo)
    if model is None:
        raise FileNotFoundError(
            f"Total model '{model_version}' not found. Run train_total_model() first."
        )

    features: DataFrame = _make_expanded_features(df)

    # Handle missing features gracefully (same as win model)
    valid = features.notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.sum() > 0:
        preds.loc[valid] = model.predict(features.loc[valid])

    logger.info(
        "predict_total: predicted %d/%d games (model=%s)",
        valid.sum(),
        len(df),
        model_version,
    )

    return preds
