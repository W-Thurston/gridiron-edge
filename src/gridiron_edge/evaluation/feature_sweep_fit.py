# src/gridiron_edge/evaluation/feature_sweep_fit.py

"""Hyperparameter search and expanding holdout refits for the feature sweep.

Mirrors the production trainer's search (same grids, budgets, sampled
combinations and fold geometry) on the matched training rows only, then
refits with the frozen hyperparameters expanding through each holdout season
(``DECISIONS.md`` D48-style). Nothing here reads a holdout label to choose a
hyperparameter, a calibration wrap or a column: the XGBoost isotonic wrap
uses production's ECE threshold measured on out-of-fold *training*
predictions, never the holdout's. Nothing here persists a model.

Public API
----------
FamilySpec              dataclass   - one (target, family) sweep unit
SearchOutcome           dataclass   - frozen hyperparameters and CV score
search_hyperparameters  train rows -> SearchOutcome
fit_predict             one fit + predictions
expanding_holdout_predictions   frozen params -> holdout predictions
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Any

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.metrics import expected_calibration_error
from gridiron_edge.models.game_prediction.base import (
    _CV_FOLDS,
    _ECE_CALIBRATION_THRESHOLD,
    GameModelType,
    _apply_params,
    _create_model,
    _get_param_grid,
    _n_iter_for,
    _resolve_cv_min_train_rows,
)

if TYPE_CHECKING:
    # pyrefly: ignore [missing-import]
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

SEARCH_SEED = 42
DEFAULT_SEED = 42

#: Supplies the feature matrix (all matched-frame rows) for an EPA window.
FeatureProvider = Callable[[int], DataFrame]


@dataclass(frozen=True)
class FamilySpec:
    """One sweep unit.

    Attributes:
        target: ``"win"`` or ``"total"``.
        model_type: Algorithm.
        feature_set_key: ``"combined"`` or ``"expanded"``.
    """

    target: str
    model_type: GameModelType
    feature_set_key: str

    @property
    def task(self) -> str:
        """``classification`` for Win, ``regression`` for Total."""
        return "classification" if self.target == "win" else "regression"

    @property
    def label(self) -> str:
        """Stable family label, for example ``win/logistic``."""
        return f"{self.target}/{self.model_type.value}"


@dataclass(frozen=True)
class SearchOutcome:
    """Frozen result of a training-only search.

    Attributes:
        params: Estimator hyperparameters (no ``epa_window``).
        epa_window: Selected EPA rolling window (4 for regression).
        cv_score: Mean CV Brier (Win) or MAE (Total) of the selected combo.
        xgb_recalibrate: Whether XGBoost gets the isotonic wrap, decided from
            out-of-fold training ECE.
        oof_ece: The out-of-fold training ECE behind that decision, if any.
    """

    params: dict[str, Any]
    epa_window: int
    cv_score: float
    xgb_recalibrate: bool
    oof_ece: float | None


def _set_seed(model: BaseEstimator, seed: int) -> None:
    """Set every ``random_state`` parameter of ``model`` (including nested)."""
    names = [k for k in model.get_params() if k.endswith("random_state")]
    if names:
        model.set_params(**dict.fromkeys(names, seed))


def _set_threads(model: BaseEstimator, threads: int) -> None:
    """Set every ``n_jobs`` parameter of ``model`` (including nested)."""
    names = [k for k in model.get_params() if k.endswith("n_jobs")]
    if names:
        model.set_params(**dict.fromkeys(names, threads))


def _predict(model: BaseEstimator, x: NDArray[np.float64], task: str) -> NDArray[np.float64]:
    if task == "classification":
        # pyrefly: ignore [missing-attribute]
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    # pyrefly: ignore [missing-attribute]
    return np.asarray(model.predict(x), dtype=float)


Predictor = Callable[[DataFrame | NDArray[np.float64]], NDArray[np.float64]]


def fit_estimator(
    spec: FamilySpec,
    params: dict[str, Any],
    x_train: DataFrame,
    y_train: Series,
    *,
    xgb_recalibrate: bool = False,
    seed: int = DEFAULT_SEED,
    threads: int | None = None,
) -> Predictor:
    """Fit once on the training rows and return a predictor.

    XGBoost classification is wrapped in production's isotonic calibration
    (``TimeSeriesSplit(3)``) when ``xgb_recalibrate`` is set. The predictor
    accepts a DataFrame or a plain array with the training column order.
    """
    # pyrefly: ignore [missing-import]
    from sklearn.calibration import CalibratedClassifierCV

    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import TimeSeriesSplit

    model, scaler = _create_model(spec.model_type, spec.task)
    _apply_params(model, params)
    _set_seed(model, seed)
    if threads is not None:
        _set_threads(model, threads)
    if (
        xgb_recalibrate
        and spec.model_type == GameModelType.XGBOOST
        and spec.task == "classification"
    ):
        model = CalibratedClassifierCV(model, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    x_tr = scaler.fit_transform(x_train) if scaler is not None else x_train.to_numpy()
    model.fit(x_tr, y_train.to_numpy())

    def predict(x_pred: DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        arr = x_pred.to_numpy() if isinstance(x_pred, DataFrame) else np.asarray(x_pred)
        arr = scaler.transform(arr) if scaler is not None else arr
        return _predict(model, arr, spec.task)

    return predict


def fit_predict(
    spec: FamilySpec,
    params: dict[str, Any],
    x_train: DataFrame,
    y_train: Series,
    x_pred: DataFrame,
    *,
    xgb_recalibrate: bool = False,
    seed: int = DEFAULT_SEED,
    threads: int | None = None,
) -> NDArray[np.float64]:
    """Fit once on the training rows and predict ``x_pred``."""
    predictor = fit_estimator(
        spec,
        params,
        x_train,
        y_train,
        xgb_recalibrate=xgb_recalibrate,
        seed=seed,
        threads=threads,
    )
    return predictor(x_pred)


def _cv_folds(
    spec: FamilySpec,
    params: dict[str, Any],
    x: DataFrame,
    y: Series,
    *,
    min_rows: int | None,
    threads: int | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Cross-validate one combo; return mean score and pooled OOF (pred, y)."""
    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=_CV_FOLDS)
    floor = _resolve_cv_min_train_rows(x_train=x, tscv=tscv, requested=min_rows)
    scores: list[float] = []
    oof_pred: list[NDArray[np.float64]] = []
    oof_y: list[NDArray[np.float64]] = []
    for train_idx, val_idx in tscv.split(x):
        if spec.task == "classification" and len(train_idx) < floor:
            continue
        pred = fit_predict(
            spec,
            params,
            x.iloc[train_idx],
            y.iloc[train_idx],
            x.iloc[val_idx],
            threads=threads,
        )
        truth = y.iloc[val_idx].to_numpy(dtype=float)
        if spec.task == "classification":
            scores.append(float(np.mean((pred - truth) ** 2)))
        else:
            scores.append(float(np.mean(np.abs(pred - truth))))
        oof_pred.append(pred)
        oof_y.append(truth)
    if not scores:
        return float("inf"), np.array([]), np.array([])
    return float(np.mean(scores)), np.concatenate(oof_pred), np.concatenate(oof_y)


def search_hyperparameters(
    spec: FamilySpec,
    feature_provider: FeatureProvider,
    columns: list[str],
    y: Series,
    train_mask: Series,
    *,
    n_workers: int | None = None,
) -> SearchOutcome:
    """Randomized search on the matched training rows only.

    Uses the production grid, iteration count and seeded sample of
    combinations, so every step samples the same combos. Combinations are
    scored in parallel worker processes with single-threaded estimators; the
    selection (first minimum in sample order) is unaffected by parallelism.

    Args:
        spec: Family under search.
        feature_provider: Returns the matched-frame feature matrix per window.
        columns: The step's feature columns.
        y: Target aligned with the matched frame.
        train_mask: Matched training rows.
        n_workers: Worker processes; ``None`` uses every CPU, ``1`` is serial.

    Raises:
        RuntimeError: If no combination produced a scored fold.
    """
    grid = _get_param_grid(spec.model_type, spec.task)
    n_iter = min(_n_iter_for(spec.model_type, spec.task), len(grid))
    rng = np.random.default_rng(SEARCH_SEED)
    sample = [int(i) for i in rng.choice(len(grid), size=n_iter, replace=False)]

    y_train = y[train_mask].reset_index(drop=True)
    x_cache: dict[int, DataFrame] = {}
    combos: list[tuple[dict[str, Any], int]] = []
    for idx in sample:
        combo = dict(grid[idx])
        window = int(combo.pop("epa_window", 4))
        if window not in x_cache:
            x_cache[window] = (
                feature_provider(window).loc[train_mask, columns].reset_index(drop=True)
            )
        combos.append((combo, window))

    workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(combos)))
    results = Parallel(n_jobs=workers)(
        delayed(_cv_folds)(
            spec, combo, x_cache[window], y_train, min_rows=None, threads=1 if workers > 1 else None
        )
        for combo, window in combos
    )

    best: tuple[float, dict[str, Any], int] | None = None
    for (combo, window), (score, _, _) in zip(combos, results, strict=True):
        if best is None or score < best[0]:
            best = (score, combo, window)
    if best is None or not np.isfinite(best[0]):
        msg = f"{spec.label}: hyperparameter search produced no scored fold"
        raise RuntimeError(msg)

    score, params, window = best
    recalibrate = False
    oof_ece: float | None = None
    if spec.model_type == GameModelType.XGBOOST and spec.task == "classification":
        _, oof_p, oof_y = _cv_folds(spec, params, x_cache[window], y_train, min_rows=None)
        oof_ece = float(expected_calibration_error(pd.Series(oof_p), pd.Series(oof_y)))
        recalibrate = oof_ece > _ECE_CALIBRATION_THRESHOLD
    return SearchOutcome(params, window, score, recalibrate, oof_ece)


def expanding_holdout_predictions(
    spec: FamilySpec,
    outcome: SearchOutcome,
    *,
    x_all: DataFrame,
    y: Series,
    season_start: Series,
    matched: Series,
    holdout_seasons: list[int],
    seed: int = DEFAULT_SEED,
) -> Series:
    """Refit with frozen hyperparameters expanding through each holdout season.

    For holdout season ``s`` the model trains on every matched row from
    seasons strictly before ``s`` (including earlier holdout seasons) and
    predicts ``s``. Returns predictions indexed like ``x_all``.

    Args:
        spec: Family.
        outcome: Frozen search outcome (never re-searched).
        x_all: Feature matrix for the outcome's window, matched-frame rows.
        y: Target aligned with ``x_all``.
        season_start: Season start year per row.
        matched: Matched-row mask.
        holdout_seasons: Season start years to predict, ascending.
        seed: Estimator seed.
    """
    out = pd.Series(np.nan, index=x_all.index, dtype=float)
    for start in holdout_seasons:
        train = matched & (season_start < start)
        pred_rows = matched & (season_start == start)
        out.loc[pred_rows] = fit_predict(
            spec,
            outcome.params,
            x_all.loc[train],
            y[train],
            x_all.loc[pred_rows],
            xgb_recalibrate=outcome.xgb_recalibrate,
            seed=seed,
        )
    return out
