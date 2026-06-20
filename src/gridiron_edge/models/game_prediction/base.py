# src/gridiron_edge/models/game_prediction/base.py

"""Game model metadata + trainer infrastructure.

Mirrors the prop_prediction/base.py architecture: shared base type from
artifact.py, game-specific metadata holds both classification metrics
(used by win_prob) and regression metrics (used by total).

Public API:
    GameModelMetadata   metadata recorded alongside a trained artifact
    GameModelType       supported algorithms enum
    GameModelSpec       describes a game model's identity + feature set
    GamesTrainer        ABC base class for game model trainers
    _create_model       module-level estimator factory
    _get_param_grid     module-level hyperparameter grid factory
    _n_iter_for         iteration count for randomized HP search

The trainer infrastructure (GameModelType, GameModelSpec, GamesTrainer)
operates as the single training surface for game-side models — both
classification (win_prob) and regression (total) tasks dispatch from
here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import product
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.model_selection import TimeSeriesSplit

# pyrefly: ignore [missing-import, untyped-import]
from tqdm import tqdm

from gridiron_edge.models.artifact import BaseModelMetadata
from gridiron_edge.models.game_prediction._epa_window import WindowData

if TYPE_CHECKING:
    # pyrefly: ignore [missing-import]
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import TimeSeriesSplit

    from gridiron_edge.models.game_prediction._columns import FeatureSet

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GameModelMetadata(BaseModelMetadata):
    """Metadata recorded alongside a trained game model artifact.

    Inherits the shared metadata fields from :class:`BaseModelMetadata`.

    Classification metrics are populated when ``task="classification"``
    (used by ``win_prob``); regression metrics are populated when
    ``task="regression"`` (used by ``total``). Unused metrics remain NaN.
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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GameModelType(StrEnum):
    """Supported game model algorithm types."""

    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


@dataclass(frozen=True)
class GameModelSpec:
    """Metadata describing a game model's identity.

    Attributes:
        name: Artifact ``model_name`` (e.g. ``"win_prob"``, ``"total"``).
        task: ``"classification"`` or ``"regression"``.
        target_col: Column in the modeling DataFrame that is the prediction
            target.
        feature_set: Per-algorithm feature set. Keys define which algorithms
            this spec supports (``TotalTrainer`` excludes ``LOGISTIC``).
        description: Human-readable description.
    """

    name: str
    task: str
    target_col: str
    feature_set: dict[GameModelType, FeatureSet]
    description: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Threshold above which XGBoost classification gets post-training isotonic
#: calibration. Mirrors the existing ``_train_xgboost`` behavior.
_ECE_CALIBRATION_THRESHOLD: float = 0.025

#: Hyperparameter search iteration counts (match existing trainers).
_N_ITER_RF_CLF: int = 50
_N_ITER_XGB_CLF: int = 75
_N_ITER_RF_REG: int = 50
_N_ITER_XGB_REG: int = 50

_CV_FOLDS: int = 5


# ---------------------------------------------------------------------------
# Model factory + hyperparameter grids
# ---------------------------------------------------------------------------


def _create_model(model_type: GameModelType, task: str) -> tuple[Any, Any]:
    """Create a model instance and optional scaler.

    For classification, ``RANDOM_FOREST`` is unconditionally wrapped in
    ``CalibratedClassifierCV(method="isotonic", cv=3)`` to correct the
    systematic overconfidence observed in tree ensembles on this dataset.
    ``XGBOOST`` is not pre-calibrated; isotonic calibration is applied
    post-training in ``GamesTrainer.train()`` only when holdout ECE
    exceeds ``_ECE_CALIBRATION_THRESHOLD``.

    Returns:
        Tuple of (model_instance, scaler_or_none). Logistic gets a
        StandardScaler; tree models get None.
    """
    if task == "classification":
        if model_type == GameModelType.LOGISTIC:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler

            # TimeSeriesSplit for inner CV preserves temporal ordering when
            # selecting C and l1_ratio. The default StratifiedKFold would
            # produce random folds over chronologically sorted data, leaking
            # future information into HP selection (game_base/H1).
            return (
                LogisticRegressionCV(
                    Cs=10,
                    cv=TimeSeriesSplit(n_splits=_CV_FOLDS),
                    solver="saga",
                    l1_ratios=(0.0, 0.5, 1.0),
                    # pyrefly: ignore [unexpected-keyword]
                    use_legacy_attributes=False,
                    scoring="neg_brier_score",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=42,
                ),
                StandardScaler(),
            )

        if model_type == GameModelType.RANDOM_FOREST:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import TimeSeriesSplit

            rf = RandomForestClassifier(n_jobs=-1, random_state=42)
            # TimeSeriesSplit for isotonic calibration: fold prediction order
            # matters because calibrating with mixed-time folds leaks the most
            # recent calibration data into older seasons' calibration curve
            # (game_base/H2).
            return (
                CalibratedClassifierCV(rf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)),
                None,
            )

        if model_type == GameModelType.XGBOOST:
            # pyrefly: ignore [missing-import]
            from xgboost import XGBClassifier

            return (
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
                None,
            )

    elif task == "regression":
        if model_type == GameModelType.LOGISTIC:
            msg: str = (
                "GameModelType.LOGISTIC is not a regression estimator. "
                "Use RANDOM_FOREST or XGBOOST for task='regression'."
            )
            raise ValueError(msg)

        if model_type == GameModelType.RANDOM_FOREST:
            # pyrefly: ignore [missing-import]
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_jobs=-1, random_state=42), None

        if model_type == GameModelType.XGBOOST:
            # pyrefly: ignore [missing-import]
            from xgboost import XGBRegressor

            return XGBRegressor(n_jobs=-1, random_state=42, verbosity=0), None

    msg = f"Unknown (model_type, task) combination: ({model_type}, {task!r})"
    raise ValueError(msg)


def _get_param_grid(model_type: GameModelType, task: str) -> list[dict[str, Any]]:
    """Return the randomized hyperparameter search grid for the given combo.

    The grid is the full Cartesian product; ``GamesTrainer.train()`` samples
    ``n_iter`` random combinations from it. Classification grids include
    ``epa_window`` as a tunable dimension; regression grids do not
    (regression always uses the standard 4-game window).
    """
    from gridiron_edge.models.game_prediction._epa_window import _EPA_WINDOW_OPTIONS

    if task == "classification":
        if model_type == GameModelType.LOGISTIC:
            # LogisticRegressionCV does its own internal CV over Cs; the only
            # outer dim is epa_window.
            return [{"epa_window": w} for w in _EPA_WINDOW_OPTIONS]

        if model_type == GameModelType.RANDOM_FOREST:
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_leaf": leaf,
                    "max_features": mf,
                    "epa_window": w,
                }
                for n, d, leaf, mf, w in product(
                    [100, 200, 300, 500],
                    [3, 4, 5, 6, None],
                    [5, 10, 20, 30],
                    ["sqrt", "log2", 0.5],
                    _EPA_WINDOW_OPTIONS,
                )
            ]

        if model_type == GameModelType.XGBOOST:
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "learning_rate": lr,
                    "subsample": s,
                    "colsample_bytree": c,
                    "min_child_weight": mcw,
                    "gamma": g,
                    "epa_window": w,
                }
                for n, d, lr, s, c, mcw, g, w in product(
                    [100, 150, 200, 300, 500],
                    [2, 3, 4, 5, 6],
                    [0.01, 0.03, 0.05, 0.1, 0.2],
                    [0.6, 0.7, 0.8, 1.0],
                    [0.6, 0.7, 0.8, 1.0],
                    [1, 5, 10, 20],
                    [0.0, 0.1, 0.3, 0.5],
                    _EPA_WINDOW_OPTIONS,
                )
            ]

    elif task == "regression":
        if model_type == GameModelType.LOGISTIC:
            msg: str = (
                "GameModelType.LOGISTIC is not a regression estimator. "
                "Use RANDOM_FOREST or XGBOOST for task='regression'."
            )
            raise ValueError(msg)

        if model_type == GameModelType.RANDOM_FOREST:
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_leaf": leaf,
                    "max_features": mf,
                }
                for n, d, leaf, mf in product(
                    [200, 300, 400, 500],
                    [8, 12, 16, 20, None],
                    [4, 8, 12, 16],
                    ["sqrt", "log2", 0.3, 0.5],
                )
            ]

        if model_type == GameModelType.XGBOOST:
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "learning_rate": lr,
                    "subsample": s,
                }
                for n, d, lr, s in product(
                    [200, 300, 400, 500],
                    [4, 6, 8, 10],
                    [0.01, 0.03, 0.05, 0.1],
                    [0.7, 0.8, 1.0],
                )
            ]

    msg = f"Unknown (model_type, task) combination: ({model_type}, {task!r})"
    raise ValueError(msg)


def _n_iter_for(model_type: GameModelType, task: str) -> int:
    """Iteration count for the randomized search."""
    from gridiron_edge.models.game_prediction._epa_window import _EPA_WINDOW_OPTIONS

    if task == "classification":
        if model_type == GameModelType.LOGISTIC:
            # Whole grid is just epa_window — sample all of it.
            return len(_EPA_WINDOW_OPTIONS)
        if model_type == GameModelType.RANDOM_FOREST:
            return _N_ITER_RF_CLF
        if model_type == GameModelType.XGBOOST:
            return _N_ITER_XGB_CLF
    elif task == "regression":
        if model_type == GameModelType.RANDOM_FOREST:
            return _N_ITER_RF_REG
        if model_type == GameModelType.XGBOOST:
            return _N_ITER_XGB_REG

    msg = f"Unknown (model_type, task) combination: ({model_type}, {task!r})"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# GamesTrainer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal search result type
# ---------------------------------------------------------------------------


@dataclass
class _SearchResult:
    """Outcome of :meth:`GamesTrainer._run_hp_search`.

    Carries the refit-best model and scaler along with the holdout split
    used for evaluation so that ``train()`` can build metadata without
    re-doing data prep.
    """

    model: Any
    scaler: Any
    params: dict[str, Any]
    score: float
    x_train: pd.DataFrame
    y_train: Series
    x_hold: pd.DataFrame
    y_hold: Series
    train_seasons: list[str]
    hold_seasons: list[str]


def _apply_params(model: BaseEstimator, params: dict[str, Any]) -> None:
    """Apply hyperparameters to a model, handling CalibratedClassifierCV wrapping.

    When the model is wrapped in :class:`CalibratedClassifierCV`, the
    underlying estimator's params are addressed via ``estimator__<name>``
    rather than directly on the wrapper. This helper detects the wrap and
    forwards params correctly so callers don't need to know the model
    structure.
    """
    # pyrefly: ignore [missing-import]
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(model, CalibratedClassifierCV):
        prefixed: dict[str, Any] = {f"estimator__{k}": v for k, v in params.items()}
        model.set_params(**prefixed)
    else:
        model.set_params(**params)


def _filter_for_walk_forward(
    x_train_orig: pd.DataFrame,
    y_train_orig: Series,
    x_hold_orig: pd.DataFrame,
    y_hold_orig: Series,
    train_through_season: str,
    *,
    df_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, Series, pd.DataFrame, Series, list[str], list[str]]:
    """Re-split train+holdout for walk-forward backfill.

    The original (x_train_orig, x_hold_orig) split used HOLDOUT_SEASONS.
    For walk-forward, we re-split: training uses seasons strictly before
    ``train_through_season`` (from either original train or holdout pool),
    and the new holdout is the single season immediately after
    ``train_through_season``.

    Args:
        x_train_orig: Original training feature matrix from the standard
            HOLDOUT_SEASONS split.
        y_train_orig: Original training target vector aligned with
            ``x_train_orig``.
        x_hold_orig: Original holdout feature matrix from the standard
            HOLDOUT_SEASONS split.
        y_hold_orig: Original holdout target vector aligned with
            ``x_hold_orig``.
        train_through_season: Season label like ``"2014-2015"``. Training
            uses everything before this; new holdout is the next season.
        df_reference: Original DataFrame with YEAR column used for lookups.

    Returns:
        Tuple of (x_train, y_train, x_hold, y_hold, train_seasons,
        hold_seasons).
    """
    # Combine into full feature matrix and target.
    x_full = pd.concat([x_train_orig, x_hold_orig])
    y_full = pd.concat([y_train_orig, y_hold_orig])

    # Look up YEAR for each row
    year_series = df_reference.loc[x_full.index, "YEAR"]
    train_through_start = int(train_through_season.split("-")[0])
    next_season_start = train_through_start + 1
    next_season_label = f"{next_season_start}-{next_season_start + 1}"

    # Use first 4 chars of YEAR ("2014-2015" -> "2014") for comparison
    year_start = year_series.astype(str).str[:4].astype(int)
    train_mask = year_start <= train_through_start
    hold_mask = year_series == next_season_label

    x_train = x_full.loc[train_mask]
    y_train = y_full.loc[train_mask]
    x_hold = x_full.loc[hold_mask]
    y_hold = y_full.loc[hold_mask]

    train_seasons = sorted(year_series.loc[train_mask].unique().tolist())
    hold_seasons = sorted(year_series.loc[hold_mask].unique().tolist())

    return x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons


class GamesTrainer(ABC):
    """Base class for game model trainers.

    Subclasses implement only ``spec`` — all training logic lives here.
    The factory pattern (``_create_model`` / ``_get_param_grid``) handles
    Logistic, RandomForest, and XGBoost transparently across both
    classification (win_prob) and regression (total) tasks.
    """

    _model: Any = None
    _scaler: Any = None

    @property
    @abstractmethod
    def spec(self) -> GameModelSpec:
        """Return the model specification."""
        ...

    def train(
        self,
        df: pd.DataFrame,
        *,
        model_type: GameModelType,
        repo: Path | None = None,
        train_through_season: str | None = None,
        persist: bool = True,
    ) -> GameModelMetadata:
        """Full training pipeline: prepare data, HP search, fit, evaluate, save.

        Args:
            df: Full modeling DataFrame from ``load_modeling_file``.
            model_type: Algorithm to use. Must be a key of
                ``self.spec.feature_set``.
            repo: Repository root override.
            train_through_season: If set (e.g. ``"2014-2015"``), training
                data is filtered to seasons strictly before this label.
                Holdout becomes the single season immediately after
                ``train_through_season``. Used for walk-forward backfill.
                When ``None``, the default HOLDOUT_SEASONS split applies.
            persist: If ``True`` (default), save the trained artifact to
                ``ArtifactStore``. Pass ``False`` for walk-forward
                intermediates that should be discarded after producing
                their season's predictions.

        Returns:
            ``GameModelMetadata`` with task-appropriate holdout metrics
            populated as first-class fields.

        Raises:
            ValueError: If ``model_type`` is not supported by this spec.
            RuntimeError: If the HP search produced no valid pipeline.
        """
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
        from gridiron_edge.models.artifact import ArtifactStore

        resolved_repo: Path = repo or get_settings().repo_root
        spec: GameModelSpec = self.spec

        if model_type not in spec.feature_set:
            supported: list[str] = [mt.value for mt in spec.feature_set]
            msg: str = (
                f"{type(self).__name__}: model_type={model_type.value!r} is not "
                f"supported by spec {spec.name!r}. Supported: {supported}"
            )
            raise ValueError(msg)

        feature_set = spec.feature_set[model_type]
        feature_fn: Callable = feature_set.feature_fn
        feature_names: list[str] = list(feature_set.feature_names)

        search: _SearchResult = self._run_hp_search(
            df=df,
            model_type=model_type,
            feature_fn=feature_fn,
            repo=resolved_repo,
        )

        self._model = search.model
        self._scaler = search.scaler

        if spec.task == "classification":
            metadata = self._build_classification_metadata(
                model_type=model_type,
                best_params=search.params,
                best_score=search.score,
                feature_names=feature_names,
                x_train=search.x_train,
                y_train=search.y_train,
                x_hold=search.x_hold,
                y_hold=search.y_hold,
                train_seasons=search.train_seasons,
                hold_seasons=search.hold_seasons,
                schema_version=CURRENT_SCHEMA_VERSION,
            )
        else:
            metadata = self._build_regression_metadata(
                model_type=model_type,
                best_params=search.params,
                best_score=search.score,
                feature_names=feature_names,
                x_train=search.x_train,
                y_train=search.y_train,
                x_hold=search.x_hold,
                y_hold=search.y_hold,
                train_seasons=search.train_seasons,
                hold_seasons=search.hold_seasons,
                schema_version=CURRENT_SCHEMA_VERSION,
            )

        ArtifactStore(resolved_repo).save(
            metadata=metadata,
            model_obj=self._model,
            scaler=self._scaler,
            overwrite=True,
        )

        if persist:
            ArtifactStore(resolved_repo).save(
                metadata=metadata,
                model_obj=self._model,
                scaler=self._scaler,
                overwrite=True,
            )
        return metadata

    def _run_hp_search(
        self,
        *,
        df: pd.DataFrame,
        model_type: GameModelType,
        feature_fn: Callable,
        repo: Path,
        train_through_season: str | None = None,
    ) -> _SearchResult:
        """Run randomized hyperparameter search; refit best on full training set.

        Returns a populated :class:`_SearchResult`. Raises ``RuntimeError`` if
        no valid pipeline was produced (e.g. all CV folds were skipped due to
        the MIN_CV_TRAIN_ROWS guard).
        """
        # pyrefly: ignore [missing-import]
        from sklearn.model_selection import TimeSeriesSplit

        spec: GameModelSpec = self.spec
        grid: list[dict[str, Any]] = _get_param_grid(model_type, spec.task)
        n_iter: int = min(_n_iter_for(model_type, spec.task), len(grid))
        rng = np.random.default_rng(42)

        window_cache: dict[int, WindowData] = {}
        tscv = TimeSeriesSplit(n_splits=_CV_FOLDS)

        best_score: float = float("inf")
        best_params: dict[str, Any] = {}
        best_model: Any = None
        best_scaler: Any = None
        best_x_train: pd.DataFrame | None = None
        best_y_train: Series | None = None
        best_x_hold: pd.DataFrame | None = None
        best_y_hold: Series | None = None
        best_train_seasons: list[str] = []
        best_hold_seasons: list[str] = []

        sample_indices: list[int] = list(rng.choice(len(grid), size=n_iter, replace=False).tolist())

        bar = tqdm(
            sample_indices,
            desc=f"  {spec.name}/{model_type.value}",
            unit="iter",
            ncols=88,
            colour="cyan",
        )
        for idx in bar:
            sampled: dict[str, Any] = dict(grid[idx])
            window: int = sampled.pop("epa_window", 4)

            x_train, y_train, x_hold, y_hold, train_szns, hold_szns = self._prepare_window(
                df=df,
                window=window,
                window_cache=window_cache,
                feature_fn=feature_fn,
                repo=repo,
                train_through_season=train_through_season,
            )

            score: float = self._cv_score(
                x_train=x_train,
                y_train=y_train,
                params=sampled,
                model_type=model_type,
                tscv=tscv,
            )

            if score < best_score:
                best_score = score
                best_params = {**sampled, "epa_window": window}
                model, scaler = _create_model(model_type, spec.task)
                _apply_params(model, sampled)
                x_tr_arr = scaler.fit_transform(x_train) if scaler is not None else x_train.values
                model.fit(x_tr_arr, y_train)
                best_model = model
                best_scaler = scaler
                best_x_train, best_y_train = x_train, y_train
                best_x_hold, best_y_hold = x_hold, y_hold
                best_train_seasons, best_hold_seasons = train_szns, hold_szns

            bar.set_postfix(best=f"{best_score:.5f}", window=window, refresh=False)

        bar.close()

        if (
            best_model is None
            or best_x_train is None
            or best_y_train is None
            or best_x_hold is None
            or best_y_hold is None
        ):
            msg = (
                f"{spec.name}/{model_type.value}: hyperparameter search produced no valid pipeline"
            )
            raise RuntimeError(msg)

        return _SearchResult(
            model=best_model,
            scaler=best_scaler,
            params=best_params,
            score=best_score,
            x_train=best_x_train,
            y_train=best_y_train,
            x_hold=best_x_hold,
            y_hold=best_y_hold,
            train_seasons=best_train_seasons,
            hold_seasons=best_hold_seasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_window(
        self,
        *,
        df: pd.DataFrame,
        window: int,
        window_cache: dict[int, Any],
        feature_fn: Callable,
        repo: Path,
        train_through_season: str | None = None,
    ) -> tuple[pd.DataFrame, Series, pd.DataFrame, Series, list[str], list[str]]:
        """Resolve train/holdout split for a given EPA window.

        Classification path uses the EPA window cache. Regression path
        ignores ``window`` (regression always uses window=4 on disk) and
        delegates to :func:`total._prepare_total_data`.
        """
        spec: GameModelSpec = self.spec

        if spec.task == "classification":
            from gridiron_edge.models.game_prediction._epa_window import (
                _get_cached_window_data,
            )

            wd: WindowData = _get_cached_window_data(window_cache, window, df, feature_fn, repo)
            x_train, y_train = wd.x_train, wd.y_train
            x_hold, y_hold = wd.x_holdout, wd.y_holdout
            train_seasons, hold_seasons = wd.train_seasons, wd.holdout_seasons

        else:
            from gridiron_edge.models.game_prediction.total import _prepare_total_data

            x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = _prepare_total_data(
                repo
            )

        if train_through_season is not None:
            x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = (
                _filter_for_walk_forward(
                    x_train,
                    y_train,
                    x_hold,
                    y_hold,
                    train_through_season=train_through_season,
                    df_reference=df,
                )
            )

        return x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons

    def _cv_score(
        self,
        *,
        x_train: pd.DataFrame,
        y_train: Series,
        params: dict[str, Any],
        model_type: GameModelType,
        tscv: TimeSeriesSplit,
    ) -> float:
        """Run cross-validated score for one parameter combo.

        Classification scores by Brier (lower is better); regression scores
        by MAE (lower is better). MIN_CV_TRAIN_ROWS guard from _features
        applies only to classification.
        """
        from gridiron_edge.evaluation.metrics import brier_score
        from gridiron_edge.models.game_prediction._features import MIN_CV_TRAIN_ROWS

        spec: GameModelSpec = self.spec
        fold_scores: list[float] = []

        for train_idx, val_idx in tscv.split(x_train):
            if spec.task == "classification" and len(train_idx) < MIN_CV_TRAIN_ROWS:
                continue
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            model, scaler = _create_model(model_type, spec.task)
            _apply_params(model, params)
            if scaler is not None:
                x_tr_arr = scaler.fit_transform(x_tr)
                x_val_arr = scaler.transform(x_val)
            else:
                x_tr_arr = x_tr.values
                x_val_arr = x_val.values

            model.fit(x_tr_arr, y_tr)

            if spec.task == "classification":
                probs = pd.Series(model.predict_proba(x_val_arr)[:, 1])
                fold_scores.append(brier_score(probs, y_val.astype(float).reset_index(drop=True)))
            else:
                preds = model.predict(x_val_arr)
                fold_scores.append(float(np.mean(np.abs(preds - np.asarray(y_val)))))

        if not fold_scores:
            return float("inf")
        return float(np.mean(fold_scores))

    def _build_classification_metadata(
        self,
        *,
        model_type: GameModelType,
        best_params: dict[str, Any],
        best_score: float,
        feature_names: list[str],
        x_train: pd.DataFrame,
        y_train: Series,
        x_hold: pd.DataFrame,
        y_hold: Series,
        train_seasons: list[str],
        hold_seasons: list[str],
        schema_version: int,
    ) -> GameModelMetadata:
        """Evaluate holdout for classification + apply XGB post-calibration."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from gridiron_edge.evaluation.metrics import (
            accuracy,
            brier_score,
            expected_calibration_error,
            log_loss,
            roc_auc,
        )

        spec: GameModelSpec = self.spec

        x_hold_arr = self._scaler.transform(x_hold) if self._scaler is not None else x_hold.values
        x_train_arr = (
            self._scaler.transform(x_train) if self._scaler is not None else x_train.values
        )

        hold_probs = pd.Series(self._model.predict_proba(x_hold_arr)[:, 1], index=x_hold.index)
        train_probs = pd.Series(self._model.predict_proba(x_train_arr)[:, 1], index=x_train.index)

        holdout_brier = brier_score(hold_probs, y_hold.astype(float))
        holdout_ece = expected_calibration_error(hold_probs, y_hold.astype(float))
        holdout_auc = roc_auc(hold_probs, y_hold.astype(float))
        holdout_log_loss = log_loss(hold_probs, y_hold.astype(float))
        holdout_accuracy = accuracy(hold_probs, y_hold.astype(float))
        train_brier = brier_score(train_probs, y_train.astype(float))

        calibration_applied: bool = False
        if model_type == GameModelType.XGBOOST and holdout_ece > _ECE_CALIBRATION_THRESHOLD:
            logger.info(
                "%s/%s: ECE=%.4f > %.3f — applying isotonic calibration",
                spec.name,
                model_type.value,
                holdout_ece,
                _ECE_CALIBRATION_THRESHOLD,
            )
            params_no_window: dict[str, Any] = {
                k: v for k, v in best_params.items() if k != "epa_window"
            }
            xgb_recal, _ = _create_model(model_type, spec.task)
            _apply_params(xgb_recal, params_no_window)

            # TimeSeriesSplit for post-training calibration: same rationale as
            # the RANDOM_FOREST branch above (game_base/H2). The training data
            # is already chronologically sorted at this point.
            calibration_cv = TimeSeriesSplit(n_splits=3)

            if self._scaler is not None:
                cal_pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            CalibratedClassifierCV(xgb_recal, method="isotonic", cv=calibration_cv),
                        ),
                    ]
                )
                cal_pipeline.fit(x_train, y_train)
                self._model = cal_pipeline
                self._scaler = None
            else:
                cal = CalibratedClassifierCV(xgb_recal, method="isotonic", cv=calibration_cv)
                cal.fit(x_train_arr, y_train)
                self._model = cal

            hold_probs = pd.Series(self._model.predict_proba(x_hold)[:, 1], index=x_hold.index)
            holdout_brier = brier_score(hold_probs, y_hold.astype(float))
            holdout_ece = expected_calibration_error(hold_probs, y_hold.astype(float))
            calibration_applied = True

        return GameModelMetadata(
            model_name=spec.name,
            model_type=model_type.value,
            task=spec.task,
            trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
            schema_version=schema_version,
            training_seasons=train_seasons,
            holdout_seasons=hold_seasons,
            parameters={
                **best_params,
                "cv_brier": round(best_score, 6),
                "train_brier": round(train_brier, 6),
                "overfit_gap": round(holdout_brier - train_brier, 6),
                "calibration_applied": calibration_applied,
            },
            feature_columns=feature_names,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_hold),
            holdout_brier=round(holdout_brier, 6),
            holdout_ece=round(holdout_ece, 6),
            holdout_auc=round(holdout_auc, 6),
            holdout_log_loss=round(holdout_log_loss, 6),
            holdout_accuracy=round(holdout_accuracy, 6),
        )

    def _build_regression_metadata(
        self,
        *,
        model_type: GameModelType,
        best_params: dict[str, Any],
        best_score: float,
        feature_names: list[str],
        x_train: pd.DataFrame,
        y_train: Series,
        x_hold: pd.DataFrame,
        y_hold: Series,
        train_seasons: list[str],
        hold_seasons: list[str],
        schema_version: int,
    ) -> GameModelMetadata:
        """Evaluate holdout for regression."""
        spec: GameModelSpec = self.spec

        x_hold_arr = self._scaler.transform(x_hold) if self._scaler is not None else x_hold.values
        x_train_arr = (
            self._scaler.transform(x_train) if self._scaler is not None else x_train.values
        )

        hold_preds = self._model.predict(x_hold_arr)
        train_preds = self._model.predict(x_train_arr)

        y_hold_arr = np.asarray(y_hold, dtype=float)
        y_train_arr = np.asarray(y_train, dtype=float)

        hold_mae = float(np.mean(np.abs(hold_preds - y_hold_arr)))
        hold_rmse = float(np.sqrt(np.mean((hold_preds - y_hold_arr) ** 2)))
        train_mae = float(np.mean(np.abs(train_preds - y_train_arr)))
        train_rmse = float(np.sqrt(np.mean((train_preds - y_train_arr) ** 2)))

        ss_res = float(np.sum((hold_preds - y_hold_arr) ** 2))
        ss_tot = float(np.sum((y_hold_arr - y_hold_arr.mean()) ** 2))
        hold_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return GameModelMetadata(
            model_name=spec.name,
            model_type=model_type.value,
            task=spec.task,
            trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
            schema_version=schema_version,
            training_seasons=train_seasons,
            holdout_seasons=hold_seasons,
            parameters={
                **best_params,
                "cv_mae": round(best_score, 6),
                "train_mae": round(train_mae, 6),
                "train_rmse": round(train_rmse, 6),
                "mean_target_train": float(np.mean(y_train_arr)),
                "mean_target_holdout": float(np.mean(y_hold_arr)),
            },
            feature_columns=feature_names,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_hold),
            holdout_mae=round(hold_mae, 6),
            holdout_rmse=round(hold_rmse, 6),
            holdout_r2=round(hold_r2, 6),
        )
