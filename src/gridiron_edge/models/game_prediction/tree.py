# src/gridiron_edge/models/game_prediction/tree.py

"""Tree-based game prediction models.

Two model families using the expanded feature set (107 features):

    random_forest: Random Forest with isotonic calibration
        - TimeSeriesSplit CV for hyperparameter search
        - EPA window optimization across [4, 6, 8, 10, 12]
        - CalibratedClassifierCV(isotonic) applied unconditionally

    xgboost: XGBoost gradient boosting
        - TimeSeriesSplit CV for hyperparameter search
        - EPA window optimization
        - Isotonic calibration applied conditionally (holdout ECE > 0.025)

Both models tune the EPA rolling window as a hyperparameter.

Training progress is reported via tqdm: one bar per model showing
iteration count, current best CV Brier, and ETA.

Artifact storage (Workstream 2): each trainer writes to
``data/models/{model_name}/{model_type}/``. ``model_name`` is ``"win_prob"``
for both variants; ``model_type`` is ``"random_forest"`` or ``"xgboost"``.
The factory pattern (``_make_tree_variant``) is killed in D2b — replaced
by ``GamesTrainer`` + spec subclasses. Until then ``PredictorRegistry`` keys
remain flat (``"random_forest"``, ``"xgboost"``) to limit D1b scope.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from logging import Logger
from pathlib import Path
from typing import Literal

import numpy as np
from numpy import ndarray
from numpy.random import Generator
import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import, untyped-import]
from tqdm import tqdm

from gridiron_edge.models.base import PredictorSpec

# EPA metric names — single source of truth lives in the feature module.
from gridiron_edge.models.game_prediction._epa_window import (
    _EPA_WINDOW_OPTIONS,
    WindowData,
    _get_cached_window_data,
)
from gridiron_edge.models.game_prediction._features import (
    FEATURE_SETS,
    MIN_CV_TRAIN_ROWS,
    FeatureSet,
    _is_trained,
)
from gridiron_edge.models.game_prediction.base import GameModelMetadata
from gridiron_edge.models.registry import PredictorRegistry

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared prediction helpers for tree models
# ---------------------------------------------------------------------------


def _predict_historical_tree(
    games: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared historical prediction logic for all tree-based variants.

    Args:
        games: Games DataFrame (unused — full modeling file loaded from disk).
        model_name: Model purpose (``"win_prob"``).
        model_type: Model algorithm (``"random_forest"`` or ``"xgboost"``).
        feature_fn: Feature engineering function.
        repo: Repository root.

    Returns:
        Enriched prediction DataFrame in the standard archive format.
    """
    from gridiron_edge.models.game_prediction.pipeline import predict_games

    return predict_games(
        model_name=model_name,
        model_type=model_type,
        feature_fn=feature_fn,
        repo=repo,
        is_backfilled=True,
    )


def _predict_upcoming_tree(
    schedule: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared upcoming prediction logic for all tree-based variants.

    Args:
        schedule: Upcoming games schedule DataFrame.
        model_name: Model purpose (``"win_prob"``).
        model_type: Model algorithm.
        feature_fn: Feature engineering function.
        repo: Repository root.

    Returns:
        Enriched prediction DataFrame with win probabilities, spread,
        bands, tier, and projected scores.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.accessor import DatasetAccessor
    from gridiron_edge.features.pipeline import FEATURES
    from gridiron_edge.features.registry import run_features
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.game_prediction.post_process import enrich_predictions

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    if not store.is_trained(model_name, model_type):
        logger.warning("(%s, %s): no artifact found.", model_name, model_type)
        return pd.DataFrame()

    pipeline = store.load(model_name, model_type)
    datasets = DatasetAccessor(repo=resolved_repo)

    upcoming_df: DataFrame = run_features(df=schedule, feature_names=FEATURES, datasets=datasets)
    features = feature_fn(upcoming_df)
    valid = features.notna().all(axis=1)
    upcoming_valid = upcoming_df.loc[valid].copy()
    x_feat = features.loc[valid]

    if x_feat.empty:
        return pd.DataFrame()

    probs = pipeline.predict_proba(x_feat)[:, 1]
    result = upcoming_valid[["GAME_ID", "AWAY_TEAM", "HOME_TEAM", "WEEK_NUM"]].copy()
    result["AWAY_WIN_PROB"] = probs
    result["HOME_WIN_PROB"] = 1.0 - probs
    result["AWAY_TEAM_WIN_PROB"] = (pd.Series(probs) * 100).map(lambda x: f"{x:.1f} %").values
    result["HOME_TEAM_WIN_PROB"] = (
        ((1.0 - pd.Series(probs)) * 100).map(lambda x: f"{x:.1f} %").values
    )
    result["AWAY_TEAM_ELO"] = upcoming_valid.get("TEAM_A_ELO", float("nan"))
    result["HOME_TEAM_ELO"] = upcoming_valid.get("TEAM_B_ELO", float("nan"))

    # Total model predictions (optional).
    try:
        from gridiron_edge.models.game_prediction.total import predict_total

        totals: Series = predict_total(upcoming_valid, repo=resolved_repo)
        result["model_total"] = totals.loc[upcoming_valid.loc[valid].index].values
    except (FileNotFoundError, Exception):
        logger.debug("_predict_upcoming_tree: total model not available")

    result: DataFrame = enrich_predictions(
        result,
        model_name=model_name,
        model_type=model_type,
        recalibrate=True,
        repo=resolved_repo,
    )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------


def _train_random_forest(
    df: pd.DataFrame,
    *,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> GameModelMetadata:
    """Train a Random Forest classifier with randomised hyperparameter search.

    Searches over n_estimators, max_depth, min_samples_leaf, max_features,
    and epa_window. CalibratedClassifierCV(isotonic) is applied
    unconditionally to correct systematic RF overconfidence.

    Optimisations vs the initial implementation:
    - Window cache: each unique EPA window is rebuilt and split at most once
      across all iterations, eliminating repeated parquet reads.
    - TimeSeriesSplit is instantiated once before the loop.
    - Feature importances averaged with np.array().mean(axis=0) (vectorised).
    - ``feature_set`` metadata derived from ``feature_names`` length rather
      than hardcoded to ``"combined_32"``.

    Progress is reported via a tqdm bar showing iteration count, current
    best CV Brier, and ETA.

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        feature_fn: Feature engineering function.
        feature_names: Feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        GameModelMetadata with holdout Brier, best parameters, and feature
        importances (top 10, averaged across calibration folds). Artifact
        is written to ``data/models/win_prob/random_forest/``.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.calibration import CalibratedClassifierCV

    # pyrefly: ignore [missing-import]
    from sklearn.ensemble import RandomForestClassifier

    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import TimeSeriesSplit

    # pyrefly: ignore [missing-import]
    from sklearn.pipeline import Pipeline

    # pyrefly: ignore [missing-import]
    from sklearn.preprocessing import StandardScaler

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        log_loss,
        roc_auc,
    )
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    param_grid: dict[str, list] = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_leaf": [5, 10, 20, 30],
        "max_features": ["sqrt", "log2", 0.5],
        "epa_window": _EPA_WINDOW_OPTIONS,
    }

    n_iter: int = 50
    cv_folds: int = 5
    rng: Generator = np.random.default_rng(42)

    # Window cache: keyed by window size → (df_w, x_train, y_train, x_hold, y_hold, ...)
    # Bounded by len(_EPA_WINDOW_OPTIONS); eliminates repeated parquet reads.
    window_cache: dict[int, WindowData] = {}

    # Pre-populate window=4 (fast path; also initialises *_best for static analysis)
    _wd0: WindowData = _get_cached_window_data(window_cache, 4, df, feature_fn, resolved_repo)
    x_train_best, y_train_best = _wd0.x_train, _wd0.y_train
    x_hold_best, y_hold_best = _wd0.x_holdout, _wd0.y_holdout
    train_seasons, hold_seasons = _wd0.train_seasons, _wd0.holdout_seasons

    best_cv_brier: float = float("inf")
    best_params: dict = {}
    best_pipeline = None

    param_keys: list[str] = list(param_grid.keys())

    # Instantiate once — same random_state means identical fold assignments
    # every iteration, which is correct (we want comparable CV scores).
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    bar = tqdm(
        range(n_iter),
        desc="  win_prob/random_forest",
        unit="iter",
        ncols=88,
        colour="cyan",
    )
    for iteration in bar:
        sampled: dict = {
            k: param_grid[k][int(rng.integers(len(param_grid[k])))] for k in param_keys
        }
        window: int = sampled.pop("epa_window")

        # Cache hit if this window was seen before — no disk read
        _wd: WindowData = _get_cached_window_data(
            window_cache, window, df, feature_fn, resolved_repo
        )
        x_train, y_train = _wd.x_train, _wd.y_train
        train_seasons, hold_seasons = _wd.train_seasons, _wd.holdout_seasons

        fold_briers: list[float] = []
        for train_idx, val_idx in tscv.split(x_train):
            if len(train_idx) < MIN_CV_TRAIN_ROWS:
                continue
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            rf = RandomForestClassifier(random_state=42, n_jobs=-1, **sampled)
            cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", cal)])
            pipe.fit(x_tr, y_tr)
            val_probs: Series = pd.Series(pipe.predict_proba(x_val)[:, 1])
            fold_briers.append(brier_score(val_probs, y_val.astype(float).reset_index(drop=True)))

        cv_brier: float = float(np.mean(fold_briers))
        logger.debug(
            "  iter %d/%d: window=%d params=%s cv_brier=%.5f",
            iteration + 1,
            n_iter,
            window,
            sampled,
            cv_brier,
        )

        if cv_brier < best_cv_brier:
            best_cv_brier = cv_brier
            best_params = {**sampled, "epa_window": window}
            # x_train_best / x_hold_best come from the cache — no extra rebuild
            _wd = window_cache[window]
            x_train_best, y_train_best = _wd.x_train, _wd.y_train
            x_hold_best, y_hold_best = _wd.x_holdout, _wd.y_holdout
            train_seasons, hold_seasons = _wd.train_seasons, _wd.holdout_seasons
            rf_best = RandomForestClassifier(random_state=42, n_jobs=-1, **sampled)
            cal_best = CalibratedClassifierCV(rf_best, method="isotonic", cv=3)
            best_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", cal_best)])
            best_pipeline.fit(x_train_best, y_train_best)

        bar.set_postfix(
            best=f"{best_cv_brier:.5f}",
            window=window,
            refresh=False,
        )

    bar.close()

    if best_pipeline is None:
        raise RuntimeError(
            "win_prob/random_forest: hyperparameter search produced no valid pipeline"
        )

    hold_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_hold_best)[:, 1], index=x_hold_best.index
    )
    holdout_brier: float = brier_score(hold_probs, y_hold_best.astype(float))
    holdout_ece: float = expected_calibration_error(hold_probs, y_hold_best.astype(float))
    holdout_auc: float = roc_auc(hold_probs, y_hold_best.astype(float))
    holdout_log_loss: float = log_loss(hold_probs, y_hold_best.astype(float))
    holdout_accuracy: float = accuracy(hold_probs, y_hold_best.astype(float))

    train_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_train_best)[:, 1], index=x_train_best.index
    )
    train_brier: float = brier_score(train_probs, y_train_best.astype(float))

    logger.info(
        "win_prob/random_forest: train=%.5f  holdout=%.5f  best_params=%s",
        train_brier,
        holdout_brier,
        best_params,
    )

    # Feature importances: average across calibration folds using vectorised
    # np.array().mean(axis=0) rather than a Python loop per feature.
    # CalibratedClassifierCV(cv=3) trains 3 internal fold clones;
    # clf.estimator is the *unfitted* original — use calibrated_classifiers_.
    cal_clf = best_pipeline.named_steps["clf"]
    fold_importances: ndarray = np.array(
        [cc.estimator.feature_importances_ for cc in cal_clf.calibrated_classifiers_]
    )
    importances: list[float] = fold_importances.mean(axis=0).tolist()
    importance_pairs: list[tuple[str, float]] = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    top10_importances: dict[str, float] = {f: round(imp, 6) for f, imp in importance_pairs[:10]}

    metadata = GameModelMetadata(
        model_name="win_prob",
        model_type="random_forest",
        task="classification",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        parameters={
            **best_params,
            "calibration_method": "isotonic",
            "train_brier": round(train_brier, 6),
            "holdout_ece": round(holdout_ece, 6),
            "holdout_auc": round(holdout_auc, 6),
            "holdout_log_loss": round(holdout_log_loss, 6),
            "holdout_accuracy": round(holdout_accuracy, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "cv_brier": round(best_cv_brier, 6),
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "n_train": len(x_train_best),
            "n_holdout": len(x_hold_best),
            "n_features": len(feature_names),
            "feature_set": f"combined_{len(feature_names)}",
            "top10_feature_importances": top10_importances,
        },
        feature_columns=feature_names,
        n_train_rows=len(x_train_best),
        n_holdout_rows=len(x_hold_best),
        holdout_brier=round(holdout_brier, 6),
    )

    store.save(metadata=metadata, model_obj=best_pipeline, overwrite=True)
    return metadata


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def _train_xgboost(
    df: pd.DataFrame,
    *,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> GameModelMetadata:
    """Train an XGBoost classifier with randomised hyperparameter search.

    Searches over n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, gamma, and epa_window.
    Isotonic calibration applied if holdout ECE exceeds 0.025.

    Optimisations vs the initial implementation:
    - Window cache: each unique EPA window is rebuilt and split at most once.
    - TimeSeriesSplit is instantiated once before the loop.
    - Feature importances use vectorised np.array().mean(axis=0).
    - ``feature_set`` metadata derived from feature_names length.

    Progress is reported via a tqdm bar showing iteration count, current
    best CV Brier, and ETA.

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        feature_fn: Feature engineering function.
        feature_names: Feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        GameModelMetadata with holdout Brier, best parameters, gain-based
        feature importances (top 10), and calibration status. Artifact is
        written to ``data/models/win_prob/xgboost/``.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import TimeSeriesSplit

    # pyrefly: ignore [missing-import]
    from sklearn.pipeline import Pipeline

    # pyrefly: ignore [missing-import]
    from sklearn.preprocessing import StandardScaler

    # pyrefly: ignore [missing-import]
    from xgboost import XGBClassifier

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        log_loss,
        roc_auc,
    )
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    param_grid: dict[str, list] = {
        "n_estimators": [100, 150, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10, 20],
        "gamma": [0.0, 0.1, 0.3, 0.5],
        "epa_window": _EPA_WINDOW_OPTIONS,
    }

    n_iter: int = 75
    cv_folds: int = 5
    rng: Generator = np.random.default_rng(42)

    window_cache: dict[int, WindowData] = {}

    _wd0: WindowData = _get_cached_window_data(window_cache, 4, df, feature_fn, resolved_repo)
    x_train_best, y_train_best = _wd0.x_train, _wd0.y_train
    x_hold_best, y_hold_best = _wd0.x_holdout, _wd0.y_holdout
    train_seasons, hold_seasons = _wd0.train_seasons, _wd0.holdout_seasons

    best_cv_brier: float = float("inf")
    best_params: dict = {}
    best_pipeline = None

    param_keys: list[str] = list(param_grid.keys())
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    bar = tqdm(
        range(n_iter),
        desc="  win_prob/xgboost",
        unit="iter",
        ncols=88,
        colour="cyan",
    )
    for iteration in bar:
        sampled: dict = {
            k: param_grid[k][int(rng.integers(len(param_grid[k])))] for k in param_keys
        }
        window: int = sampled.pop("epa_window")

        _wd: WindowData = _get_cached_window_data(
            window_cache, window, df, feature_fn, resolved_repo
        )
        x_train, y_train = _wd.x_train, _wd.y_train
        train_seasons, hold_seasons = _wd.train_seasons, _wd.holdout_seasons

        fold_briers: list[float] = []
        for train_idx, val_idx in tscv.split(x_train):
            if len(train_idx) < MIN_CV_TRAIN_ROWS:
                continue
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **sampled,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", xgb)])
            pipe.fit(x_tr, y_tr)
            val_probs: Series = pd.Series(pipe.predict_proba(x_val)[:, 1])
            fold_briers.append(brier_score(val_probs, y_val.astype(float).reset_index(drop=True)))

        cv_brier = float(np.mean(fold_briers))
        logger.debug(
            "  iter %d/%d: window=%d params=%s cv_brier=%.5f",
            iteration + 1,
            n_iter,
            window,
            sampled,
            cv_brier,
        )

        if cv_brier < best_cv_brier:
            best_cv_brier = cv_brier
            best_params = {**sampled, "epa_window": window}
            _wd = window_cache[window]
            x_train_best, y_train_best = _wd.x_train, _wd.y_train
            x_hold_best, y_hold_best = _wd.x_holdout, _wd.y_holdout
            train_seasons, hold_seasons = _wd.train_seasons, _wd.holdout_seasons
            xgb_best = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **sampled,
            )
            best_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", xgb_best)])
            best_pipeline.fit(x_train_best, y_train_best)

        bar.set_postfix(
            best=f"{best_cv_brier:.5f}",
            window=window,
            refresh=False,
        )

    bar.close()

    if best_pipeline is None:
        raise RuntimeError("win_prob/xgboost: hyperparameter search produced no valid pipeline")

    hold_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_hold_best)[:, 1], index=x_hold_best.index
    )
    holdout_brier: float = brier_score(hold_probs, y_hold_best.astype(float))
    holdout_ece: float = expected_calibration_error(hold_probs, y_hold_best.astype(float))
    holdout_auc: float = roc_auc(hold_probs, y_hold_best.astype(float))
    holdout_log_loss: float = log_loss(hold_probs, y_hold_best.astype(float))
    holdout_accuracy: float = accuracy(hold_probs, y_hold_best.astype(float))

    train_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_train_best)[:, 1], index=x_train_best.index
    )
    train_brier: float = brier_score(train_probs, y_train_best.astype(float))

    # Apply isotonic calibration if ECE indicates overconfidence
    _ece_calibration_threshold: float = 0.025
    calibration_applied: bool = False
    if holdout_ece > _ece_calibration_threshold:
        logger.info(
            "win_prob/xgboost: ECE=%.4f > %.3f — applying isotonic calibration",
            holdout_ece,
            _ece_calibration_threshold,
        )
        # pyrefly: ignore [missing-import]
        from sklearn.calibration import CalibratedClassifierCV

        xgb_recal = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **{k: v for k, v in best_params.items() if k != "epa_window"},
        )
        cal_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(xgb_recal, method="isotonic", cv=3)),
            ]
        )
        cal_pipeline.fit(x_train_best, y_train_best)
        hold_probs = pd.Series(
            cal_pipeline.predict_proba(x_hold_best)[:, 1], index=x_hold_best.index
        )
        holdout_brier = brier_score(hold_probs, y_hold_best.astype(float))
        holdout_ece = expected_calibration_error(hold_probs, y_hold_best.astype(float))
        best_pipeline = cal_pipeline
        calibration_applied = True
        logger.info(
            "win_prob/xgboost: post-calibration holdout Brier=%.5f  ECE=%.4f",
            holdout_brier,
            holdout_ece,
        )

    logger.info(
        "win_prob/xgboost: train=%.5f  holdout=%.5f  ECE=%.4f  calibrated=%s  best_params=%s",
        train_brier,
        holdout_brier,
        holdout_ece,
        calibration_applied,
        best_params,
    )

    # Feature importances: vectorised average across calibration folds.
    # Two cases depending on whether isotonic calibration was applied:
    #   - Not calibrated: pipeline["clf"] is XGBClassifier directly.
    #   - Calibrated: pipeline["clf"] is CalibratedClassifierCV; use
    #     calibrated_classifiers_[i].estimator for fitted instances.
    xgb_step = best_pipeline.named_steps["clf"]
    if hasattr(xgb_step, "calibrated_classifiers_"):
        fold_importances_xgb: ndarray = np.array(
            [cc.estimator.feature_importances_ for cc in xgb_step.calibrated_classifiers_]
        )
        importances_list: list[float] = fold_importances_xgb.mean(axis=0).tolist()
    else:
        importances_list = xgb_step.feature_importances_.tolist()

    importance_pairs: list[tuple[str, float]] = sorted(
        zip(feature_names, importances_list, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    top10_importances: dict[str, float] = {f: round(imp, 6) for f, imp in importance_pairs[:10]}

    metadata = GameModelMetadata(
        model_name="win_prob",
        model_type="xgboost",
        task="classification",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        parameters={
            **best_params,
            "calibration_applied": calibration_applied,
            "holdout_ece": round(holdout_ece, 6),
            "holdout_auc": round(holdout_auc, 6),
            "holdout_log_loss": round(holdout_log_loss, 6),
            "holdout_accuracy": round(holdout_accuracy, 6),
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "cv_brier": round(best_cv_brier, 6),
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "n_train": len(x_train_best),
            "n_holdout": len(x_hold_best),
            "n_features": len(feature_names),
            "feature_set": f"combined_{len(feature_names)}",
            "top10_feature_importances": top10_importances,
        },
        feature_columns=feature_names,
        n_train_rows=len(x_train_best),
        n_holdout_rows=len(x_hold_best),
        holdout_brier=round(holdout_brier, 6),
    )

    store.save(metadata=metadata, model_obj=best_pipeline, overwrite=True)
    return metadata


# ---------------------------------------------------------------------------
# Model variant factory
# ---------------------------------------------------------------------------


def _make_tree_variant(
    registry_key: str,
    description: str,
    *,
    model_name: str,
    feature_set: FeatureSet,
    model_type: Literal["rf", "xgb"],
) -> type:
    """Produce and register a tree-based predictor class for a given variant.

    Eliminates the ~35-line boilerplate class body required per variant.
    The produced class is functionally identical to a hand-written class:
    it has a ``spec``, implements ``train``, ``is_trained``,
    ``predict_historical``, and ``predict_upcoming``, and is registered
    with ``PredictorRegistry`` immediately.

    Note (Workstream 2): this factory will be deleted in D2b and replaced
    by ``GamesTrainer`` + ``WinProbTrainer`` spec subclasses. For D1b the
    factory stays so the registry surface is unchanged — only artifact
    paths move to the new ``data/models/{model_name}/{model_type}/`` scheme.

    Args:
        registry_key: ``PredictorRegistry`` key (e.g. ``"random_forest"``).
            Currently flat; becomes composite (``"win_prob_random_forest"``)
            in D2a/D2b.
        description: Human-readable description shown in ``gridiron models list``.
        model_name: Artifact ``model_name`` (e.g. ``"win_prob"``).
        feature_set: A ``FeatureSet`` from ``FEATURE_SETS``.
        model_type: ``"rf"`` for Random Forest, ``"xgb"`` for XGBoost.

    Returns:
        The produced and registered class object.
    """
    _train_fn = _train_random_forest if model_type == "rf" else _train_xgboost
    _feature_fn = feature_set.feature_fn
    _feature_names = feature_set.feature_names
    _artifact_type = "random_forest" if model_type == "rf" else "xgboost"

    def train(self: object, df: pd.DataFrame, *, repo: Path | None = None) -> GameModelMetadata:
        return _train_fn(
            df,
            feature_fn=_feature_fn,
            feature_names=_feature_names,
            repo=repo,
        )

    def is_trained(self: object, *, repo: Path | None = None) -> bool:
        # pyrefly: ignore [bad-argument-type]
        return _is_trained(model_name, _artifact_type, repo)

    def predict_historical(
        self: object, games: pd.DataFrame, *, repo: Path | None = None
    ) -> pd.DataFrame:
        return _predict_historical_tree(
            games,
            model_name=model_name,
            model_type=_artifact_type,
            feature_fn=_feature_fn,
            repo=repo,
        )

    def predict_upcoming(
        self: object, schedule: pd.DataFrame, *, repo: Path | None = None
    ) -> pd.DataFrame:
        return _predict_upcoming_tree(
            schedule,
            model_name=model_name,
            model_type=_artifact_type,
            feature_fn=_feature_fn,
            repo=repo,
        )

    family = "RandomForest" if model_type == "rf" else "XGBoost"
    cls_name = f"{family}{registry_key.title().replace('_', '')}Predictor"
    cls = type(
        cls_name,
        (),
        {
            "spec": PredictorSpec(name=registry_key, description=description, trainable=True),
            "train": train,
            "is_trained": is_trained,
            "predict_historical": predict_historical,
            "predict_upcoming": predict_upcoming,
            "__doc__": description,
        },
    )
    PredictorRegistry.register(cls)
    return cls


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------

RandomForestPredictor = _make_tree_variant(
    "random_forest",
    "Random Forest — expanded features, isotonic calibration, TimeSeriesSplit CV",
    model_name="win_prob",
    feature_set=FEATURE_SETS["expanded"],
    model_type="rf",
)

XGBoostPredictor = _make_tree_variant(
    "xgboost",
    "XGBoost gradient boosting — expanded features, conditional calibration, TimeSeriesSplit CV",
    model_name="win_prob",
    feature_set=FEATURE_SETS["expanded"],
    model_type="xgb",
)
