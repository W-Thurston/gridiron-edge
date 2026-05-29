# src/gridiron_edge/models/game_prediction/tree.py

"""Tree-based game prediction models (Phase 20d).

Two variants built on the same 32-feature combined set as logistic_v3,
designed to capture non-linear EPA x Elo interaction effects:

    random_forest_v1: Random Forest with isotonic calibration
        RandomizedSearchCV over n_estimators, max_depth, min_samples_leaf,
        max_features, and EPA rolling window.  CalibratedClassifierCV
        (isotonic) applied post-fit to correct systematic RF overconfidence.
        Expected Brier: 0.218-0.222.

    xgboost_v1: Gradient boosted trees
        RandomizedSearchCV over n_estimators, max_depth, learning_rate,
        subsample, colsample_bytree, min_child_weight, gamma, and EPA window.
        XGBoost's binary:logistic objective produces well-calibrated
        probabilities natively; isotonic calibration applied if ECE > 0.025.
        Expected Brier: 0.215-0.220.

Both models tune the EPA rolling window as a hyperparameter (resolves the
rolling-window-as-hyperparameter backlog item from Phase 19).

Training progress is reported via tqdm: one bar per model showing
iteration count, current best CV Brier, and ETA.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import datetime as dt
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm

from gridiron_edge.models.base import PredictorSpec
from gridiron_edge.models.game_prediction._shared import (
    _COMBINED_FEATURES,
    _EXPANDED_FEATURES,
    _SCHEMA_VERSION,
    _is_trained,
    _make_combined_features,
    _make_expanded_features,
    _prepare_data,
)
from gridiron_edge.models.registry import PredictorRegistry

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import ModelMetadata

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EPA window infrastructure
# ---------------------------------------------------------------------------

# EPA column suffixes as they appear in epa_by_game.parquet (lowercase)
_EPA_RAW_COLS: Final[list[str]] = [
    "off_epa_per_play",
    "off_pass_epa",
    "off_rush_epa",
    "off_success_rate",
    "def_epa_per_play",
    "def_pass_epa",
    "def_rush_epa",
    "def_success_rate",
]

# Mapping from epa_by_game column name → modeling file column suffix
_EPA_COL_MAP: Final[dict[str, str]] = {
    "off_epa_per_play": "OFF_EPA_PER_PLAY",
    "off_pass_epa": "OFF_PASS_EPA",
    "off_rush_epa": "OFF_RUSH_EPA",
    "off_success_rate": "OFF_SUCCESS_RATE",
    "def_epa_per_play": "DEF_EPA_PER_PLAY",
    "def_pass_epa": "DEF_PASS_EPA",
    "def_rush_epa": "DEF_RUSH_EPA",
    "def_success_rate": "DEF_SUCCESS_RATE",
}

# EPA window values searched as a hyperparameter
_EPA_WINDOW_OPTIONS: Final[list[int]] = [1, 2, 3, 4, 6, 8]


def _rebuild_features_with_window(
    df: pd.DataFrame,
    *,
    window: int,
    repo: Path,
) -> pd.DataFrame:
    """Recompute rolling EPA features with a configurable window size.

    The standard modeling file uses a fixed 4-game rolling window.  This
    function loads the raw game-level EPA data and recomputes rolling
    averages with a different window, then splices the result back into
    the modeling DataFrame.  Called during hyperparameter search when
    epa_window is a tunable parameter.

    Fast path: if window == 4, returns df unchanged (no disk read needed).

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        window: Rolling window size (number of prior games to average).
        repo: Repository root (for loading epa_by_game.parquet).

    Returns:
        Modeling DataFrame with TEAM_A_* and TEAM_B_* EPA columns
        recomputed using the requested window.  NaN rows from incomplete
        windows are retained; callers apply the NaN mask after feature
        engineering.
    """
    from gridiron_edge.datasets.loaders import load_epa_by_game

    if window == 4:
        return df

    epa_raw: pd.DataFrame = load_epa_by_game(repo)
    if epa_raw.empty:
        logger.warning("epa_by_game.parquet not found — returning df unchanged")
        return df

    epa_sorted: pd.DataFrame = epa_raw.sort_values(["season", "week", "team"]).copy()

    # Compute rolling mean per team with shift(1) to prevent lookahead
    rolled_parts: list[pd.DataFrame] = []
    for _team, grp in epa_sorted.groupby("team", sort=False):
        grp_sorted = grp.sort_values(["season", "week"]).copy()
        for col in _EPA_RAW_COLS:
            grp_sorted[f"{col}_roll"] = (
                grp_sorted[col].shift(1).rolling(window=window, min_periods=1).mean()
            )
        rolled_parts.append(grp_sorted)

    rolled: pd.DataFrame = pd.concat(rolled_parts, ignore_index=True)

    roll_cols: list[str] = [f"{c}_roll" for c in _EPA_RAW_COLS]
    lookup: pd.DataFrame = rolled[["season", "week", "team", *roll_cols]].copy()

    # Build season int → YEAR string mapping from the modeling file itself
    # (epa_by_game uses int seasons like 2024; modeling file uses "2024-2025")
    year_to_season: dict[str, int] = {}
    for year_str in df["YEAR"].unique():
        with contextlib.suppress(ValueError, IndexError):
            year_to_season[year_str] = int(str(year_str).split("-")[0])

    lookup["YEAR"] = lookup["season"].map({v: k for k, v in year_to_season.items()})
    lookup = lookup.dropna(subset=["YEAR"])

    # Drop existing EPA columns before merging updated ones
    team_a_epa_cols = [f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]
    team_b_epa_cols = [f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]
    df = df.copy().drop(columns=team_a_epa_cols + team_b_epa_cols, errors="ignore")

    # Merge TEAM_A EPA
    team_a_merge = lookup.rename(
        columns={
            "team": "TEAM_A",
            "week": "WEEK_NUM",
            **{f"{c}_roll": f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS},
        }
    )[["TEAM_A", "YEAR", "WEEK_NUM", *[f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]]]
    df = df.merge(team_a_merge, on=["TEAM_A", "YEAR", "WEEK_NUM"], how="left")

    # Merge TEAM_B EPA
    team_b_merge = lookup.rename(
        columns={
            "team": "TEAM_B",
            "week": "WEEK_NUM",
            **{f"{c}_roll": f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS},
        }
    )[["TEAM_B", "YEAR", "WEEK_NUM", *[f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]]]
    df = df.merge(team_b_merge, on=["TEAM_B", "YEAR", "WEEK_NUM"], how="left")

    return df


# ---------------------------------------------------------------------------
# Shared prediction helpers for tree models
# ---------------------------------------------------------------------------


def _predict_historical_tree(
    games: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared historical prediction logic for tree model variants.

    Reads epa_window from stored metadata and rebuilds EPA features
    accordingly, ensuring predictions always use the same window as
    training.

    Args:
        games: Games DataFrame (unused — modeling file loaded from disk).
        model_version: Registered model version string.
        feature_fn: Feature engineering function.
        repo: Repository root.

    Returns:
        Prediction DataFrame in the standard archive format.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    if not store.is_trained(model_version):
        logger.warning("%s: no artifact found.", model_version)
        return pd.DataFrame()

    pipeline = store.load(model_version)
    metadata = store.read_metadata(model_version)
    epa_window: int = int(metadata.parameters.get("epa_window", 4))

    df: DataFrame = load_modeling_file(resolved_repo, required_schema_version=_SCHEMA_VERSION)

    if epa_window != 4:
        df = _rebuild_features_with_window(df, window=epa_window, repo=resolved_repo)

    features = feature_fn(df)
    valid = features.notna().all(axis=1)
    df_valid = df.loc[valid].copy()
    x_feat = features.loc[valid]

    if x_feat.empty:
        return pd.DataFrame()

    probs = pipeline.predict_proba(x_feat)[:, 1]
    df_valid = df_valid.copy()
    df_valid["_prob"] = probs

    away_rows = df_valid.loc[df_valid["HOME_FIELD"] == 0].copy()
    away_rows = away_rows.drop_duplicates(subset=["GAME_ID"])

    ts = dt.datetime(1970, 1, 1)
    return pd.DataFrame(
        {
            "predicted_at": ts,
            "model_version": model_version,
            "season": away_rows["YEAR"],
            "week": away_rows["WEEK_NUM"].astype(int),
            "game_id": away_rows["GAME_ID"],
            "game_date": "",
            "away_team": away_rows["TEAM_A"],
            "home_team": away_rows["TEAM_B"],
            "away_elo": float("nan"),
            "home_elo": float("nan"),
            "away_win_prob": away_rows["_prob"],
            "home_win_prob": 1.0 - away_rows["_prob"],
        }
    ).reset_index(drop=True)


def _predict_upcoming_tree(
    schedule: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared upcoming prediction logic for tree model variants.

    Args:
        schedule: Upcoming games schedule DataFrame.
        model_version: Registered model version string.
        feature_fn: Feature engineering function.
        repo: Repository root.

    Returns:
        Prediction DataFrame with AWAY_WIN_PROB / HOME_WIN_PROB columns.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.accessor import DatasetAccessor
    from gridiron_edge.features.pipeline import FEATURES
    from gridiron_edge.features.registry import run_features
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    if not store.is_trained(model_version):
        logger.warning("%s: no artifact found.", model_version)
        return pd.DataFrame()

    pipeline = store.load(model_version)
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
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------


def _train_random_forest(
    df: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> ModelMetadata:
    """Train a Random Forest classifier with randomised hyperparameter search.

    Searches over n_estimators, max_depth, min_samples_leaf, max_features,
    and epa_window.  CalibratedClassifierCV (isotonic) is applied post-fit
    to correct the systematic overconfidence in RF predict_proba.

    Progress is reported via a tqdm bar showing iteration count, current
    best CV Brier, and ETA.

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        model_version: Model version string for artifact naming.
        feature_fn: Feature engineering function (combined features).
        feature_names: Feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        ModelMetadata with holdout Brier, best parameters, and top-10
        feature importances by mean decrease in impurity.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.calibration import CalibratedClassifierCV

    # pyrefly: ignore [missing-import]
    from sklearn.ensemble import RandomForestClassifier

    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import StratifiedKFold

    # pyrefly: ignore [missing-import]
    from sklearn.pipeline import Pipeline

    # pyrefly: ignore [missing-import]
    from sklearn.preprocessing import StandardScaler

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import brier_score
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
    from gridiron_edge.models.artifact import ArtifactStore, ModelMetadata

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
    rng = np.random.default_rng(42)

    # Pre-initialise *_best with the standard window=4 split so static
    # analysis can verify they are always bound before use.  These are
    # overwritten on the first iteration (initial best_cv_brier = inf
    # guarantees the branch fires).
    _df_init = _rebuild_features_with_window(df, window=4, repo=resolved_repo)
    x_train_best, y_train_best, x_hold_best, y_hold_best, train_seasons, hold_seasons = (
        _prepare_data(_df_init, feature_fn)
    )

    best_cv_brier: float = float("inf")
    best_params: dict = {}
    best_pipeline = None

    param_keys = list(param_grid.keys())

    bar = tqdm(
        range(n_iter),
        desc=f"  {model_version}",
        unit="iter",
        ncols=88,
        colour="cyan",
    )
    for iteration in bar:
        sampled: dict = {
            k: param_grid[k][int(rng.integers(len(param_grid[k])))] for k in param_keys
        }
        window: int = sampled.pop("epa_window")

        df_w = _rebuild_features_with_window(df, window=window, repo=resolved_repo)
        x_train, y_train, _x_hold, _y_hold, train_seasons, hold_seasons = _prepare_data(
            df_w, feature_fn
        )

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_briers: list[float] = []
        for train_idx, val_idx in skf.split(x_train, y_train):
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            rf = RandomForestClassifier(random_state=42, n_jobs=-1, **sampled)
            cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", cal)])
            pipe.fit(x_tr, y_tr)
            val_probs = pd.Series(pipe.predict_proba(x_val)[:, 1])
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
            df_best = _rebuild_features_with_window(df, window=window, repo=resolved_repo)
            x_train_best, y_train_best, x_hold_best, y_hold_best, train_seasons, hold_seasons = (
                _prepare_data(df_best, feature_fn)
            )
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
        raise RuntimeError(f"{model_version}: hyperparameter search produced no valid pipeline")

    hold_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_hold_best)[:, 1], index=x_hold_best.index
    )
    holdout_brier: float = brier_score(hold_probs, y_hold_best.astype(float))

    train_probs: Series = pd.Series(
        best_pipeline.predict_proba(x_train_best)[:, 1], index=x_train_best.index
    )
    train_brier: float = brier_score(train_probs, y_train_best.astype(float))

    logger.info(
        "%s: train=%.5f  holdout=%.5f  best_params=%s",
        model_version,
        train_brier,
        holdout_brier,
        best_params,
    )

    # Extract feature importances by averaging across the CV folds stored
    # in calibrated_classifiers_.  CalibratedClassifierCV(cv=3) trains 3
    # internal fold clones; clf.estimator is the *unfitted* original and
    # must not be used.  Averaging across folds gives a more stable
    # importance estimate than any single fold.
    cal_clf = best_pipeline.named_steps["clf"]
    fold_importances: list[list[float]] = [
        cc.estimator.feature_importances_.tolist() for cc in cal_clf.calibrated_classifiers_
    ]
    importances: list[float] = [
        float(np.mean([fold[i] for fold in fold_importances])) for i in range(len(feature_names))
    ]
    importance_pairs = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    top10_importances: dict[str, float] = {f: round(imp, 6) for f, imp in importance_pairs[:10]}

    metadata = ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        holdout_brier=round(holdout_brier, 6),
        parameters={
            **best_params,
            "calibration_method": "isotonic",
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "cv_brier": round(best_cv_brier, 6),
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "n_train": len(x_train_best),
            "n_holdout": len(x_hold_best),
            "n_features": len(feature_names),
            "feature_set": "combined_32",
            "top10_feature_importances": top10_importances,
        },
        feature_columns=feature_names,
    )

    store.save(model_version, best_pipeline, metadata=metadata)
    return metadata


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def _train_xgboost(
    df: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> ModelMetadata:
    """Train an XGBoost classifier with randomised hyperparameter search.

    Searches over n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, gamma, and epa_window.
    Isotonic calibration applied if holdout ECE exceeds 0.025.

    Progress is reported via a tqdm bar showing iteration count, current
    best CV Brier, and ETA.

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        model_version: Model version string for artifact naming.
        feature_fn: Feature engineering function (combined features).
        feature_names: Feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        ModelMetadata with holdout Brier, best parameters, gain-based
        feature importances (top 10), and calibration status.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import StratifiedKFold

    # pyrefly: ignore [missing-import]
    from sklearn.pipeline import Pipeline

    # pyrefly: ignore [missing-import]
    from sklearn.preprocessing import StandardScaler

    # pyrefly: ignore [missing-import]
    from xgboost import XGBClassifier

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import brier_score, expected_calibration_error
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION
    from gridiron_edge.models.artifact import ArtifactStore, ModelMetadata

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
    rng = np.random.default_rng(42)

    _df_init = _rebuild_features_with_window(df, window=4, repo=resolved_repo)
    x_train_best, y_train_best, x_hold_best, y_hold_best, train_seasons, hold_seasons = (
        _prepare_data(_df_init, feature_fn)
    )

    best_cv_brier: float = float("inf")
    best_params: dict = {}
    best_pipeline = None

    param_keys = list(param_grid.keys())

    bar = tqdm(
        range(n_iter),
        desc=f"  {model_version}",
        unit="iter",
        ncols=88,
        colour="cyan",
    )
    for iteration in bar:
        sampled: dict = {
            k: param_grid[k][int(rng.integers(len(param_grid[k])))] for k in param_keys
        }
        window: int = sampled.pop("epa_window")

        df_w = _rebuild_features_with_window(df, window=window, repo=resolved_repo)
        x_train, y_train, _x_hold, _y_hold, train_seasons, hold_seasons = _prepare_data(
            df_w, feature_fn
        )

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_briers: list[float] = []
        for train_idx, val_idx in skf.split(x_train, y_train):
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
            val_probs = pd.Series(pipe.predict_proba(x_val)[:, 1])
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
            df_best = _rebuild_features_with_window(df, window=window, repo=resolved_repo)
            x_train_best, y_train_best, x_hold_best, y_hold_best, train_seasons, hold_seasons = (
                _prepare_data(df_best, feature_fn)
            )
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
        raise RuntimeError(f"{model_version}: hyperparameter search produced no valid pipeline")

    hold_probs = pd.Series(best_pipeline.predict_proba(x_hold_best)[:, 1], index=x_hold_best.index)
    holdout_brier = brier_score(hold_probs, y_hold_best.astype(float))
    holdout_ece: float = expected_calibration_error(hold_probs, y_hold_best.astype(float))

    train_probs = pd.Series(
        best_pipeline.predict_proba(x_train_best)[:, 1], index=x_train_best.index
    )
    train_brier = brier_score(train_probs, y_train_best.astype(float))

    # Apply isotonic calibration if ECE indicates overconfidence
    _ece_calibration_threshold: float = 0.025
    calibration_applied: bool = False
    if holdout_ece > _ece_calibration_threshold:
        logger.info(
            "%s: ECE=%.4f > %.3f — applying isotonic calibration",
            model_version,
            holdout_ece,
            _ece_calibration_threshold,
        )
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
            "%s: post-calibration holdout Brier=%.5f  ECE=%.4f",
            model_version,
            holdout_brier,
            holdout_ece,
        )

    logger.info(
        "%s: train=%.5f  holdout=%.5f  ECE=%.4f  calibrated=%s  best_params=%s",
        model_version,
        train_brier,
        holdout_brier,
        holdout_ece,
        calibration_applied,
        best_params,
    )

    # Extract gain-based feature importances.
    # Two cases depending on whether isotonic calibration was applied:
    #   - Not calibrated: pipeline["clf"] is XGBClassifier directly.
    #   - Calibrated: pipeline["clf"] is CalibratedClassifierCV wrapping
    #     XGBClassifier.  clf.estimator is the unfitted original — must use
    #     calibrated_classifiers_[i].estimator for the fitted instances.
    #     Average across folds for a stable importance estimate.
    xgb_step = best_pipeline.named_steps["clf"]
    if hasattr(xgb_step, "calibrated_classifiers_"):
        fold_importances_xgb: list[list[float]] = [
            cc.estimator.feature_importances_.tolist() for cc in xgb_step.calibrated_classifiers_
        ]
        feat_imp_arr = [
            float(np.mean([fold[i] for fold in fold_importances_xgb]))
            for i in range(len(feature_names))
        ]
        feat_imp = feat_imp_arr
    else:
        feat_imp = xgb_step.feature_importances_.tolist()

    feat_imp_list: list[float] = feat_imp if isinstance(feat_imp, list) else feat_imp.tolist()
    importance_pairs = sorted(
        zip(feature_names, feat_imp_list, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    top10_importances: dict[str, float] = {f: round(imp, 6) for f, imp in importance_pairs[:10]}

    metadata = ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        holdout_brier=round(holdout_brier, 6),
        parameters={
            **best_params,
            "calibration_applied": calibration_applied,
            "holdout_ece": round(holdout_ece, 6),
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "cv_brier": round(best_cv_brier, 6),
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "n_train": len(x_train_best),
            "n_holdout": len(x_hold_best),
            "n_features": len(feature_names),
            "feature_set": "combined_32",
            "top10_feature_importances": top10_importances,
        },
        feature_columns=feature_names,
    )

    store.save(model_version, best_pipeline, metadata=metadata)
    return metadata


# ---------------------------------------------------------------------------
# random_forest_v1
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class RandomForestV1Predictor:
    """Random Forest classifier on combined features (32) with isotonic calibration.

    Addresses the logistic ceiling (Brier 0.22057) by capturing non-linear
    EPA x Elo interaction effects through tree splitting.  Isotonic
    calibration corrects systematic RF overconfidence, targeting ECE
    comparable to logistic_v3's 0.015.

    Tuned: n_estimators, max_depth, min_samples_leaf, max_features,
    epa_window (resolves rolling-window-as-hyperparameter backlog item).
    """

    spec = PredictorSpec(
        name="random_forest_v1",
        description="Random Forest — combined features (32), isotonic calibration",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train Random Forest with randomised hyperparameter search."""
        return _train_random_forest(
            df,
            model_version="random_forest_v1",
            feature_fn=_make_combined_features,
            feature_names=_COMBINED_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for random_forest_v1."""
        return _is_trained("random_forest_v1", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate random_forest_v1 predictions for all historical games."""
        return _predict_historical_tree(
            games,
            model_version="random_forest_v1",
            feature_fn=_make_combined_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate random_forest_v1 predictions for upcoming games."""
        return _predict_upcoming_tree(
            schedule,
            model_version="random_forest_v1",
            feature_fn=_make_combined_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# xgboost_v1
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class XGBoostV1Predictor:
    """XGBoost gradient boosted trees on combined features (32).

    Captures non-linear EPA x Elo interactions via sequential boosting.
    Uses binary:logistic objective which produces well-calibrated
    probabilities natively.  Isotonic calibration applied automatically
    if holdout ECE exceeds 0.025.

    Tuned: n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, gamma, epa_window.
    """

    spec = PredictorSpec(
        name="xgboost_v1",
        description="XGBoost gradient boosting — combined features (32)",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train XGBoost with randomised hyperparameter search."""
        return _train_xgboost(
            df,
            model_version="xgboost_v1",
            feature_fn=_make_combined_features,
            feature_names=_COMBINED_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for xgboost_v1."""
        return _is_trained("xgboost_v1", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate xgboost_v1 predictions for all historical games."""
        return _predict_historical_tree(
            games,
            model_version="xgboost_v1",
            feature_fn=_make_combined_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate xgboost_v1 predictions for upcoming games."""
        return _predict_upcoming_tree(
            schedule,
            model_version="xgboost_v1",
            feature_fn=_make_combined_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# random_forest_v2
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class RandomForestV2Predictor:
    """Random Forest on expanded Phase 20e feature set (51 features).

    Extends random_forest_v1 with all Phase 20e Category A features:
    rest/schedule stress, weather effects, travel, divisional game flag,
    and franchise-level home field advantage coefficient.

    Same architecture and hyperparameter search as v1; expanded feature
    set allows the model to capture fatigue, scheduling, and environmental
    effects that the v1 combined feature set cannot represent.
    """

    spec = PredictorSpec(
        name="random_forest_v2",
        description="Random Forest — expanded Phase 20e features (51), isotonic calibration",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train Random Forest v2 on the expanded Phase 20e feature set."""
        return _train_random_forest(
            df,
            model_version="random_forest_v2",
            feature_fn=_make_expanded_features,
            feature_names=_EXPANDED_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for random_forest_v2."""
        return _is_trained("random_forest_v2", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate random_forest_v2 predictions for all historical games."""
        return _predict_historical_tree(
            games,
            model_version="random_forest_v2",
            feature_fn=_make_expanded_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate random_forest_v2 predictions for upcoming games."""
        return _predict_upcoming_tree(
            schedule,
            model_version="random_forest_v2",
            feature_fn=_make_expanded_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# xgboost_v2
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class XGBoostV2Predictor:
    """XGBoost on expanded Phase 20e feature set (51 features).

    Extends xgboost_v1 with all Phase 20e Category A features:
    rest/schedule stress, weather effects, travel, divisional game flag,
    and franchise-level home field advantage coefficient.

    Same architecture and hyperparameter search as v1.  XGBoost's
    tree-splitting mechanism is well-suited to the mix of binary flags
    (IS_DIV_GAME, IS_DOME, SHORT_WEEK, POST_BYE) and continuous values
    (DAYS_REST, WIND_SPEED_MPH, TEMP_F, KM_TRAVELED) in the new features.
    """

    spec = PredictorSpec(
        name="xgboost_v2",
        description="XGBoost gradient boosting — expanded Phase 20e features (51)",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train XGBoost v2 on the expanded Phase 20e feature set."""
        return _train_xgboost(
            df,
            model_version="xgboost_v2",
            feature_fn=_make_expanded_features,
            feature_names=_EXPANDED_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for xgboost_v2."""
        return _is_trained("xgboost_v2", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate xgboost_v2 predictions for all historical games."""
        return _predict_historical_tree(
            games,
            model_version="xgboost_v2",
            feature_fn=_make_expanded_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate xgboost_v2 predictions for upcoming games."""
        return _predict_upcoming_tree(
            schedule,
            model_version="xgboost_v2",
            feature_fn=_make_expanded_features,
            repo=repo,
        )
