# src/gridiron_edge/models/game_prediction/logistic.py

"""Logistic regression game prediction models.

Variants:

    logistic: Combined differential + raw features (32 features)
        - LogisticRegressionCV with TimeSeriesSplit CV
        - L2 regularisation, 10 candidate C values
        - StandardScaler preprocessing

    elasticnet (factory only, not registered): Same data shape, SAGA solver,
        elastic net penalty tuning both C and l1_ratio. Retained as a
        train helper for future champions. Removed in D2b.

All variants:
    - Same holdout split (HOLDOUT_SEASONS) for fair comparison with tree models
    - LogisticRegressionCV with 5-fold CV over 10 regularisation strengths
    - StandardScaler in sklearn Pipeline
    - Ties excluded (15 games since 1999, ~0.2%)
    - NaN rows excluded (covers pre-2006 and incomplete rolling windows)

Artifact storage (Workstream 2): trainers write to
``data/models/{model_name}/{model_type}/``. ``model_name`` is ``"win_prob"``
for all variants; ``model_type`` is ``"logistic"`` or ``"elasticnet"``.
The factory pattern (``_make_logistic_variant``) and ``_train_elasticnet``
are killed in D2b — replaced by ``GamesTrainer`` + ``WinProbTrainer``.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from logging import Logger
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.models.base import PredictorSpec
from gridiron_edge.models.game_prediction._columns import (
    FeatureSet,
)
from gridiron_edge.models.game_prediction._features import (
    FEATURE_SETS,
    _is_trained,
    _prepare_data,
)
from gridiron_edge.models.game_prediction.base import GameModelMetadata

# pyrefly: ignore [missing-import]
from gridiron_edge.models.game_prediction.pipeline import predict_games
from gridiron_edge.models.game_prediction.post_process import enrich_predictions
from gridiron_edge.models.registry import PredictorRegistry

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared logistic training logic
# ---------------------------------------------------------------------------


def _train_logistic(
    df: pd.DataFrame,
    *,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> GameModelMetadata:
    """Shared logistic regression training logic for all variants.

    Uses LogisticRegressionCV with 5-fold CV over 10 regularisation
    strengths (Cs), scored by neg_brier_score. Writes the artifact to
    ``data/models/win_prob/logistic/``.

    Args:
        df: Full modeling DataFrame.
        feature_fn: Feature engineering function.
        feature_names: List of feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        GameModelMetadata with holdout Brier and training details.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegressionCV

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
    from gridiron_edge.models.game_prediction._features import MIN_CV_TRAIN_ROWS

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = _prepare_data(df, feature_fn)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=[  # pyrefly: ignore[bad-argument-type]
                        (train_idx, val_idx)
                        for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(x_train)
                        if len(train_idx) >= MIN_CV_TRAIN_ROWS
                    ],
                    scoring="neg_brier_score",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Fitting win_prob/logistic on %d training rows...", len(x_train))
    pipeline.fit(x_train, y_train)

    hold_probs: Series = pd.Series(pipeline.predict_proba(x_hold)[:, 1], index=x_hold.index)
    holdout_brier: float = brier_score(hold_probs, y_hold.astype(float))
    holdout_ece: float = expected_calibration_error(hold_probs, y_hold.astype(float))
    holdout_auc: float = roc_auc(hold_probs, y_hold.astype(float))
    holdout_log_loss: float = log_loss(hold_probs, y_hold.astype(float))
    holdout_accuracy: float = accuracy(hold_probs, y_hold.astype(float))

    train_probs: Series = pd.Series(pipeline.predict_proba(x_train)[:, 1], index=x_train.index)
    train_brier: float = brier_score(train_probs, y_train.astype(float))
    best_c = float(pipeline.named_steps["clf"].C_[0])

    logger.info(
        "win_prob/logistic: train=%.5f  holdout=%.5f  C=%.4f",
        train_brier,
        holdout_brier,
        best_c,
    )

    metadata = GameModelMetadata(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        parameters={
            "best_C": best_c,
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "holdout_ece": round(holdout_ece, 6),
            "holdout_auc": round(holdout_auc, 6),
            "holdout_log_loss": round(holdout_log_loss, 6),
            "holdout_accuracy": round(holdout_accuracy, 6),
            "n_train": len(x_train),
            "n_holdout": len(x_hold),
            "n_features": len(feature_names),
            "feature_set": f"combined_{len(feature_names)}",
            "epa_window": 4,
            "cv_folds": 5,
        },
        feature_columns=feature_names,
        n_train_rows=len(x_train),
        n_holdout_rows=len(x_hold),
        holdout_brier=round(holdout_brier, 6),
    )

    store.save(metadata=metadata, model_obj=pipeline, overwrite=True)
    return metadata


def _train_elasticnet(
    df: pd.DataFrame,
    *,
    feature_fn: Callable,
    feature_names: list[str],
    l1_ratios: list[float],
    repo: Path | None,
) -> GameModelMetadata:
    """Train elastic net logistic regression and save artifact.

    Uses the SAGA solver which supports elastic net penalties. Tunes
    both regularisation strength C and L1/L2 mix ratio via CV. Writes
    the artifact to ``data/models/win_prob/elasticnet/``.

    Currently not wired through the registry — retained as a future
    champion candidate. Removed in D2b.

    Args:
        df: Full modeling DataFrame.
        feature_fn: Feature engineering function.
        feature_names: Feature column names produced by feature_fn.
        l1_ratios: L1/L2 mix ratios to search (0.0=ridge, 1.0=lasso).
        repo: Repository root.

    Returns:
        GameModelMetadata with holdout Brier, best C, and best l1_ratio.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegressionCV

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
    from gridiron_edge.models.game_prediction._features import MIN_CV_TRAIN_ROWS

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = _prepare_data(df, feature_fn)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=[  # pyrefly: ignore[bad-argument-type]
                        (train_idx, val_idx)
                        for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(x_train)
                        if len(train_idx) >= MIN_CV_TRAIN_ROWS
                    ],
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=l1_ratios,
                    scoring="neg_brier_score",
                    max_iter=3000,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Fitting win_prob/elasticnet on %d training rows...", len(x_train))
    pipeline.fit(x_train, y_train)

    clf = pipeline.named_steps["clf"]
    best_c = float(clf.C_[0])
    best_l1_ratio = float(clf.l1_ratio_[0])

    hold_probs: Series = pd.Series(pipeline.predict_proba(x_hold)[:, 1], index=x_hold.index)
    holdout_brier: float = brier_score(hold_probs, y_hold.astype(float))
    holdout_ece: float = expected_calibration_error(hold_probs, y_hold.astype(float))
    holdout_auc: float = roc_auc(hold_probs, y_hold.astype(float))
    holdout_log_loss: float = log_loss(hold_probs, y_hold.astype(float))
    holdout_accuracy: float = accuracy(hold_probs, y_hold.astype(float))

    train_probs: Series = pd.Series(pipeline.predict_proba(x_train)[:, 1], index=x_train.index)
    train_brier: float = brier_score(train_probs, y_train.astype(float))

    logger.info(
        "win_prob/elasticnet: train=%.5f  holdout=%.5f  C=%.4f  l1_ratio=%.2f",
        train_brier,
        holdout_brier,
        best_c,
        best_l1_ratio,
    )

    coefs = clf.coef_[0]
    nonzero_features: list[str] = [
        f for f, c in zip(feature_names, coefs, strict=True) if abs(c) > 1e-6
    ]

    metadata = GameModelMetadata(
        model_name="win_prob",
        model_type="elasticnet",
        task="classification",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        parameters={
            "best_C": best_c,
            "best_l1_ratio": best_l1_ratio,
            "l1_ratios_searched": l1_ratios,
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "holdout_ece": round(holdout_ece, 6),
            "holdout_auc": round(holdout_auc, 6),
            "holdout_log_loss": round(holdout_log_loss, 6),
            "holdout_accuracy": round(holdout_accuracy, 6),
            "n_train": len(x_train),
            "n_holdout": len(x_hold),
            "n_features": len(feature_names),
            "n_nonzero_features": len(nonzero_features),
            "nonzero_features": nonzero_features,
            "feature_set": f"combined_{len(feature_names)}",
            "epa_window": 4,
            "cv_folds": 5,
        },
        feature_columns=feature_names,
        n_train_rows=len(x_train),
        n_holdout_rows=len(x_hold),
        holdout_brier=round(holdout_brier, 6),
    )

    store.save(metadata=metadata, model_obj=pipeline, overwrite=True)
    return metadata


# ---------------------------------------------------------------------------
# Shared prediction helpers
# ---------------------------------------------------------------------------


def _predict_historical_logistic(
    games: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared historical prediction logic for all logistic variants."""
    return predict_games(
        model_name=model_name,
        model_type=model_type,
        feature_fn=feature_fn,
        repo=repo,
        is_backfilled=True,
    )


def _predict_upcoming_logistic(
    schedule: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared upcoming prediction logic for all logistic variants.

    Args:
        schedule: Upcoming games schedule DataFrame.
        model_name: Model purpose (``"win_prob"``).
        model_type: Model algorithm (``"logistic"`` or ``"elasticnet"``).
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

    result: DataFrame = enrich_predictions(
        result,
        model_name=model_name,
        model_type=model_type,
        recalibrate=True,
        repo=resolved_repo,
    )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model variant factory
# ---------------------------------------------------------------------------


def _make_logistic_variant(
    registry_key: str,
    description: str,
    *,
    model_name: str,
    feature_set: FeatureSet,
    elasticnet: bool = False,
    l1_ratios: list[float] | None = None,
) -> type:
    """Produce and register a logistic predictor class for a given variant.

    Eliminates the ~35-line boilerplate class body required per variant.
    The produced class is functionally identical to a hand-written class.

    Note (Workstream 2): this factory will be deleted in D2b and replaced
    by ``GamesTrainer`` + ``WinProbTrainer``. For D1b the factory stays so
    the registry surface is unchanged — only artifact paths move to the
    new ``data/models/{model_name}/{model_type}/`` scheme.

    Args:
        registry_key: ``PredictorRegistry`` key (e.g. ``"logistic"``).
            Currently flat; becomes composite (``"win_prob_logistic"``)
            in D2a/D2b.
        description: Human-readable description shown in ``gridiron models list``.
        model_name: Artifact ``model_name`` (e.g. ``"win_prob"``).
        feature_set: A ``FeatureSet`` from ``FEATURE_SETS``.
        elasticnet: If ``True``, use ``_train_elasticnet``; otherwise
            ``_train_logistic``. Drives both the trainer choice and the
            artifact ``model_type`` ("elasticnet" vs "logistic").
        l1_ratios: L1/L2 mix ratios passed to ``_train_elasticnet``.
            Ignored when ``elasticnet=False``. Defaults to
            ``[0.0, 0.1, 0.5, 0.9, 1.0]``.

    Returns:
        The produced and registered class object.
    """
    _feature_fn = feature_set.feature_fn
    _feature_names: list[str] = feature_set.feature_names
    _l1_ratios: list[float] = l1_ratios if l1_ratios is not None else [0.0, 0.1, 0.5, 0.9, 1.0]
    _artifact_type: str = "elasticnet" if elasticnet else "logistic"

    def train(self: object, df: pd.DataFrame, *, repo: Path | None = None) -> GameModelMetadata:
        if elasticnet:
            return _train_elasticnet(
                df,
                feature_fn=_feature_fn,
                feature_names=_feature_names,
                l1_ratios=_l1_ratios,
                repo=repo,
            )
        return _train_logistic(
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
        return _predict_historical_logistic(
            games,
            model_name=model_name,
            model_type=_artifact_type,
            feature_fn=_feature_fn,
            repo=repo,
        )

    def predict_upcoming(
        self: object, schedule: pd.DataFrame, *, repo: Path | None = None
    ) -> pd.DataFrame:
        return _predict_upcoming_logistic(
            schedule,
            model_name=model_name,
            model_type=_artifact_type,
            feature_fn=_feature_fn,
            repo=repo,
        )

    cls = type(
        f"Logistic{registry_key.title().replace('_', '')}Predictor",
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

LogisticPredictor = _make_logistic_variant(
    "logistic",
    "Logistic regression — combined features (32), TimeSeriesSplit CV",
    model_name="win_prob",
    feature_set=FEATURE_SETS["combined"],
)
