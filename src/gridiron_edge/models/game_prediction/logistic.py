# src/gridiron_edge/models/game_prediction/logistic.py

"""Logistic regression game prediction models.

Four variants with different feature engineering and regularisation:

    logistic_v1: Differential features only (10 features)
        TEAM_A - TEAM_B for each metric. Interpretable — positive coefficient
        always means "helps TEAM_A win". Least expressive but most robust.

    logistic_v2: Raw features for both teams (22 features)
        Passes all raw columns directly. The model learns its own
        relationships between team metrics and win probability.

    logistic_v3: Differential + raw combined (32 features)
        Both engineering approaches together. Maximally expressive —
        captures both the matchup differential and absolute team quality.
        Current best model: Brier 0.22057, AUC 0.68289.

    logistic_v4: Elastic net regularisation on combined features (32 features)
        Same feature set as v3 but with elastic net (L1+L2) regularisation
        via the SAGA solver. Tunes both regularisation strength C and L1/L2
        mix ratio. L1 component drives irrelevant features to exactly zero.

All variants:
    - Same holdout split (2023-2026) for fair comparison with tree models
    - LogisticRegressionCV with 5-fold CV over 10 regularisation strengths
    - StandardScaler in sklearn Pipeline
    - Ties excluded (15 games since 1999, ~0.2%)
    - NaN rows excluded (covers pre-2006 and incomplete rolling windows)
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.models.base import PredictorSpec
from gridiron_edge.models.game_prediction._shared import (
    _COMBINED_FEATURES,
    _DIFF_FEATURES,
    _RAW_FEATURES,
    _SCHEMA_VERSION,
    _is_trained,
    _make_combined_features,
    _make_diff_features,
    _make_raw_features,
    _prepare_data,
)
from gridiron_edge.models.registry import PredictorRegistry

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import ModelMetadata

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared logistic training logic
# ---------------------------------------------------------------------------


def _train_logistic(
    df: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> ModelMetadata:
    """Shared logistic regression training logic for all variants.

    Uses LogisticRegressionCV with 5-fold CV over 10 regularisation
    strengths (Cs), scored by neg_brier_score.

    Args:
        df: Full modeling DataFrame.
        model_version: Model version string for artifact naming.
        feature_fn: Feature engineering function.
        feature_names: List of feature column names produced by feature_fn.
        repo: Repository root.

    Returns:
        ModelMetadata with holdout Brier and training details.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegressionCV

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

    x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = _prepare_data(df, feature_fn)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=5,
                    scoring="neg_brier_score",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Fitting %s on %d training rows...", model_version, len(x_train))
    pipeline.fit(x_train, y_train)

    hold_probs: Series = pd.Series(pipeline.predict_proba(x_hold)[:, 1], index=x_hold.index)
    holdout_brier: float = brier_score(hold_probs, y_hold.astype(float))

    train_probs: Series = pd.Series(pipeline.predict_proba(x_train)[:, 1], index=x_train.index)
    train_brier: float = brier_score(train_probs, y_train.astype(float))
    best_c = float(pipeline.named_steps["clf"].C_[0])

    logger.info(
        "%s: train=%.5f  holdout=%.5f  C=%.4f",
        model_version,
        train_brier,
        holdout_brier,
        best_c,
    )

    metadata = ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        holdout_brier=round(holdout_brier, 6),
        parameters={
            "best_C": best_c,
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "n_train": len(x_train),
            "n_holdout": len(x_hold),
            "n_features": len(feature_names),
            "feature_set": model_version,
            "epa_window": 4,
            "cv_folds": 5,
        },
        feature_columns=feature_names,
    )

    store.save(model_version, pipeline, metadata=metadata)
    return metadata


def _train_elasticnet(
    df: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    feature_names: list[str],
    l1_ratios: list[float],
    repo: Path | None,
) -> ModelMetadata:
    """Train elastic net logistic regression and save artifact.

    Uses the SAGA solver which supports elastic net penalties. Tunes
    both regularisation strength C and L1/L2 mix ratio via CV.

    Args:
        df: Full modeling DataFrame.
        model_version: Model version string for artifact naming.
        feature_fn: Feature engineering function.
        feature_names: Feature column names produced by feature_fn.
        l1_ratios: L1/L2 mix ratios to search (0.0=ridge, 1.0=lasso).
        repo: Repository root.

    Returns:
        ModelMetadata with holdout Brier, best C, and best l1_ratio.
    """
    from datetime import UTC, datetime

    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegressionCV

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

    x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons = _prepare_data(df, feature_fn)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=5,
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

    logger.info("Fitting %s (elastic net) on %d training rows...", model_version, len(x_train))
    pipeline.fit(x_train, y_train)

    clf = pipeline.named_steps["clf"]
    best_c = float(clf.C_[0])
    best_l1_ratio = float(clf.l1_ratio_[0])

    hold_probs: Series = pd.Series(pipeline.predict_proba(x_hold)[:, 1], index=x_hold.index)
    holdout_brier: float = brier_score(hold_probs, y_hold.astype(float))

    train_probs: Series = pd.Series(pipeline.predict_proba(x_train)[:, 1], index=x_train.index)
    train_brier: float = brier_score(train_probs, y_train.astype(float))

    logger.info(
        "%s: train=%.5f  holdout=%.5f  C=%.4f  l1_ratio=%.2f",
        model_version,
        train_brier,
        holdout_brier,
        best_c,
        best_l1_ratio,
    )

    coefs = clf.coef_[0]
    nonzero_features = [f for f, c in zip(feature_names, coefs, strict=True) if abs(c) > 1e-6]

    metadata = ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=CURRENT_SCHEMA_VERSION,
        training_seasons=train_seasons,
        holdout_seasons=hold_seasons,
        holdout_brier=round(holdout_brier, 6),
        parameters={
            "best_C": best_c,
            "best_l1_ratio": best_l1_ratio,
            "l1_ratios_searched": l1_ratios,
            "train_brier": round(train_brier, 6),
            "overfit_gap": round(holdout_brier - train_brier, 6),
            "n_train": len(x_train),
            "n_holdout": len(x_hold),
            "n_features": len(feature_names),
            "n_nonzero_features": len(nonzero_features),
            "nonzero_features": nonzero_features,
            "feature_set": model_version,
            "epa_window": 4,
            "cv_folds": 5,
        },
        feature_columns=feature_names,
    )

    store.save(model_version, pipeline, metadata=metadata)
    return metadata


# ---------------------------------------------------------------------------
# Shared prediction helpers
# ---------------------------------------------------------------------------


def _predict_historical_logistic(
    games: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared historical prediction logic for all logistic variants.

    Args:
        games: Games DataFrame (unused — full modeling file loaded from disk).
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
    df: DataFrame = load_modeling_file(resolved_repo, required_schema_version=_SCHEMA_VERSION)

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


def _predict_upcoming_logistic(
    schedule: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared upcoming prediction logic for all logistic variants.

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
# logistic_v1 — differential features (10)
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class LogisticV1Predictor:
    """Logistic regression on TEAM_A - TEAM_B differential features.

    10 features: HOME_FIELD, ELO_DIFF, 8 EPA differentials.
    Interpretable: positive coefficient always means "helps TEAM_A win".
    """

    spec = PredictorSpec(
        name="logistic_v1",
        description="Logistic regression — differential features (10)",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train on differential features."""
        return _train_logistic(
            df,
            model_version="logistic_v1",
            feature_fn=_make_diff_features,
            feature_names=_DIFF_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for logistic_v1."""
        return _is_trained("logistic_v1", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v1 predictions for all historical games."""
        return _predict_historical_logistic(
            games,
            model_version="logistic_v1",
            feature_fn=_make_diff_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v1 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule,
            model_version="logistic_v1",
            feature_fn=_make_diff_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# logistic_v2 — raw features for both teams (22)
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class LogisticV2Predictor:
    """Logistic regression on raw features for both teams.

    22 features: HOME_FIELD, TEAM_A_ELO, TEAM_B_ELO, 8 EPA cols per team.
    The model learns team relationships directly without differential engineering.
    """

    spec = PredictorSpec(
        name="logistic_v2",
        description="Logistic regression — raw features both teams (22)",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train on raw features."""
        return _train_logistic(
            df,
            model_version="logistic_v2",
            feature_fn=_make_raw_features,
            feature_names=_RAW_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for logistic_v2."""
        return _is_trained("logistic_v2", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v2 predictions for all historical games."""
        return _predict_historical_logistic(
            games,
            model_version="logistic_v2",
            feature_fn=_make_raw_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v2 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule,
            model_version="logistic_v2",
            feature_fn=_make_raw_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# logistic_v3 — combined differential + raw (32)
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class LogisticV3Predictor:
    """Logistic regression on differential and raw features combined.

    32 features: 10 differentials + 21 raw (HOME_FIELD deduplicated).
    Maximally expressive for a linear model — captures both matchup
    differential and absolute team quality independently.
    Current best: Brier 0.22057, AUC 0.68289.
    """

    spec = PredictorSpec(
        name="logistic_v3",
        description="Logistic regression — differential + raw combined (32)",
        trainable=True,
    )

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train on combined features."""
        return _train_logistic(
            df,
            model_version="logistic_v3",
            feature_fn=_make_combined_features,
            feature_names=_COMBINED_FEATURES,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for logistic_v3."""
        return _is_trained("logistic_v3", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v3 predictions for all historical games."""
        return _predict_historical_logistic(
            games,
            model_version="logistic_v3",
            feature_fn=_make_combined_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v3 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule,
            model_version="logistic_v3",
            feature_fn=_make_combined_features,
            repo=repo,
        )


# ---------------------------------------------------------------------------
# logistic_v4 — elastic net regularisation on combined features (32)
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class LogisticV4Predictor:
    """Logistic regression with elastic net regularisation.

    Same 32-feature combined set as logistic_v3 but uses elastic net
    (L1 + L2) regularisation via the SAGA solver. Tunes both:
        C:         regularisation strength (10 values, log-spaced)
        l1_ratio:  mix between L1 (feature selection) and L2 (shrinkage)

    L1 component drives irrelevant feature coefficients exactly to zero.
    """

    spec = PredictorSpec(
        name="logistic_v4",
        description="Logistic regression — elastic net, combined features (32)",
        trainable=True,
    )

    L1_RATIOS: Final[list[float]] = [0.1, 0.3, 0.5, 0.7, 0.9]

    def train(self, df: pd.DataFrame, *, repo: Path | None = None) -> ModelMetadata:
        """Train elastic net logistic regression on combined features."""
        return _train_elasticnet(
            df,
            model_version="logistic_v4",
            feature_fn=_make_combined_features,
            feature_names=_COMBINED_FEATURES,
            l1_ratios=self.L1_RATIOS,
            repo=repo,
        )

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for logistic_v4."""
        return _is_trained("logistic_v4", repo)

    def predict_historical(self, games: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v4 predictions for all historical games."""
        return _predict_historical_logistic(
            games,
            model_version="logistic_v4",
            feature_fn=_make_combined_features,
            repo=repo,
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v4 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule,
            model_version="logistic_v4",
            feature_fn=_make_combined_features,
            repo=repo,
        )
