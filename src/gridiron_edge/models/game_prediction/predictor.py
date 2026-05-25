# src/gridiron_edge/models/game_prediction/predictor.py

"""Game prediction models implementing the Predictor + Trainable protocols.

Three logistic regression variants with different feature engineering:

    logistic_v1: Differential features only (10 features)
        TEAM_A - TEAM_B for each metric. Interpretable — positive coefficient
        always means "helps TEAM_A win". Least expressive but most robust.

    logistic_v2: Raw features for both teams (22 features)
        Passes all raw columns directly. The model learns its own
        relationships between team metrics and win probability.

    logistic_v3: Differential + raw combined (32 features)
        Both engineering approaches together. Maximally expressive —
        captures both the matchup differential and absolute team quality.

All variants:
    - Same holdout split as Elo tuning (2023-2026) for fair comparison
    - LogisticRegressionCV with 5-fold CV over 10 regularisation strengths
    - StandardScaler to handle feature scale differences
    - Ties excluded from training (15 games since 1999, ~0.2%)
    - Pre-2006 rows excluded (EPA features unreliable before nflfastR model)
    - NaN rows excluded (incomplete rolling windows)

Adding a new model:
    1. Create a class implementing Predictor + Trainable
    2. Decorate with @PredictorRegistry.register
    3. Implement train(), predict_historical(), predict_upcoming()
    No other changes required.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from gridiron_edge.models.base import PredictorSpec
from gridiron_edge.models.registry import PredictorRegistry

if TYPE_CHECKING:
    from logging import Logger

    from pandas import DataFrame, Series

    from gridiron_edge.models.artifact import ModelMetadata

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLDOUT_SEASONS: Final[frozenset[str]] = frozenset(["2023-2024", "2024-2025", "2025-2026"])

# Schema version this module was designed for
_SCHEMA_VERSION: Final[int] = 2

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

_EPA_SUFFIXES: Final[list[str]] = [
    "OFF_EPA_PER_PLAY",
    "OFF_PASS_EPA",
    "OFF_RUSH_EPA",
    "OFF_SUCCESS_RATE",
    "DEF_EPA_PER_PLAY",
    "DEF_PASS_EPA",
    "DEF_RUSH_EPA",
    "DEF_SUCCESS_RATE",
]

# Raw feature columns (22 total)
_RAW_FEATURES: Final[list[str]] = (
    ["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"]
    + [f"TEAM_A_{s}" for s in _EPA_SUFFIXES]
    + [f"TEAM_B_{s}" for s in _EPA_SUFFIXES]
)

# Differential feature names (10 total)
_DIFF_FEATURES: Final[list[str]] = ["HOME_FIELD", "ELO_DIFF"] + [f"{s}_DIFF" for s in _EPA_SUFFIXES]

# Combined feature names (32 total)
_COMBINED_FEATURES: Final[list[str]] = _DIFF_FEATURES + [
    c for c in _RAW_FEATURES if c != "HOME_FIELD"
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _make_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer TEAM_A - TEAM_B differential features.

    Args:
        df: Modeling DataFrame with raw feature columns.

    Returns:
        DataFrame with 10 differential features.
    """
    out = pd.DataFrame(index=df.index)
    out["HOME_FIELD"] = df["HOME_FIELD"]
    out["ELO_DIFF"] = df["TEAM_A_ELO"] - df["TEAM_B_ELO"]
    for suffix in _EPA_SUFFIXES:
        out[f"{suffix}_DIFF"] = df[f"TEAM_A_{suffix}"] - df[f"TEAM_B_{suffix}"]
    return out


def _make_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select raw features for both teams.

    Args:
        df: Modeling DataFrame with raw feature columns.

    Returns:
        DataFrame with 22 raw features.
    """
    return df.loc[:, _RAW_FEATURES].copy()


def _make_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine differential and raw features.

    Args:
        df: Modeling DataFrame with raw feature columns.

    Returns:
        DataFrame with 32 combined features.
    """
    diff: DataFrame = _make_diff_features(df)
    raw_no_home: DataFrame = df.loc[:, [c for c in _RAW_FEATURES if c != "HOME_FIELD"]].copy()
    return pd.concat([diff, raw_no_home], axis=1)


# ---------------------------------------------------------------------------
# Shared training logic
# ---------------------------------------------------------------------------


def _prepare_data(
    df: pd.DataFrame,
    feature_fn: Callable,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare train/holdout split for a given feature engineering function.

    Excludes:
    - Ties (RESULT == 0.5)
    - Pre-2006 rows (unreliable EPA)
    - Rows with any NaN feature value

    Args:
        df: Full modeling DataFrame.
        feature_fn: Function that takes df and returns feature DataFrame.

    Returns:
        x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons
    """
    # Exclude ties (15 games since 1999, ~0.2% of games)
    df = df.loc[df["RESULT"] != 0.5, :].copy()

    # Note: no explicit pre-2006 filter needed — the NaN mask below
    # already excludes rows where EPA features are missing (all pre-2006
    # rows and week-1 rows without rolling window data).
    features = feature_fn(df)
    valid = features.notna().all(axis=1)
    df = df.loc[valid].copy()
    features = features.loc[valid].copy()

    y = df["RESULT"].astype(int)
    train_mask = ~df["YEAR"].isin(HOLDOUT_SEASONS)
    hold_mask = df["YEAR"].isin(HOLDOUT_SEASONS)

    logger.info(
        "Train: %d rows | Holdout: %d rows",
        train_mask.sum(),
        hold_mask.sum(),
    )

    return (
        features.loc[train_mask],
        y.loc[train_mask],
        features.loc[hold_mask],
        y.loc[hold_mask],
        sorted(df.loc[train_mask, "YEAR"].unique().tolist()),
        sorted(df.loc[hold_mask, "YEAR"].unique().tolist()),
    )


def _train_logistic(
    df: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    feature_names: list[str],
    repo: Path | None,
) -> ModelMetadata:
    """Shared logistic regression training logic for all variants.

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


def _predict_historical_logistic(
    games: pd.DataFrame,
    *,
    model_version: str,
    feature_fn: Callable,
    repo: Path | None,
) -> pd.DataFrame:
    """Shared historical prediction logic for all logistic variants."""
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

    # The modeling file has two rows per game (WINNER-as-TEAM_A and LOSER-as-TEAM_A).
    # Keep only the away-team row (HOME_FIELD == 0) so the archive has exactly
    # one prediction per game, expressed as away_win_prob — consistent with Elo.
    # Neutral site games (both rows have HOME_FIELD=0 after flipping) are handled
    # by deduplicating on game_id and keeping the first occurrence.
    away_rows = df_valid.loc[df_valid["HOME_FIELD"] == 0].copy()

    # Deduplicate neutral-site games (both teams listed as away)
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
    """Shared upcoming prediction logic for all logistic variants."""
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


def _is_trained(model_version: str, repo: Path | None) -> bool:
    """Check if a trained artifact exists."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    return ArtifactStore(resolved_repo).is_trained(model_version)


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
            games, model_version="logistic_v1", feature_fn=_make_diff_features, repo=repo
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v1 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule, model_version="logistic_v1", feature_fn=_make_diff_features, repo=repo
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
            games, model_version="logistic_v2", feature_fn=_make_raw_features, repo=repo
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v2 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule, model_version="logistic_v2", feature_fn=_make_raw_features, repo=repo
        )


# ---------------------------------------------------------------------------
# logistic_v3 — combined differential + raw (32)
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class LogisticV3Predictor:
    """Logistic regression on differential and raw features combined.

    32 features: 10 differentials + 21 raw (HOME_FIELD deduplicated).
    Maximally expressive — captures both matchup differential and
    absolute team quality independently.
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
            games, model_version="logistic_v3", feature_fn=_make_combined_features, repo=repo
        )

    def predict_upcoming(self, schedule: pd.DataFrame, *, repo: Path | None = None) -> pd.DataFrame:
        """Generate logistic_v3 predictions for upcoming games."""
        return _predict_upcoming_logistic(
            schedule, model_version="logistic_v3", feature_fn=_make_combined_features, repo=repo
        )
