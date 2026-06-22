# src/gridiron_edge/models/game_prediction/_features.py

"""Feature engineering functions, feature set registry, and training helpers.

Imports column definitions from ``_columns.py`` and defines the functions
that operate on DataFrames. ``FEATURE_SETS`` lives here because it must
reference the functions defined in this module.

Public API
----------
FEATURE_SETS        dict[str, FeatureSet]   — registry of named feature sets
_make_diff_features         DataFrame -> DataFrame (24 cols)
_make_raw_features          DataFrame -> DataFrame (47 cols)
_make_combined_features     DataFrame -> DataFrame (70 cols)
_make_expanded_features     DataFrame -> DataFrame (107 cols)
_prepare_data               DataFrame -> train/holdout split tuple
_is_trained                 str, str, Path | None -> bool
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from logging import Logger
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.core.settings import get_settings
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.game_prediction._columns import (
    _COMBINED_FEATURES,
    _DIFF_FEATURES,
    _EPA_SUFFIXES,
    _EXPANDED_FEATURES,
    _GAME_FEATURES,
    _RAW_FEATURES,
    _TEAM_FEATURES,
    FeatureSet,
)

logger: Logger = logging.getLogger(__name__)

# Minimum training rows per CV fold.  TimeSeriesSplit's early folds can
# be too small for high-dimensional feature sets (e.g. 107 features on
# ~1,600 rows).  Folds below this threshold are skipped during HP search.
MIN_CV_TRAIN_ROWS: int = 4000

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
        DataFrame with 19 raw features.
    """
    return df.loc[:, _RAW_FEATURES].copy()


def _make_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine differential and raw features.

    Args:
        df: Modeling DataFrame with raw feature columns.

    Returns:
        DataFrame with 28 combined features.
    """
    diff: DataFrame = _make_diff_features(df)
    raw_no_home: DataFrame = df.loc[:, [c for c in _RAW_FEATURES if c != "HOME_FIELD"]].copy()
    return pd.concat([diff, raw_no_home], axis=1)


def _make_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all features with the v1 combined set.

    Extends _make_combined_features with the 35 columns:
    game-level features (IS_DIV_GAME, weather, venue) and per-team
    features (rest, travel, franchise HFA). Game-level features are
    identical for both team perspectives in a row — the model learns
    their influence on win probability directly.

    Missing columns (e.g. WIND_SPEED_MPH for dome games not yet backfilled)
    produce NaN rows which _prepare_data excludes from training automatically.

    Args:
        df: Modeling DataFrame with all schema v3 feature columns.

    Returns:
        DataFrame with 63 expanded features.
    """
    base: DataFrame = _make_combined_features(df)
    extended_cols: list[str] = [c for c in _GAME_FEATURES + _TEAM_FEATURES if c in df.columns]
    extended: DataFrame = df.loc[:, extended_cols].copy()
    return pd.concat([base, extended], axis=1)


# ---------------------------------------------------------------------------
# Feature set registry
# ---------------------------------------------------------------------------

# Populated here because FEATURE_SETS must reference the functions above.
# Callers import FEATURE_SETS["combined"] rather than the raw constants.
FEATURE_SETS: dict[str, FeatureSet] = {
    "diff": FeatureSet(
        name="diff_24",
        feature_fn=_make_diff_features,
        feature_names=_DIFF_FEATURES,
    ),
    "raw": FeatureSet(
        name="raw_47",
        feature_fn=_make_raw_features,
        feature_names=list(_RAW_FEATURES),
    ),
    "combined": FeatureSet(
        name="combined_70",
        feature_fn=_make_combined_features,
        feature_names=_COMBINED_FEATURES,
    ),
    "expanded": FeatureSet(
        name="expanded_107",
        feature_fn=_make_expanded_features,
        feature_names=_EXPANDED_FEATURES,
    ),
}


# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------


def _prepare_data(
    df: pd.DataFrame,
    feature_fn: Callable,
) -> tuple[pd.DataFrame, Series, pd.DataFrame, Series, list[str], list[str]]:
    """Prepare train/holdout split for a given feature engineering function.

    Excludes:
    - Ties (RESULT == 0.5)
    - Rows with any NaN feature value (covers pre-2006 and week-1 rows)

    Args:
        df: Full modeling DataFrame.
        feature_fn: Function that takes df and returns feature DataFrame.

    Returns:
        Tuple of (x_train, y_train, x_hold, y_hold, train_seasons, hold_seasons).
        Season lists are returned as ``"YYYY-YYYY"`` strings to match the
        :class:`gridiron_edge.models.artifact.BaseModelMetadata` convention.
    """
    df = df.loc[df["RESULT"] != 0.5, :].copy()
    # Sort chronologically so TimeSeriesSplit respects temporal ordering.
    df = df.sort_values(["YEAR", "WEEK_NUM"]).reset_index(drop=True)

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

    train_seasons: list[str] = sorted(df.loc[train_mask, "YEAR"].unique().tolist())
    hold_seasons: list[str] = sorted(df.loc[hold_mask, "YEAR"].unique().tolist())

    return (
        features.loc[train_mask],
        y.loc[train_mask],
        features.loc[hold_mask],
        y.loc[hold_mask],
        train_seasons,
        hold_seasons,
    )


def _is_trained(model_name: str, model_type: str, repo: Path | None) -> bool:
    """Check if a trained artifact exists for the given (model_name, model_type) pair.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        repo: Repository root. Defaults to settings repo root.

    Returns:
        True if an artifact exists, False otherwise.
    """
    resolved_repo: Path = repo or get_settings().repo_root
    return ArtifactStore(resolved_repo).is_trained(model_name, model_type)
