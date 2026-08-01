# src/gridiron_edge/models/game_prediction/_features.py

"""Feature engineering functions, feature set registry, and training helpers.

Imports column definitions from ``_columns.py`` and defines the functions
that operate on DataFrames. ``FEATURE_SETS`` lives here because it must
reference the functions defined in this module.

Public API
----------
FEATURE_SETS        dict[str, FeatureSet]   - registry of named feature sets
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
from typing import cast

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
from gridiron_edge.models.game_prediction.game_schema import (
    HOME_WIN_TARGET,
)

logger: Logger = logging.getLogger(__name__)


#: Default minimum training rows per CV fold.
#:
#: TimeSeriesSplit's early folds can be too small for high-dimensional
#: feature sets (e.g. 107 features on ~1,600 rows). Folds below this
#: threshold are skipped by ``GamesTrainer._cv_score`` during HP search.
#:
#: This is the *default* for champion training (all-history splits where
#: the training pool is ~13k rows and folds 2-5 comfortably clear the
#: guard). Walk-forward backfill trains on much smaller pools and
#: overrides this via ``GamesTrainer.train(min_cv_train_rows=...)`` —
#: see the "walk-forward data sufficiency" contract at the top of
#: ``evaluation/backfill.py``.
#:
#: Backlog: consider scaling this with training-pool size rather than
#: relying on callers to override.
MIN_CV_TRAIN_ROWS: int = 4000


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _make_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer canonical HOME - AWAY differential features."""
    out = pd.DataFrame(index=df.index)
    out["ELO_DIFF"] = df["HOME_ELO"] - df["AWAY_ELO"]
    for suffix in _EPA_SUFFIXES:
        out[f"{suffix}_DIFF"] = df[f"HOME_{suffix}"] - df[f"AWAY_{suffix}"]
    return out.loc[:, _DIFF_FEATURES]


def _make_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select ordered canonical Away and Home raw features."""
    return df.loc[:, _RAW_FEATURES].copy()


def _make_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine canonical differentials and direct Away/Home values."""
    diff: DataFrame = _make_diff_features(df)
    raw: DataFrame = _make_raw_features(df)
    return pd.concat([diff, raw], axis=1).loc[:, _COMBINED_FEATURES]


def _make_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add available canonical game and team-state features."""
    base: DataFrame = _make_combined_features(df)
    extended_columns: list[str] = [
        column for column in [*_GAME_FEATURES, *_TEAM_FEATURES] if column in df.columns
    ]
    extended = df.loc[:, extended_columns].copy()
    return pd.concat([base, extended], axis=1)


# ---------------------------------------------------------------------------
# Feature set registry
# ---------------------------------------------------------------------------

# Populated here because FEATURE_SETS must reference the functions above.
# Callers import FEATURE_SETS["combined"] rather than the raw constants.
FEATURE_SETS: dict[str, FeatureSet] = {
    "diff": FeatureSet(
        name="diff_37",
        feature_fn=_make_diff_features,
        feature_names=_DIFF_FEATURES,
    ),
    "raw": FeatureSet(
        name="raw_74",
        feature_fn=_make_raw_features,
        feature_names=list(_RAW_FEATURES),
    ),
    "combined": FeatureSet(
        name="combined_111",
        feature_fn=_make_combined_features,
        feature_names=_COMBINED_FEATURES,
    ),
    "expanded": FeatureSet(
        name="expanded_152",
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
) -> tuple[
    pd.DataFrame,
    Series,
    pd.DataFrame,
    Series,
    list[str],
    list[str],
]:
    """Prepare canonical Win train and holdout data.

    Tied games have a null ``HOME_WIN`` target and are excluded.
    Rows with any unavailable model feature are also excluded.

    Args:
        df: Canonical one-row-per-game modeling DataFrame.
        feature_fn: Function that returns the model feature matrix.

    Returns:
        Train features, train target, holdout features, holdout target,
        sorted training seasons, and sorted holdout seasons.
    """
    if HOME_WIN_TARGET not in df.columns:
        raise ValueError(
            f"Canonical Win modeling data is missing required target column: {HOME_WIN_TARGET}"
        )

    df = df.dropna(subset=[HOME_WIN_TARGET]).copy()

    df = df.sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
        ignore_index=True,
    )

    features = feature_fn(df)
    valid = features.notna().all(axis=1)
    valid_index = features.index[valid]

    df = df.reindex(valid_index).copy()
    features = features.reindex(valid_index).copy()

    y = df[HOME_WIN_TARGET].astype(int)

    train_mask = ~df["YEAR"].isin(HOLDOUT_SEASONS)
    hold_mask = df["YEAR"].isin(HOLDOUT_SEASONS)

    logger.info(
        "Train: %d rows | Holdout: %d rows",
        train_mask.sum(),
        hold_mask.sum(),
    )

    train_seasons: list[str] = sorted(
        df.loc[
            train_mask,
            "YEAR",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    hold_seasons: list[str] = sorted(
        df.loc[
            hold_mask,
            "YEAR",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    train_index = df.index[train_mask]
    hold_index = df.index[hold_mask]

    x_train = cast(
        DataFrame,
        features.reindex(train_index),
    )
    y_train = cast(
        Series,
        y.reindex(train_index),
    )
    x_hold = cast(
        DataFrame,
        features.reindex(hold_index),
    )
    y_hold = cast(
        Series,
        y.reindex(hold_index),
    )

    return (
        x_train,
        y_train,
        x_hold,
        y_hold,
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
