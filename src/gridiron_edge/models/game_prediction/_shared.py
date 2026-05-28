# src/gridiron_edge/models/game_prediction/_shared.py

"""Shared infrastructure for all game-prediction model variants.

This module owns the feature constants, feature engineering functions,
and the train/holdout split helper used by every model family
(logistic, tree-based, and future variants).

Nothing in this module imports from sibling model modules — it is a
pure dependency leaf so any model file can import from it safely.

Public API
----------
HOLDOUT_SEASONS       frozenset[str] — seasons reserved for evaluation
_SCHEMA_VERSION       int            — modeling file schema version this module expects
_EPA_SUFFIXES         list[str]      — ordered EPA metric suffixes
_RAW_FEATURES         list[str]      — 22 raw feature column names
_DIFF_FEATURES        list[str]      — 10 differential feature column names
_COMBINED_FEATURES    list[str]      — 32 combined feature column names

_make_diff_features     DataFrame → DataFrame (10 cols)
_make_raw_features      DataFrame → DataFrame (22 cols)
_make_combined_features DataFrame → DataFrame (32 cols)
_prepare_data           DataFrame → train/holdout split tuple
_is_trained             str, Path | None → bool
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame, Series

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """
    df = df.loc[df["RESULT"] != 0.5, :].copy()

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


def _is_trained(model_version: str, repo: Path | None) -> bool:
    """Check if a trained artifact exists for the given model version.

    Args:
        model_version: Registered model version string.
        repo: Repository root. Defaults to settings repo root.

    Returns:
        True if an artifact exists, False otherwise.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    return ArtifactStore(resolved_repo).is_trained(model_version)
