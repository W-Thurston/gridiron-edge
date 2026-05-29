# src/gridiron_edge/models/game_prediction/_shared.py

"""Shared infrastructure for all game-prediction model variants.

This module owns the feature constants, feature engineering functions,
the named FeatureSet registry, and the train/holdout split helper used
by every model family (logistic, tree-based, and future variants).

Nothing in this module imports from sibling model modules — it is a
pure dependency leaf so any model file can import from it safely.

Public API
----------
HOLDOUT_SEASONS         frozenset[str] — seasons reserved for evaluation
_SCHEMA_VERSION         int            — modeling file schema version this module expects
                                         (imported from features.manifest; single source of truth)
_EPA_SUFFIXES           list[str]      — ordered EPA metric suffixes
_RAW_FEATURES           list[str]      — 22 raw feature column names
_DIFF_FEATURES          list[str]      — 10 differential feature column names
_COMBINED_FEATURES      list[str]      — 32 combined feature column names
_GAME_FEATURES          list[str]      — 7 game-level Phase 20e feature names
_TEAM_FEATURES_V2       list[str]      — 12 per-team Phase 20e feature names
_EXPANDED_FEATURES      list[str]      — 51 combined + Phase 20e feature names

FeatureSet              dataclass      — named bundle of (feature_fn, feature_names)
FEATURE_SETS            dict[str, FeatureSet] — registry of named feature sets

_make_diff_features       DataFrame → DataFrame (10 cols)
_make_raw_features        DataFrame → DataFrame (22 cols)
_make_combined_features   DataFrame → DataFrame (32 cols)
_make_expanded_features   DataFrame → DataFrame (51 cols, Phase 20e)
_prepare_data             DataFrame → train/holdout split tuple
_is_trained               str, Path | None → bool
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

# Sourced from manifest.py — single source of truth for schema version.
# Bump CURRENT_SCHEMA_VERSION there and all models pick it up automatically.
from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION as _CURRENT

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLDOUT_SEASONS: Final[frozenset[str]] = frozenset(["2023-2024", "2024-2025", "2025-2026"])

# Schema version this module was designed for.
_SCHEMA_VERSION: Final[int] = _CURRENT

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

# Phase 20e new feature columns — added to schema v3
# Game-level features (same value for both team perspectives in a row)
_GAME_FEATURES: Final[list[str]] = [
    "IS_DIV_GAME",
    "IS_DOME",
    "WIND_SPEED_MPH",
    "TEMP_F",
    "PRECIP_FLAG",
    "IS_NEUTRAL_SITE",
    "ALTITUDE",
]

# Per-team features (asymmetric — TEAM_A and TEAM_B values differ)
_TEAM_FEATURES_V2: Final[list[str]] = [
    "TEAM_A_DAYS_REST",
    "TEAM_B_DAYS_REST",
    "TEAM_A_SHORT_WEEK",
    "TEAM_B_SHORT_WEEK",
    "TEAM_A_POST_BYE",
    "TEAM_B_POST_BYE",
    "TEAM_A_KM_TRAVELED",
    "TEAM_B_KM_TRAVELED",
    "TEAM_A_TZ_SHIFT",
    "TEAM_B_TZ_SHIFT",
    "TEAM_A_FRANCHISE_HFA",
    "TEAM_B_FRANCHISE_HFA",
]

# Expanded feature set (32 combined + 19 Phase 20e = 51 total)
_EXPANDED_FEATURES: Final[list[str]] = _COMBINED_FEATURES + _GAME_FEATURES + _TEAM_FEATURES_V2


# ---------------------------------------------------------------------------
# Named feature sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """A named bundle of feature function + column list.

    Used by the model variant factories in ``logistic.py`` and ``tree.py``
    to declare which features a model variant uses.  Gives each combination
    a stable name so metadata records ``feature_set='combined_32'`` rather
    than a raw function pointer, and factory call sites are self-documenting.

    Attributes:
        name: Short identifier used in artifact metadata (e.g. ``"combined_32"``).
            Also recorded in ``ModelMetadata.parameters["feature_set"]``.
        feature_fn: Callable that takes a modeling DataFrame and returns
            a feature DataFrame with exactly ``feature_names`` columns.
        feature_names: Ordered list of column names produced by ``feature_fn``.
    """

    name: str
    feature_fn: Callable
    feature_names: list[str]


# Populated after function definitions below; referenced here for documentation.
# Callers import FEATURE_SETS["combined"] rather than the raw constants.
FEATURE_SETS: dict[str, FeatureSet]


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


def _make_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all Phase 20e features with the v1 combined set.

    Extends _make_combined_features with the 19 Phase 20e columns:
    game-level features (IS_DIV_GAME, weather, venue) and per-team
    features (rest, travel, franchise HFA).  Game-level features are
    identical for both team perspectives in a row — the model learns
    their influence on win probability directly.

    Missing columns (e.g. WIND_SPEED_MPH for dome games not yet backfilled)
    produce NaN rows which _prepare_data excludes from training automatically.

    Args:
        df: Modeling DataFrame with all schema v3 feature columns.

    Returns:
        DataFrame with 51 expanded features.
    """
    base: DataFrame = _make_combined_features(df)
    phase_20e_cols = [c for c in _GAME_FEATURES + _TEAM_FEATURES_V2 if c in df.columns]
    phase_20e: DataFrame = df.loc[:, phase_20e_cols].copy()
    return pd.concat([base, phase_20e], axis=1)


# Populate FEATURE_SETS now that functions are defined.
FEATURE_SETS = {
    "diff": FeatureSet(
        name="diff_10",
        feature_fn=_make_diff_features,
        feature_names=_DIFF_FEATURES,
    ),
    "raw": FeatureSet(
        name="raw_22",
        feature_fn=_make_raw_features,
        feature_names=list(_RAW_FEATURES),
    ),
    "combined": FeatureSet(
        name="combined_32",
        feature_fn=_make_combined_features,
        feature_names=_COMBINED_FEATURES,
    ),
    "expanded": FeatureSet(
        name="expanded_51",
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
