# src/gridiron_edge/models/game_prediction/_columns.py

"""Feature column definitions and data structures for game-prediction models.

Pure-data leaf — no I/O, no pandas operations, no training logic.
Any module that needs to know column names or the FeatureSet contract
imports from here without pulling in sklearn or training infrastructure.

Public API
----------
_SCHEMA_VERSION     int          — modeling file schema version (from manifest)
_EPA_SUFFIXES       list[str]    — (22) ordered EPA metric suffixes (derived from epa.py)
_RAW_FEATURES       list[str]    — (47) raw feature column names
_DIFF_FEATURES      list[str]    — (24) differential feature column names
_COMBINED_FEATURES  list[str]    — (70) combined feature column names
_GAME_FEATURES      list[str]    — (9) game-level feature names
_TEAM_FEATURES_V2   list[str]    — (28) per-team feature names
_EXPANDED_FEATURES  list[str]    — (107) combined + feature names
FeatureSet          dataclass    — named bundle of (feature_fn, feature_names)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

# Schema version — single source of truth in features.manifest.
from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION as _CURRENT

# EPA metric names — single source of truth in features.team.epa.
from gridiron_edge.features.team.epa import EPA_COLS as _EPA_COLS_RAW

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: Final[int] = _CURRENT

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

# EPA metric suffixes in uppercase — derived from the feature module's
# canonical list so additions to EPA_COLS propagate here automatically.
_EPA_SUFFIXES: Final[list[str]] = [c.upper() for c in _EPA_COLS_RAW]

# Raw feature columns (19 total)
_RAW_FEATURES: Final[list[str]] = (
    ["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"]
    + [f"TEAM_A_{s}" for s in _EPA_SUFFIXES]
    + [f"TEAM_B_{s}" for s in _EPA_SUFFIXES]
)

# Differential feature names (10 total)
_DIFF_FEATURES: Final[list[str]] = ["HOME_FIELD", "ELO_DIFF"] + [f"{s}_DIFF" for s in _EPA_SUFFIXES]

# Combined feature names (28 total)
_COMBINED_FEATURES: Final[list[str]] = _DIFF_FEATURES + [
    c for c in _RAW_FEATURES if c != "HOME_FIELD"
]

# new feature columns — added to schema v3
# Game-level features (same value for both team perspectives in a row)
_GAME_FEATURES: Final[list[str]] = [
    "IS_DIV_GAME",
    "IS_DOME",
    "WIND_SPEED_MPH",
    "TEMP_F",
    "PRECIP_FLAG",
    "FEELS_LIKE_F",
    "HUMIDITY_PCT",
    "VISIBILITY_M",
    "SNOW_FLAG",
    "LOW_VIS_FLAG",
    "WIND_CHILL_DELTA",
    "IS_NEUTRAL_SITE",
    "ALTITUDE",
    "IS_PRIMETIME",
    "WEEK_NUM",
]

# Per-team features (asymmetric — TEAM_A and TEAM_B values differ)
_TEAM_FEATURES_V2: Final[list[str]] = [
    "TEAM_A_DAYS_REST",
    "TEAM_B_DAYS_REST",
    "TEAM_A_SHORT_WEEK",
    "TEAM_B_SHORT_WEEK",
    "TEAM_A_POST_BYE",
    "TEAM_B_POST_BYE",
    "TEAM_A_REST_DIFF",
    "TEAM_B_REST_DIFF",
    "TEAM_A_KM_TRAVELED",
    "TEAM_B_KM_TRAVELED",
    "TEAM_A_TZ_SHIFT",
    "TEAM_B_TZ_SHIFT",
    "TEAM_A_FRANCHISE_HFA",
    "TEAM_B_FRANCHISE_HFA",
    "TEAM_A_WINS",
    "TEAM_A_LOSSES",
    "TEAM_A_WIN_PCT",
    "TEAM_A_WIN_STREAK",
    "TEAM_A_LOSS_STREAK",
    "TEAM_B_WINS",
    "TEAM_B_LOSSES",
    "TEAM_B_WIN_PCT",
    "TEAM_B_WIN_STREAK",
    "TEAM_B_LOSS_STREAK",
    "TEAM_A_SOS",
    "TEAM_A_SOV",
    "TEAM_B_SOS",
    "TEAM_B_SOV",
]

# Expanded feature set (34 combined + 37 = 71 total)
_EXPANDED_FEATURES: Final[list[str]] = _COMBINED_FEATURES + _GAME_FEATURES + _TEAM_FEATURES_V2


# ---------------------------------------------------------------------------
# FeatureSet dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """A named bundle of feature function + column list.

    Used by the model variant factories in ``logistic.py`` and ``tree.py``
    to declare which features a model variant uses. Gives each combination
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
