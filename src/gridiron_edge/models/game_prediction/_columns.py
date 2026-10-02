# src/gridiron_edge/models/game_prediction/_columns.py

"""Feature column definitions and data structures for game-prediction models.

Pure-data leaf - no I/O, no pandas operations, no training logic.
Any module that needs to know column names or the FeatureSet contract
imports from here without pulling in sklearn or training infrastructure.

Public API
----------
_SCHEMA_VERSION     int          - modeling file schema version (from manifest)
_EPA_SUFFIXES       list[str]    - ordered EPA metric suffixes (derived from epa.py)
_RAW_FEATURES       list[str]    - raw feature column names
_DIFF_FEATURES      list[str]    - differential feature column names
_COMBINED_FEATURES  list[str]    - combined feature column names
_GAME_FEATURES      list[str]    - game-level feature names
_TEAM_FEATURES      list[str]    - per-team feature names
_EXPANDED_FEATURES  list[str]    - combined + feature names
FeatureSet          dataclass    - named bundle of (feature_fn, feature_names)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

# Schema version - single source of truth in features.manifest.
from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION as _CURRENT

# Head-coach profile metric names - single source of truth in
# features.team.coach_profile.
from gridiron_edge.features.team.coach_profile import (
    COACH_PROFILE_METRICS as _COACH_PROFILE_METRICS_RAW,
)

# EPA metric names - single source of truth in features.team.epa.
from gridiron_edge.features.team.epa import EPA_COLS as _EPA_COLS_RAW

# QB profile metric names - single source of truth in features.team.qb_profile.
from gridiron_edge.features.team.qb_profile import QB_PROFILE_METRICS as _QB_PROFILE_METRICS_RAW

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: Final[int] = _CURRENT

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

# EPA metric suffixes in uppercase - derived from the feature module's
# canonical list so additions to EPA_COLS propagate here automatically.
_EPA_SUFFIXES: Final[list[str]] = [c.upper() for c in _EPA_COLS_RAW]

# QB and head-coach profile suffixes. Also already-final per-team values
# (a resolved starter's own career/rolling stats; a coach's own career
# win %/tenure), not raw per-game stats, so they follow the same
# manual-list pattern - see DECISIONS.md.
_QB_PROFILE_SUFFIXES: Final[list[str]] = [metric.upper() for metric in _QB_PROFILE_METRICS_RAW]
_COACH_PROFILE_SUFFIXES: Final[list[str]] = [
    metric.upper() for metric in _COACH_PROFILE_METRICS_RAW
]

# Raw Away/Home feature columns.
_RAW_FEATURES: Final[list[str]] = (
    ["AWAY_ELO", "HOME_ELO"]
    + [f"AWAY_{suffix}" for suffix in _EPA_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _EPA_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _COACH_PROFILE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _COACH_PROFILE_SUFFIXES]
)

# Differential feature names. Every differential is HOME - AWAY.
_DIFF_FEATURES: Final[list[str]] = (
    ["ELO_DIFF"]
    + [f"{suffix}_DIFF" for suffix in _EPA_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _COACH_PROFILE_SUFFIXES]
)

# Neutral-site flag. The Logistic Win model removes its home-field term at
# neutral sites through this column (D74).
_NEUTRAL_SITE_FEATURE: Final[str] = "IS_NEUTRAL_SITE"

# Differential features, direct Away/Home values, then the neutral-site flag.
_COMBINED_FEATURES: Final[list[str]] = [
    *_DIFF_FEATURES,
    *_RAW_FEATURES,
    _NEUTRAL_SITE_FEATURE,
]

# Matchup-level features with one value per game.
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
    "GAME_SITE_ALTITUDE",
    "IS_PRIMETIME",
    "WEEK_NUM",
]

# Team-state features produced directly by the canonical feature pipeline.
_TEAM_FEATURES: Final[list[str]] = [
    "AWAY_DAYS_REST",
    "HOME_DAYS_REST",
    "AWAY_SHORT_WEEK",
    "HOME_SHORT_WEEK",
    "AWAY_POST_BYE",
    "HOME_POST_BYE",
    "DAYS_REST_DIFF",
    "AWAY_KM_TRAVELED",
    "HOME_KM_TRAVELED",
    "AWAY_TZ_SHIFT",
    "HOME_TZ_SHIFT",
    "HOME_FRANCHISE_HFA",
    "AWAY_WINS",
    "AWAY_LOSSES",
    "AWAY_WIN_PCT",
    "AWAY_WIN_STREAK",
    "AWAY_LOSS_STREAK",
    "HOME_WINS",
    "HOME_LOSSES",
    "HOME_WIN_PCT",
    "HOME_WIN_STREAK",
    "HOME_LOSS_STREAK",
    "AWAY_SOS",
    "AWAY_SOV",
    "HOME_SOS",
    "HOME_SOV",
    "AWAY_QB_CHANGE_FLAG",
    "HOME_QB_CHANGE_FLAG",
    "AWAY_NEW_COACHING_STAFF_FLAG",
    "HOME_NEW_COACHING_STAFF_FLAG",
]

_EXPANDED_FEATURES: Final[list[str]] = [
    *_COMBINED_FEATURES,
    *[column for column in _GAME_FEATURES if column not in _COMBINED_FEATURES],
    *_TEAM_FEATURES,
]


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
