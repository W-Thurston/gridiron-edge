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

# Opponent-adjusted EPA metric names - single source of truth in
# features.team.opponent_adjusted_epa.
from gridiron_edge.features.team.opponent_adjusted_epa import (
    ADJUSTED_EPA_METRICS as _ADJUSTED_EPA_METRICS_RAW,
)

# Projected-total-plays column name - single source of truth in
# features.team.projected_total_plays.
from gridiron_edge.features.team.projected_total_plays import (
    PROJECTED_TOTAL_PLAYS as _PROJECTED_TOTAL_PLAYS_RAW,
)

# QB profile metric names - single source of truth in features.team.qb_profile.
from gridiron_edge.features.team.qb_profile import QB_PROFILE_METRICS as _QB_PROFILE_METRICS_RAW
from gridiron_edge.features.team.rating_ensemble import (
    RATING_ENSEMBLE_METRICS as _RATING_ENSEMBLE_METRICS_RAW,
)

# Same-season performance metric names (form trend, Pythagorean, one-score
# record) - single source of truth in features.team.record.
from gridiron_edge.features.team.record import (
    SEASON_PERFORMANCE_METRICS as _SEASON_PERFORMANCE_METRICS_RAW,
)

# Unit-matchup efficiency suffixes - single source of truth in
# features.team.unit_efficiency.
from gridiron_edge.features.team.unit_efficiency import (
    UNIT_EFFICIENCY_SUFFIXES as _UNIT_EFFICIENCY_SUFFIXES_RAW,
)

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

# Opponent-adjusted EPA suffixes. These are already-final per-team values
# (not raw per-game stats needing a further rolling average), so they
# follow AWAY_ELO/HOME_ELO/ELO_DIFF's manual-list pattern below rather
# than EPA_COLS' suffix-derivation mechanism - see DECISIONS.md D63.
_OPPONENT_ADJUSTED_EPA_SUFFIXES: Final[list[str]] = [
    metric.upper() for metric in _ADJUSTED_EPA_METRICS_RAW
]

# Unit-matchup efficiency suffixes. Like opponent-adjusted EPA, these are
# already-final values (here, a cross-team combination) rather than raw
# per-game stats, so they follow the same manual-list pattern.
_UNIT_EFFICIENCY_SUFFIXES: Final[list[str]] = list(_UNIT_EFFICIENCY_SUFFIXES_RAW)

# QB and head-coach profile suffixes. Also already-final per-team values
# (a resolved starter's own career/rolling stats; a coach's own career
# win %/tenure), not raw per-game stats, so they follow the same
# manual-list pattern - see DECISIONS.md.
_QB_PROFILE_SUFFIXES: Final[list[str]] = [metric.upper() for metric in _QB_PROFILE_METRICS_RAW]
_COACH_PROFILE_SUFFIXES: Final[list[str]] = [
    metric.upper() for metric in _COACH_PROFILE_METRICS_RAW
]

# Same-season performance suffixes. Already-final per-team values built from
# completed results, so they follow the same manual-list pattern.
_SEASON_PERFORMANCE_SUFFIXES: Final[list[str]] = [
    metric.upper() for metric in _SEASON_PERFORMANCE_METRICS_RAW
]

# Rating-ensemble suffixes (Glicko-2, Massey/Colley/Keener, QB rating). Already-final
# per-team values, so they follow the same manual-list pattern - see DECISIONS.md D69.
_RATING_ENSEMBLE_SUFFIXES: Final[list[str]] = [
    metric.upper() for metric in _RATING_ENSEMBLE_METRICS_RAW
]

# Raw Away/Home feature columns.
_RAW_FEATURES: Final[list[str]] = (
    ["AWAY_ELO", "HOME_ELO"]
    + [f"AWAY_{suffix}" for suffix in _EPA_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _EPA_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _OPPONENT_ADJUSTED_EPA_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _OPPONENT_ADJUSTED_EPA_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _UNIT_EFFICIENCY_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _UNIT_EFFICIENCY_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _COACH_PROFILE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _COACH_PROFILE_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _SEASON_PERFORMANCE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _SEASON_PERFORMANCE_SUFFIXES]
    + [f"AWAY_{suffix}" for suffix in _RATING_ENSEMBLE_SUFFIXES]
    + [f"HOME_{suffix}" for suffix in _RATING_ENSEMBLE_SUFFIXES]
)

# Differential feature names. Every differential is HOME - AWAY.
_DIFF_FEATURES: Final[list[str]] = (
    ["ELO_DIFF"]
    + [f"{suffix}_DIFF" for suffix in _EPA_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _OPPONENT_ADJUSTED_EPA_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _UNIT_EFFICIENCY_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _QB_PROFILE_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _COACH_PROFILE_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _SEASON_PERFORMANCE_SUFFIXES]
    + [f"{suffix}_DIFF" for suffix in _RATING_ENSEMBLE_SUFFIXES]
)

# Differential features followed by direct Away/Home values.
_COMBINED_FEATURES: Final[list[str]] = [
    *_DIFF_FEATURES,
    *_RAW_FEATURES,
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
    "WIND_SPEED_BIN",
    "IS_ARTIFICIAL_TURF",
    "LEAGUE_PPG_TO_DATE",
    "WEEK_NUM",
    _PROJECTED_TOTAL_PLAYS_RAW,
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
    "AWAY_BACK_TO_BACK_ROAD",
    "HOME_BACK_TO_BACK_ROAD",
    "AWAY_BODY_CLOCK_KICKOFF_HOUR",
    "HOME_BODY_CLOCK_KICKOFF_HOUR",
]

_EXPANDED_FEATURES: Final[list[str]] = [
    *_COMBINED_FEATURES,
    *_GAME_FEATURES,
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
