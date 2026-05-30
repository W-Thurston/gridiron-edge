# src/gridiron_edge/models/game_prediction/_shared.py

"""Re-export shim — content has been split into _columns.py and _features.py.

This module re-exports everything so existing imports continue to work
without modification. Update logistic.py and tree.py to import from the
new modules directly when convenient.

    Pure data / column definitions:  _columns.py
    Feature functions + training:    _features.py
"""

from gridiron_edge.core.constants import HOLDOUT_SEASONS as HOLDOUT_SEASONS
from gridiron_edge.models.game_prediction._columns import (  # noqa: F401
    _COMBINED_FEATURES,
    _DIFF_FEATURES,
    _EPA_SUFFIXES,
    _EXPANDED_FEATURES,
    _GAME_FEATURES,
    _RAW_FEATURES,
    _SCHEMA_VERSION,
    _TEAM_FEATURES_V2,
    FeatureSet,
)
from gridiron_edge.models.game_prediction._features import (  # noqa: F401
    FEATURE_SETS,
    _is_trained,
    _make_combined_features,
    _make_diff_features,
    _make_expanded_features,
    _make_raw_features,
    _prepare_data,
)
