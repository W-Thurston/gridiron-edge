# src/gridiron_edge/models/game_prediction/predictor.py

"""Compatibility shim — superseded by the game_prediction package split.

The monolithic predictor.py has been split into:
    _shared.py  — feature engineering infrastructure
    logistic.py — logistic regression variants (v1-v4)
    tree.py     — Random Forest and XGBoost variants

This file exists so that any existing code importing
``gridiron_edge.models.game_prediction.predictor`` continues to work.
All public names are re-exported from their new homes.

New model code should import directly from the relevant sub-module.
"""

from gridiron_edge.models.game_prediction._shared import (  # noqa: F401
    _COMBINED_FEATURES,
    _DIFF_FEATURES,
    _EPA_SUFFIXES,
    _RAW_FEATURES,
    _SCHEMA_VERSION,
    HOLDOUT_SEASONS,
    _is_trained,
    _make_combined_features,
    _make_diff_features,
    _make_raw_features,
    _prepare_data,
)

# pyrefly: ignore [missing-import]
from gridiron_edge.models.game_prediction.logistic import (  # noqa: F401
    LogisticV1Predictor,
    LogisticV2Predictor,
    LogisticV3Predictor,
    LogisticV4Predictor,
    _predict_historical_logistic,
    _predict_upcoming_logistic,
    _train_elasticnet,
    _train_logistic,
)

# pyrefly: ignore [missing-import]
from gridiron_edge.models.game_prediction.tree import (  # noqa: F401
    _EPA_COL_MAP,
    _EPA_RAW_COLS,
    _EPA_WINDOW_OPTIONS,
    RandomForestV1Predictor,
    XGBoostV1Predictor,
    _predict_historical_tree,
    _predict_upcoming_tree,
    _rebuild_features_with_window,
    _train_random_forest,
    _train_xgboost,
)
