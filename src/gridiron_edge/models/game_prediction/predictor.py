# src/gridiron_edge/models/game_prediction/predictor.py

"""Game prediction model registry entry point.

This module is the single import that callers use to ensure all game
prediction models are registered with PredictorRegistry.  It contains
no model definitions itself — those live in the split module files:

    logistic.py  — logistic_v1, logistic_v2, logistic_v3, logistic_v4
    tree.py      — random_forest_v1, random_forest_v2,
                   xgboost_v1, xgboost_v2

Importing this module is sufficient to register every model:

    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

Re-exports are provided for backward compatibility with any code that
imports model classes directly from this module.

History: this module previously contained all model class definitions
directly.  It was refactored into split files (logistic.py, tree.py)
during Phase 20d/20e.  The shim pattern preserves backward compatibility
for all callers that import predictor.py as a side-effect registration
trigger or by named attribute.
"""

# Side-effect imports — registers all game prediction models with
# PredictorRegistry.  logistic.py and tree.py use variant factories
# that call PredictorRegistry.register() directly at module load time.

# registration happens as a side effect of the import itself.
import gridiron_edge.models.game_prediction.logistic

# Re-exports for backward compatibility — callers that do
# ``from gridiron_edge.models.game_prediction.predictor import X``
# continue to work without modification.
from gridiron_edge.models.game_prediction.logistic import (  # noqa: F401
    LogisticV1Predictor,
    LogisticV2Predictor,
    LogisticV3Predictor,
    LogisticV4Predictor,
)
import gridiron_edge.models.game_prediction.tree  # noqa: F401
from gridiron_edge.models.game_prediction.tree import (  # noqa: F401
    RandomForestV1Predictor,
    RandomForestV2Predictor,
    XGBoostV1Predictor,
    XGBoostV2Predictor,
)
