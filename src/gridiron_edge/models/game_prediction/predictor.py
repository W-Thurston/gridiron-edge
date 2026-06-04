# src/gridiron_edge/models/game_prediction/predictor.py

"""Game prediction model registry entry point.

This module is the single import that callers use to ensure all game
prediction models are registered with PredictorRegistry.  It contains
no model definitions itself — those live in the split module files:

    logistic.py  — logistic (champion)
    tree.py      — random_forest (champion), xgboost (champion)

Importing this module is sufficient to register every model:

    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

Re-exports are provided for backward compatibility with any code that
imports model classes directly from this module.

History: this module previously contained all model class definitions
directly.  It was refactored into split files (logistic.py, tree.py).
The shim pattern preserves backward compatibility
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
    LogisticPredictor,
)
import gridiron_edge.models.game_prediction.tree  # noqa: F401
from gridiron_edge.models.game_prediction.tree import (  # noqa: F401
    RandomForestPredictor,
    XGBoostPredictor,
)
