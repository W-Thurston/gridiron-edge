# src/gridiron_edge/models/game_prediction/__init__.py

"""Game prediction model package.

Importing this package registers all prediction models with
ModelRegistry.  Any code that needs to enumerate or instantiate
game-prediction models should import this package (or any sub-module)
before calling ModelRegistry.names().

Sub-modules:
    _columns.py  - feature column definitions
    _features.py - feature engineering + training helpers
    logistic  - logistic (champion)
    tree      - random_forest, xgboost (champions)

The old monolithic predictor.py is superseded by this package.  The
import path ``gridiron_edge.models.game_prediction.predictor`` is kept
as a compatibility shim (it simply re-imports this package) so any
existing code or CLI imports continue to work unchanged.
"""
