# src/gridiron_edge/models/catalog.py

"""Central catalog of model pairs used by full-retrain and manifest writers.

Single source of truth for what pairs get promoted. If a new
``(model_name, model_type)`` pair is added, updating this file
propagates to full-retrain and to the manifest-writing CLI flags.
"""

from __future__ import annotations

GAME_MODEL_PAIRS: list[tuple[str, str]] = [
    ("win_prob", "elo"),
    ("win_prob", "logistic"),
    ("win_prob", "random_forest"),
    ("win_prob", "xgboost"),
    ("total", "random_forest"),
    ("total", "xgboost"),
]

PROP_STAT_FAMILIES: list[str] = [
    "qb_pass_yards",
    "qb_rush_yards",
    "rb_rush_yards",
    "wr_rec_yards",
    "te_rec_yards",
]

PROP_ALGORITHMS: list[str] = ["elasticnet", "random_forest", "xgboost"]
