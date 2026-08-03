# src/gridiron_edge/core/constants.py

"""Shared schema constants used across the feature, transform, and model layers.

Centralising sentinel values here means a single edit when any convention
is retired (e.g. replacing ``"NULL_VALUE"`` with ``"H"`` for home games).
"""

# ---------------------------------------------------------------------------
# Holdout seasons
# ---------------------------------------------------------------------------

# Seasons reserved for model evaluation - excluded from training.
# UPDATE: add the new season label here at the start of each season so that
# new games are held out of training and reserved for live evaluation.
# Both the model layer (_shared.py) and the Elo tuner (tune.py) import this.
HOLDOUT_SEASONS: frozenset[str] = frozenset(["2023-2024", "2024-2025", "2025-2026"])

# ---------------------------------------------------------------------------
# Expansion franchise start seasons
# ---------------------------------------------------------------------------

# Maps franchise long name → first season label (YYYY-YYYY+1).
# Used by the Elo engine to assign expansion_elo instead of initial_elo
# in a team's first season. Single source of truth for both tune.py and
# ratings/elo/table.py.
EXPANSION_TEAMS: dict[str, str] = {
    "Carolina Panthers": "1995-1996",
    "Jacksonville Jaguars": "1995-1996",
    "Baltimore Ravens": "1996-1997",
    "Houston Texans": "2002-2003",
}

# ---------------------------------------------------------------------------
# Historical team code normalization
# ---------------------------------------------------------------------------

# Maps relocated/renamed team abbreviations to their current short codes.
# nflverse and PFR use era-appropriate abbreviations (OAK for pre-2020
# Raiders, SD for pre-2017 Chargers, STL for pre-2016 Rams, JAC pre-2013
# Jaguars). This map normalizes them to current short codes for joins
# against current team-keyed data.
TEAM_CODE_NORMALIZATION: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX",
}
