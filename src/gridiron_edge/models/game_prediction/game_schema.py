# src/gridiron_edge/models/game_prediction/game_schema.py

"""Canonical home/away schema for game prediction.

The game-prediction domain uses one stable row per game.

Away and home identities retain the same meaning in historical training,
upcoming prediction, forecast events, weekly products, market joins, and
published outputs.

All differential features use the home-team perspective:

    differential = home value - away value

Positive differential values therefore favor the home team.

Win models predict HOME_WIN_PROB directly. AWAY_WIN_PROB is its
complement.

Spread values use the existing home-oriented market convention:

    negative MODEL_SPREAD -> home team favored
    positive MODEL_SPREAD -> away team favored
"""

from __future__ import annotations

from typing import Final

GAME_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

OPTIONAL_GAME_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_DATE",
    "IS_NEUTRAL_SITE",
)

GAME_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_SCORE",
    "HOME_SCORE",
)

GAME_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "HOME_WIN",
    "ACTUAL_MARGIN",
    "ACTUAL_TOTAL",
)

WIN_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_WIN_PROB",
    "HOME_WIN_PROB",
)

GAME_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    *GAME_IDENTITY_COLUMNS,
    *WIN_PROBABILITY_COLUMNS,
    "MODEL_SPREAD",
    "MODEL_TOTAL",
    "PROJECTED_AWAY_SCORE",
    "PROJECTED_HOME_SCORE",
)

AWAY_FEATURE_PREFIX: Final[str] = "AWAY_"
HOME_FEATURE_PREFIX: Final[str] = "HOME_"

HOME_WIN_TARGET: Final[str] = "HOME_WIN"
HOME_WIN_PROBABILITY: Final[str] = "HOME_WIN_PROB"
AWAY_WIN_PROBABILITY: Final[str] = "AWAY_WIN_PROB"

ACTUAL_MARGIN_TARGET: Final[str] = "ACTUAL_MARGIN"
ACTUAL_TOTAL_TARGET: Final[str] = "ACTUAL_TOTAL"

MODEL_SPREAD_COLUMN: Final[str] = "MODEL_SPREAD"
MODEL_TOTAL_COLUMN: Final[str] = "MODEL_TOTAL"

PROJECTED_AWAY_SCORE_COLUMN: Final[str] = "PROJECTED_AWAY_SCORE"
PROJECTED_HOME_SCORE_COLUMN: Final[str] = "PROJECTED_HOME_SCORE"


def home_minus_away_feature_name(base_name: str) -> str:
    """Return the canonical differential feature name.

    Args:
        base_name: Stable feature stem such as ``ELO`` or
            ``OFF_EPA_PER_PLAY``.

    Returns:
        The canonical ``<base_name>_DIFF`` column name.

    Raises:
        ValueError: If ``base_name`` is empty or contains surrounding
            whitespace.
    """
    if not base_name:
        raise ValueError("base_name must not be empty.")

    if base_name != base_name.strip():
        raise ValueError("base_name must not contain surrounding whitespace.")

    return f"{base_name}_DIFF"


def away_feature_name(base_name: str) -> str:
    """Return the canonical away-team feature name."""
    if not base_name:
        raise ValueError("base_name must not be empty.")

    if base_name != base_name.strip():
        raise ValueError("base_name must not contain surrounding whitespace.")

    return f"{AWAY_FEATURE_PREFIX}{base_name}"


def home_feature_name(base_name: str) -> str:
    """Return the canonical home-team feature name."""
    if not base_name:
        raise ValueError("base_name must not be empty.")

    if base_name != base_name.strip():
        raise ValueError("base_name must not contain surrounding whitespace.")

    return f"{HOME_FEATURE_PREFIX}{base_name}"
