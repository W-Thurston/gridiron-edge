"""Shared enums for prop predictions, evaluation, and venue categorization.

Centralises three semantic enumerations that were previously plain strings
scattered across the codebase:

- ``Lean``: Over / Under / No Edge prop bet recommendation.
- ``ConfidenceTier``: High / Moderate / Low confidence classification.
- ``RoofType``: Stadium roof configuration values from the canonical
  games dataset, with two named groupings reflecting two legitimate
  but distinct semantic interpretations (see below).

All three use StrEnum. Each member's value is the canonical string
serialized into DataFrame, archive, and API fields. Internal code should
use enum members where practical and serialize through value at data
boundaries.

Roof type groupings
-------------------
The audit caught a semantic discrepancy between two files that both
produce an "is_dome"-style flag from the ``ROOF`` column:

- ``features/team/weather.py``: previously ``_DOME_ROOF_VALUES =
  {"dome", "retractable"}``. Used to determine whether to override
  OWM weather data with controlled-environment defaults. For this use
  case, the question is "does this stadium have a roof that can keep
  weather out at all?" - so retractable counts, even when the roof
  is open. The new alias is :data:`COVERED_STADIUMS`.

- ``features/player/game_context.py``: previously ``_DOME_ROOFS =
  {"dome", "closed"}``. Used as a player prop feature. For this use
  case, the question is "was the game actually played indoors?" - so
  closed-roof games count, but retractable-open games do not. The
  new alias is :data:`DOME_LIKE_ROOFS`.

Both interpretations are correct for their respective use cases. The
named groupings make the distinction visible.
"""

from __future__ import annotations

from enum import StrEnum


class Lean(StrEnum):
    """Canonical serialized prop recommendation values."""

    OVER = "Over"
    UNDER = "Under"
    NO_EDGE = "No Edge"


class ConfidenceTier(StrEnum):
    """Canonical serialized prediction-confidence values."""

    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


class RoofType(StrEnum):
    """Stadium ROOF column values from the canonical games dataset."""

    DOME = "dome"
    OUTDOORS = "outdoors"
    OPEN = "open"
    CLOSED = "closed"
    RETRACTABLE = "retractable"


# ---------------------------------------------------------------------------
# Roof type groupings
# ---------------------------------------------------------------------------

#: Stadiums where the venue itself provides a controlled environment,
#: regardless of whether the roof was open or closed on game day.
#: Use for OWM weather overrides where the question is whether the
#: physical building shields the field from the elements at all.
COVERED_STADIUMS: frozenset[RoofType] = frozenset({RoofType.DOME, RoofType.RETRACTABLE})

#: Games actually played indoors. Use as a prop model feature where
#: what matters is whether the players experienced indoor conditions,
#: regardless of stadium type.
DOME_LIKE_ROOFS: frozenset[RoofType] = frozenset({RoofType.DOME, RoofType.CLOSED})
