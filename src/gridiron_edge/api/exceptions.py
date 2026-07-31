# src/gridiron_edge/api/exceptions.py

"""API-specific exceptions signaling data-state gaps from loaders to routes.

Loaders raise these when the underlying data isn't in a state that can
produce a valid response. Routes catch and translate to structured
field_status metadata per D14.

Distinct from ChampionNotFoundError (which lives in evaluation/) —
these are surface-of-the-API concerns, not domain-model concerns.
"""

from __future__ import annotations


class OddsUnavailableError(Exception):
    """Raised when the current market snapshot is missing or empty.

    Missing market data is an operational data-state gap rather than an API
    failure. Downstream routes catch this exception and mark affected fields
    unavailable instead of returning an HTTP error.
    """
