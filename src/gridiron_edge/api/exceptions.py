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
    """Raised when the current odds snapshot is missing or empty.

    Not a bug: the odds snapshot is written by ``gridiron ingest fetch-odds``
    and may not have been refreshed yet. Downstream routes catch this
    and mark the affected fields with an Unavailable slug rather than
    returning an HTTP error.
    """
