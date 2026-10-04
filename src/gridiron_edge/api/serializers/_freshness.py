# src/gridiron_edge/api/serializers/_freshness.py
"""Serialize the market freshness context."""

from __future__ import annotations

from datetime import datetime

from gridiron_edge.api.schemas.freshness import MarketFreshness
from gridiron_edge.market.market_freshness import MAX_MARKET_AGE


def market_freshness(evaluated_at: datetime) -> MarketFreshness:
    """Return the server clock and the configured age limit."""
    return MarketFreshness(
        evaluated_at=evaluated_at,
        max_age_seconds=int(MAX_MARKET_AGE.total_seconds()),
    )
