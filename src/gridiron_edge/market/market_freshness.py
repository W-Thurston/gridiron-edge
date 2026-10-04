# src/gridiron_edge/market/market_freshness.py

"""The one market age limit and the clock every edge boundary uses (``DECISIONS.md`` D80)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

# Provisional: derived by ``gridiron evaluate quote-movement`` (docs/reports/quote-movement.md)
# as the age at which a quote has drifted by more than the sportsbooks disagree among
# themselves. Rerun that command as more weeks are collected and change this value only
# together with a DECISIONS.md entry.
MAX_MARKET_AGE: Final[timedelta] = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class FreshnessInputs:
    """The clock and age limit an edge calculation needs, always supplied together."""

    as_of: datetime
    max_market_age: timedelta


def freshness_inputs(*, now: datetime | None = None) -> FreshnessInputs:
    """Return the ``as_of`` and ``max_market_age`` for an edge calculation.

    The evaluator stays clock-free; each boundary (edge stage, verification, CLI, API)
    supplies the clock through this one function so none of them can forget the limit.
    """
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware UTC.")
    return FreshnessInputs(as_of=moment.astimezone(UTC), max_market_age=MAX_MARKET_AGE)


def format_age(age: timedelta) -> str:
    """Render a quote age for people: minutes, hours or days."""
    seconds = max(age.total_seconds(), 0.0)
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def market_age_label(fetched_at: datetime | None, *, now: datetime | None = None) -> str:
    """Describe a snapshot's age against the limit, e.g. ``3.2 days old, over the 24 h limit``."""
    if fetched_at is None:
        return "unknown"
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    age = moment - fetched_at.astimezone(UTC)
    limit_hours = MAX_MARKET_AGE.total_seconds() / 3600
    state = "over" if age > MAX_MARKET_AGE else "within"
    return f"{format_age(age)} old, {state} the {limit_hours:g} h limit"
