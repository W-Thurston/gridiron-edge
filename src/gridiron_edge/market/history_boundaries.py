"""Leakage-safe observed boundaries for canonical quote history.

The module selects earliest observed and latest eligible pregame observations.
It does not identify opening or closing lines and does not calculate movement,
CLV, backtest, or recommendation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.ingest.odds.store import (
    OBSERVATION_SORT_COLUMNS,
    validate_quote_rows,
)
from gridiron_edge.market.history_coverage import HISTORY_IDENTITY_COLUMNS


class QuoteBoundaryStatus(StrEnum):
    """Availability of leakage-safe observed quote boundaries."""

    AVAILABLE = "available"
    KICKOFF_UNAVAILABLE = "kickoff_unavailable"
    KICKOFF_CONFLICT = "kickoff_conflict"
    NO_ELIGIBLE_PREGAME_OBSERVATION = "no_eligible_pregame_observation"


@dataclass(frozen=True, slots=True)
class SelectedQuoteObservation:
    """Immutable exact quote observation selected as one boundary."""

    fetched_at: datetime
    sportsbook_updated_at: datetime | None
    commence_time: datetime | None
    is_live: bool
    odds: float | None
    line: float | None


@dataclass(frozen=True, slots=True)
class QuoteHistoryBoundary:
    """Observed boundaries for one exact historical quote identity."""

    status: QuoteBoundaryStatus
    provider: str
    provider_event_id: str | None
    sportsbook: str | None
    game_id: str
    market: str
    side: str
    observation_count: int
    distinct_fetch_count: int
    repeated_observation_evidence_available: bool
    earliest_observed: SelectedQuoteObservation
    latest_eligible_pregame: SelectedQuoteObservation | None


def select_quote_history_boundaries(
    observations: DataFrame,
) -> tuple[QuoteHistoryBoundary, ...]:
    """Select observed boundaries without interpreting market movement."""
    rows = validate_quote_rows(observations)
    if rows.empty:
        return ()

    ordered = rows.sort_values(
        list(OBSERVATION_SORT_COLUMNS),
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    boundaries = [
        _select_identity_boundary(group)
        for _, group in ordered.groupby(
            list(HISTORY_IDENTITY_COLUMNS),
            dropna=False,
            sort=True,
        )
    ]
    return tuple(boundaries)


def _select_identity_boundary(group: DataFrame) -> QuoteHistoryBoundary:
    """Select boundaries for one already grouped historical identity."""
    ordered = group.sort_values(
        list(OBSERVATION_SORT_COLUMNS),
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    first: Series = ordered.iloc[0]
    kickoff_values = ordered["commence_time"].dropna().drop_duplicates()

    status: QuoteBoundaryStatus
    latest: SelectedQuoteObservation | None
    if kickoff_values.empty:
        status = QuoteBoundaryStatus.KICKOFF_UNAVAILABLE
        latest = None
    elif len(kickoff_values) > 1:
        status = QuoteBoundaryStatus.KICKOFF_CONFLICT
        latest = None
    else:
        kickoff = kickoff_values.iloc[0]
        eligible = ordered.loc[
            ordered["is_live"].eq(False) & ordered["fetched_at"].lt(kickoff),
            :,
        ]
        if eligible.empty:
            status = QuoteBoundaryStatus.NO_ELIGIBLE_PREGAME_OBSERVATION
            latest = None
        else:
            status = QuoteBoundaryStatus.AVAILABLE
            latest = _selected_observation(eligible.iloc[-1])

    distinct_fetch_count = int(ordered["fetched_at"].nunique())
    return QuoteHistoryBoundary(
        status=status,
        provider=str(first["provider"]),
        provider_event_id=_nullable_text(first["provider_event_id"]),
        sportsbook=_nullable_text(first["sportsbook"]),
        game_id=str(first["game_id"]),
        market=str(first["market"]),
        side=str(first["side"]),
        observation_count=len(ordered),
        distinct_fetch_count=distinct_fetch_count,
        repeated_observation_evidence_available=distinct_fetch_count > 1,
        earliest_observed=_selected_observation(first),
        latest_eligible_pregame=latest,
    )


def _selected_observation(row: Series) -> SelectedQuoteObservation:
    """Convert one normalized pandas row to an immutable observation."""
    return SelectedQuoteObservation(
        fetched_at=_datetime(row["fetched_at"]),
        sportsbook_updated_at=_nullable_datetime(row["sportsbook_updated_at"]),
        commence_time=_nullable_datetime(row["commence_time"]),
        is_live=bool(row["is_live"]),
        odds=_nullable_float(row["odds"]),
        line=_nullable_float(row["line"]),
    )


def _datetime(value: object) -> datetime:
    """Return one normalized pandas timestamp as a Python datetime."""
    # pyrefly: ignore [bad-argument-type]
    timestamp = pd.Timestamp(value)
    return timestamp.to_pydatetime()


def _nullable_datetime(value: object) -> datetime | None:
    """Return one optional normalized timestamp as a Python datetime."""
    # pyrefly: ignore [no-matching-overload]
    missing = bool(pd.isna(value))
    return None if missing else _datetime(value)


def _nullable_float(value: object) -> float | None:
    """Return one optional normalized numeric quote value."""
    # pyrefly: ignore [no-matching-overload]
    missing = bool(pd.isna(value))
    if missing:
        return None
    # pyrefly: ignore [bad-argument-type]
    return float(value)


def _nullable_text(value: object) -> str | None:
    """Return one optional normalized text identity value."""
    # pyrefly: ignore [no-matching-overload]
    missing = bool(pd.isna(value))
    return None if missing else str(value)
