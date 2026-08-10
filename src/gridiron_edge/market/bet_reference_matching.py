"""Pure matching of bet reference provenance to exact quote observations.

The matcher verifies recorded reference evidence only. It does not select
closeout observations or calculate movement, CLV, qualification, or
recommendation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TypedDict

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.ingest.odds.store import validate_quote_rows
from gridiron_edge.market.history_boundaries import SelectedQuoteObservation

_REQUIRED_BET_COLUMNS: tuple[str, ...] = (
    "bet_id",
    "game_id",
    "market_type",
    "side",
    "reference_provider",
    "reference_provider_event_id",
    "reference_sportsbook",
    "reference_market_fetched_at",
    "reference_sportsbook_updated_at",
    "reference_commence_time",
    "reference_american_odds",
    "reference_line",
)
_REFERENCE_COLUMNS: tuple[str, ...] = (
    "reference_provider",
    "reference_provider_event_id",
    "reference_sportsbook",
    "reference_market_fetched_at",
    "reference_sportsbook_updated_at",
    "reference_commence_time",
    "reference_american_odds",
    "reference_line",
)
_REFERENCE_TERM_FIELDS: tuple[str, ...] = (
    "reference_sportsbook_updated_at",
    "reference_commence_time",
    "reference_american_odds",
    "reference_line",
)


class _BetReferenceIdentity(TypedDict):
    """Typed identity shared by reference-match result branches."""

    bet_id: str
    provider: str | None
    provider_event_id: str | None
    sportsbook: str | None
    game_id: str
    market: str
    side: str
    reference_fetched_at: datetime | None


class BetReferenceMatchStatus(StrEnum):
    """Resolution of one recorded bet reference against quote history."""

    MATCHED = "matched"
    MANUAL_BET = "manual_bet"
    OBSERVATION_NOT_FOUND = "observation_not_found"
    AMBIGUOUS_OBSERVATION = "ambiguous_observation"
    REFERENCE_TERMS_CONFLICT = "reference_terms_conflict"


@dataclass(frozen=True, slots=True)
class BetReferenceMatch:
    """Diagnostic match between one bet reference and quote history."""

    bet_id: str
    status: BetReferenceMatchStatus
    provider: str | None
    provider_event_id: str | None
    sportsbook: str | None
    game_id: str
    market: str
    side: str
    reference_fetched_at: datetime | None
    matched_observation: SelectedQuoteObservation | None
    mismatched_fields: tuple[str, ...]


def match_bet_references(
    bets: DataFrame,
    observations: DataFrame,
) -> tuple[BetReferenceMatch, ...]:
    """Match immutable bet references to exact quote observations."""
    bet_rows = _validate_bets(bets)
    quote_rows = validate_quote_rows(observations)
    if bet_rows.empty:
        return ()

    matches = [
        _match_bet(row, quote_rows)
        for _, row in bet_rows.sort_values("bet_id", kind="stable").iterrows()
    ]
    return tuple(matches)


def _validate_bets(bets: DataFrame) -> DataFrame:
    """Validate only the bet fields required by the pure matcher."""
    missing = sorted(set(_REQUIRED_BET_COLUMNS) - set(bets.columns))
    if missing:
        raise ValueError("Bet reference matching requires columns: " + ", ".join(missing))

    rows = bets.loc[:, _REQUIRED_BET_COLUMNS].copy()
    if rows.empty:
        return rows

    for column in ("bet_id", "game_id", "market_type", "side"):
        values = rows[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Bet column {column!r} must contain nonempty values.")
        rows[column] = values.astype("string").str.strip()

    if rows["bet_id"].duplicated().any():
        raise ValueError("Bet reference matching requires unique bet_id values.")

    for index, row in rows.iterrows():
        _validate_bet_reference(row, index=index)
    return rows


def _validate_bet_reference(row: Series, *, index: object) -> None:
    """Validate one optional reference root and its timestamps."""
    if all(_is_missing(row[column]) for column in _REFERENCE_COLUMNS):
        return

    provider = _nullable_text(row["reference_provider"])
    if provider is None or not provider.strip():
        raise ValueError(f"Bet row {index!r} requires a nonempty reference_provider.")
    fetched_at = _nullable_utc_datetime(row["reference_market_fetched_at"])
    if fetched_at is None:
        raise ValueError(f"Bet row {index!r} requires reference_market_fetched_at.")

    for column in ("reference_provider_event_id", "reference_sportsbook"):
        value = _nullable_text(row[column])
        if value is not None and not value.strip():
            raise ValueError(f"Bet row {index!r} has empty {column}.")

    for column in (
        "reference_sportsbook_updated_at",
        "reference_commence_time",
    ):
        _nullable_utc_datetime(row[column])


def _match_bet(row: Series, observations: DataFrame) -> BetReferenceMatch:
    """Resolve one validated bet against canonical observations."""
    bet_id = str(row["bet_id"])
    game_id = str(row["game_id"])
    market = str(row["market_type"])
    side = str(row["side"])

    if all(_is_missing(row[column]) for column in _REFERENCE_COLUMNS):
        return BetReferenceMatch(
            bet_id=bet_id,
            status=BetReferenceMatchStatus.MANUAL_BET,
            provider=None,
            provider_event_id=None,
            sportsbook=None,
            game_id=game_id,
            market=market,
            side=side,
            reference_fetched_at=None,
            matched_observation=None,
            mismatched_fields=(),
        )

    provider = _nullable_text(row["reference_provider"])
    provider_event_id = _nullable_text(row["reference_provider_event_id"])
    sportsbook = _nullable_text(row["reference_sportsbook"])
    fetched_at = _nullable_utc_datetime(row["reference_market_fetched_at"])
    assert provider is not None
    assert fetched_at is not None

    candidates = observations.loc[
        observations["provider"].eq(provider)
        & _nullable_identity_mask(observations["provider_event_id"], provider_event_id)
        & _nullable_identity_mask(observations["sportsbook"], sportsbook)
        & observations["game_id"].eq(game_id)
        & observations["market"].eq(market)
        & observations["side"].eq(side)
        & observations["fetched_at"].eq(pd.Timestamp(fetched_at)),
        :,
    ]

    base: _BetReferenceIdentity = {
        "bet_id": bet_id,
        "provider": provider,
        "provider_event_id": provider_event_id,
        "sportsbook": sportsbook,
        "game_id": game_id,
        "market": market,
        "side": side,
        "reference_fetched_at": fetched_at,
    }
    if candidates.empty:
        return BetReferenceMatch(
            status=BetReferenceMatchStatus.OBSERVATION_NOT_FOUND,
            matched_observation=None,
            mismatched_fields=(),
            **base,
        )
    if len(candidates) > 1:
        return BetReferenceMatch(
            status=BetReferenceMatchStatus.AMBIGUOUS_OBSERVATION,
            matched_observation=None,
            mismatched_fields=(),
            **base,
        )

    candidate: Series = candidates.iloc[0]
    mismatched = _mismatched_reference_terms(row, candidate)
    if mismatched:
        return BetReferenceMatch(
            status=BetReferenceMatchStatus.REFERENCE_TERMS_CONFLICT,
            matched_observation=None,
            mismatched_fields=mismatched,
            **base,
        )
    return BetReferenceMatch(
        status=BetReferenceMatchStatus.MATCHED,
        matched_observation=_selected_observation(candidate),
        mismatched_fields=(),
        **base,
    )


def _mismatched_reference_terms(
    bet: Series,
    observation: Series,
) -> tuple[str, ...]:
    """Return conflicting immutable reference terms in canonical order."""
    comparisons = (
        (
            "reference_sportsbook_updated_at",
            _nullable_utc_datetime(bet["reference_sportsbook_updated_at"]),
            _nullable_utc_datetime(observation["sportsbook_updated_at"]),
        ),
        (
            "reference_commence_time",
            _nullable_utc_datetime(bet["reference_commence_time"]),
            _nullable_utc_datetime(observation["commence_time"]),
        ),
        (
            "reference_american_odds",
            _nullable_float(bet["reference_american_odds"]),
            _nullable_float(observation["odds"]),
        ),
        (
            "reference_line",
            _nullable_float(bet["reference_line"]),
            _nullable_float(observation["line"]),
        ),
    )
    return tuple(field for field, expected, actual in comparisons if expected != actual)


def _nullable_identity_mask(values: Series, expected: str | None) -> Series:
    """Return exact null-aware identity equality."""
    return values.isna() if expected is None else values.eq(expected)


def _selected_observation(row: Series) -> SelectedQuoteObservation:
    """Convert one canonical quote row to immutable matched evidence."""
    fetched_at = _nullable_utc_datetime(row["fetched_at"])
    assert fetched_at is not None
    return SelectedQuoteObservation(
        fetched_at=fetched_at,
        sportsbook_updated_at=_nullable_utc_datetime(row["sportsbook_updated_at"]),
        commence_time=_nullable_utc_datetime(row["commence_time"]),
        is_live=bool(row["is_live"]),
        odds=_nullable_float(row["odds"]),
        line=_nullable_float(row["line"]),
    )


def _nullable_utc_datetime(value: object) -> datetime | None:
    """Normalize one nullable scalar timestamp and require UTC."""
    if _is_missing(value):
        return None
    # pyrefly: ignore [bad-argument-type]
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("Reference timestamps must be timezone-aware UTC.")
    return timestamp.to_pydatetime()


def _nullable_float(value: object) -> float | None:
    """Convert one nullable normalized numerical scalar."""
    if _is_missing(value):
        return None
    # pyrefly: ignore [bad-argument-type]
    return float(value)


def _nullable_text(value: object) -> str | None:
    """Convert one nullable text scalar without inventing identity."""
    return None if _is_missing(value) else str(value)


def _is_missing(value: object) -> bool:
    """Return scalar missingness at the pandas boundary."""
    # pyrefly: ignore [no-matching-overload]
    return bool(pd.isna(value))
