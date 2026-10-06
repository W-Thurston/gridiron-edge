"""Validated latest-eligible-pregame market closeout and CLV.

Closeout uses exact provider-aware market identity and the maximum eligible
pregame fetch timestamp. It never interprets raw storage order as market state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, cast

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.ingest.odds.store import validate_quote_rows
from gridiron_edge.market.bet_reference_matching import (
    BetReferenceMatch,
    BetReferenceMatchStatus,
    match_bet_references,
)
from gridiron_edge.market.candidate_issuance import (
    CandidateIssuance,
    CandidateIssuanceRow,
    candidate_issuance_row_id,
)
from gridiron_edge.market.clv import closing_line_value, spread_clv, total_clv
from gridiron_edge.market.odds_math import american_to_implied_prob


class MarketCloseoutReferenceKind(StrEnum):
    """Origin of immutable evidence being closed."""

    CANDIDATE_ISSUANCE = "candidate_issuance"
    RECORDED_WAGER = "recorded_wager"


class MarketCloseoutStatus(StrEnum):
    """Resolution of one reference against canonical quote history."""

    AVAILABLE = "available"
    REFERENCE_UNAVAILABLE = "reference_unavailable"
    REFERENCE_MISSING = "reference_missing"
    REFERENCE_AMBIGUOUS = "reference_ambiguous"
    REFERENCE_CONFLICT = "reference_conflict"
    CLOSEOUT_MISSING = "closeout_missing"
    KICKOFF_UNAVAILABLE = "kickoff_unavailable"
    KICKOFF_CONFLICT = "kickoff_conflict"
    LIVE_ONLY = "live_only"
    POST_KICKOFF_ONLY = "post_kickoff_only"
    NO_ELIGIBLE_PREGAME_OBSERVATION = "no_eligible_pregame_observation"
    LATEST_OBSERVATION_AMBIGUOUS = "latest_observation_ambiguous"
    LATEST_OBSERVATION_CONFLICT = "latest_observation_conflict"
    REFERENCE_PRICE_UNAVAILABLE = "reference_price_unavailable"
    CLOSEOUT_PRICE_UNAVAILABLE = "closeout_price_unavailable"
    REFERENCE_LINE_UNAVAILABLE = "reference_line_unavailable"
    CLOSEOUT_LINE_UNAVAILABLE = "closeout_line_unavailable"


class MarketClvKind(StrEnum):
    """Semantic unit of one validated CLV calculation."""

    MONEYLINE_PRICE = "moneyline_price"
    SPREAD_POINTS = "spread_points"
    TOTAL_POINTS = "total_points"


@dataclass(frozen=True, slots=True)
class MarketCloseoutReference:
    """Immutable exact market evidence to close."""

    reference_id: str
    reference_kind: MarketCloseoutReferenceKind
    provider: str | None
    provider_event_id: str | None
    sportsbook: str | None
    game_id: str
    market: str
    side: str
    reference_fetched_at: datetime | None
    reference_sportsbook_updated_at: datetime | None
    reference_kickoff: datetime | None
    reference_is_live: bool | None
    reference_american_price: int | None
    reference_line: float | None


@dataclass(frozen=True, slots=True)
class MarketCloseoutResult:
    """Validated market closeout evidence and optional CLV."""

    reference: MarketCloseoutReference
    status: MarketCloseoutStatus
    closeout_fetched_at: datetime | None = None
    closeout_sportsbook_updated_at: datetime | None = None
    closeout_kickoff: datetime | None = None
    closeout_is_live: bool | None = None
    closeout_american_price: int | None = None
    closeout_line: float | None = None
    clv_kind: MarketClvKind | None = None
    clv: float | None = None


def close_market_reference(
    reference: MarketCloseoutReference,
    observations: DataFrame,
) -> MarketCloseoutResult:
    """Close one immutable reference against exact latest eligible evidence."""
    _validate_reference(reference)
    rows = validate_quote_rows(observations)
    result: MarketCloseoutResult
    if not _reference_root_available(reference):
        result = _unavailable(reference, MarketCloseoutStatus.REFERENCE_UNAVAILABLE)
    else:
        provider = reference.provider
        assert provider is not None
        matching = cast(
            DataFrame,
            rows.loc[
                rows["provider"].eq(provider)
                & _nullable_identity_mask(
                    rows["provider_event_id"],
                    reference.provider_event_id,
                )
                & _nullable_identity_mask(rows["sportsbook"], reference.sportsbook)
                & rows["game_id"].eq(reference.game_id)
                & rows["market"].eq(reference.market)
                & rows["side"].eq(reference.side),
                :,
            ],
        )
        result = _close_matching_history(reference, matching)
    return result


def _close_matching_history(
    reference: MarketCloseoutReference,
    matching: DataFrame,
) -> MarketCloseoutResult:
    """Resolve exact-identity history to one validated closeout result."""
    result: MarketCloseoutResult
    if matching.empty:
        result = _unavailable(reference, MarketCloseoutStatus.CLOSEOUT_MISSING)
    else:
        kickoff_values = matching["commence_time"].dropna().drop_duplicates()
        if kickoff_values.empty:
            result = _unavailable(reference, MarketCloseoutStatus.KICKOFF_UNAVAILABLE)
        elif len(kickoff_values) > 1:
            result = _unavailable(reference, MarketCloseoutStatus.KICKOFF_CONFLICT)
        else:
            kickoff = _datetime(kickoff_values.iloc[0])
            before_kickoff = cast(
                DataFrame,
                matching.loc[matching["fetched_at"].lt(pd.Timestamp(kickoff)), :],
            )
            eligible = cast(
                DataFrame,
                before_kickoff.loc[before_kickoff["is_live"].eq(False), :],
            )
            result = _close_eligible_history(
                reference,
                matching=matching,
                before_kickoff=before_kickoff,
                eligible=eligible,
            )
    return result


def _close_eligible_history(
    reference: MarketCloseoutReference,
    *,
    matching: DataFrame,
    before_kickoff: DataFrame,
    eligible: DataFrame,
) -> MarketCloseoutResult:
    """Select and evaluate the maximum eligible pregame fetch."""
    if eligible.empty:
        return _unavailable(
            reference,
            _empty_eligibility_status(matching, before_kickoff),
        )
    maximum_fetch = eligible["fetched_at"].max()
    selected_rows = cast(
        DataFrame,
        eligible.loc[eligible["fetched_at"].eq(maximum_fetch), :],
    )
    if len(selected_rows) > 1:
        return _unavailable(reference, _multiple_maximum_status(selected_rows))
    selected = cast(Series, selected_rows.iloc[0])
    evidence = _selected_evidence(selected)
    status, clv_kind, clv = _calculate_clv(reference, selected)
    return MarketCloseoutResult(
        reference=reference,
        status=status,
        closeout_fetched_at=evidence["fetched_at"],
        closeout_sportsbook_updated_at=evidence["sportsbook_updated_at"],
        closeout_kickoff=evidence["kickoff"],
        closeout_is_live=evidence["is_live"],
        closeout_american_price=evidence["american_price"],
        closeout_line=evidence["line"],
        clv_kind=clv_kind,
        clv=clv,
    )


def close_candidate_issuance(
    issuance: CandidateIssuance,
    observations: DataFrame,
) -> tuple[MarketCloseoutResult, ...]:
    """Close every immutable issuance row without re-resolving its evidence."""
    results = [
        close_market_reference(
            _candidate_reference(issuance.issuance_id, row),
            observations,
        )
        for row in issuance.rows
    ]
    return tuple(sorted(results, key=lambda result: result.reference.reference_id))


def close_candidate_rows(
    issuance: CandidateIssuance,
    rows: tuple[CandidateIssuanceRow, ...],
    observations: DataFrame,
) -> tuple[MarketCloseoutResult, ...]:
    """Close chosen issuance rows, each against its own game's observations only.

    Matching is :func:`close_market_reference`'s, which already requires the row's game, so a
    row closes exactly as it would against the whole history; splitting the history by game
    only keeps each search small.
    """
    quotes = validate_quote_rows(observations)
    by_game = {str(game_id): frame for game_id, frame in quotes.groupby("game_id", sort=False)}
    empty = quotes.iloc[0:0]
    results = [
        close_market_reference(
            _candidate_reference(issuance.issuance_id, row),
            by_game.get(row.game_id, empty),
        )
        for row in rows
    ]
    return tuple(sorted(results, key=lambda result: result.reference.reference_id))


def close_recorded_wagers(
    bets: DataFrame,
    observations: DataFrame,
) -> tuple[MarketCloseoutResult, ...]:
    """Close recorded wagers only after exact immutable reference matching."""
    matches = match_bet_references(bets, observations)
    results = [_close_recorded_wager_match(match, observations) for match in matches]
    return tuple(sorted(results, key=lambda result: result.reference.reference_id))


def _close_recorded_wager_match(
    match: BetReferenceMatch,
    observations: DataFrame,
) -> MarketCloseoutResult:
    """Map one reference diagnostic into the common closeout contract."""
    reference = _recorded_wager_reference(match)
    status_by_match = {
        BetReferenceMatchStatus.MANUAL_BET: MarketCloseoutStatus.REFERENCE_UNAVAILABLE,
        BetReferenceMatchStatus.OBSERVATION_NOT_FOUND: MarketCloseoutStatus.REFERENCE_MISSING,
        BetReferenceMatchStatus.AMBIGUOUS_OBSERVATION: (MarketCloseoutStatus.REFERENCE_AMBIGUOUS),
        BetReferenceMatchStatus.REFERENCE_TERMS_CONFLICT: (MarketCloseoutStatus.REFERENCE_CONFLICT),
    }
    if match.status is BetReferenceMatchStatus.MATCHED:
        return close_market_reference(reference, observations)
    return _unavailable(reference, status_by_match[match.status])


def _recorded_wager_reference(match: BetReferenceMatch) -> MarketCloseoutReference:
    """Adapt one exact recorded-wager reference without mutating the ledger."""
    observation = match.matched_observation
    return MarketCloseoutReference(
        reference_id=match.bet_id,
        reference_kind=MarketCloseoutReferenceKind.RECORDED_WAGER,
        provider=match.provider,
        provider_event_id=match.provider_event_id,
        sportsbook=match.sportsbook,
        game_id=match.game_id,
        market=match.market,
        side=match.side,
        reference_fetched_at=match.reference_fetched_at,
        reference_sportsbook_updated_at=(
            None if observation is None else observation.sportsbook_updated_at
        ),
        reference_kickoff=None if observation is None else observation.commence_time,
        reference_is_live=None if observation is None else observation.is_live,
        reference_american_price=(
            None if observation is None or observation.odds is None else int(observation.odds)
        ),
        reference_line=None if observation is None else observation.line,
    )


def _candidate_reference(
    issuance_id: str,
    row: CandidateIssuanceRow,
) -> MarketCloseoutReference:
    """Adapt one immutable candidate row into the common reference contract."""
    return MarketCloseoutReference(
        reference_id=candidate_issuance_row_id(issuance_id, row),
        reference_kind=MarketCloseoutReferenceKind.CANDIDATE_ISSUANCE,
        provider=row.provider,
        provider_event_id=row.provider_event_id,
        sportsbook=row.sportsbook,
        game_id=row.game_id,
        market=row.market,
        side=row.side,
        reference_fetched_at=row.fetched_at,
        reference_sportsbook_updated_at=row.sportsbook_updated_at,
        reference_kickoff=row.kickoff,
        reference_is_live=row.is_live,
        reference_american_price=row.american_price,
        reference_line=row.line,
    )


def _validate_reference(reference: MarketCloseoutReference) -> None:
    """Validate stable identity and market-side invariants."""
    for label, value in (
        ("reference_id", reference.reference_id),
        ("game_id", reference.game_id),
        ("market", reference.market),
        ("side", reference.side),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty.")
    valid_sides = {
        "moneyline": {"away", "home"},
        "spread": {"away", "home"},
        "total": {"over", "under"},
    }
    if reference.market not in valid_sides:
        raise ValueError(f"Unsupported closeout market: {reference.market!r}.")
    if reference.side not in valid_sides[reference.market]:
        raise ValueError(
            f"Unsupported closeout market-side pair: {reference.market}/{reference.side}."
        )
    for label, value in (
        ("reference_fetched_at", reference.reference_fetched_at),
        ("reference_sportsbook_updated_at", reference.reference_sportsbook_updated_at),
        ("reference_kickoff", reference.reference_kickoff),
    ):
        if value is not None:
            _require_utc(value, label=label)


def _reference_root_available(reference: MarketCloseoutReference) -> bool:
    """Return whether exact source identity can support validated closeout."""
    return (
        reference.provider is not None
        and bool(reference.provider.strip())
        and reference.reference_fetched_at is not None
        and reference.reference_is_live is False
    )


def _empty_eligibility_status(
    matching: DataFrame,
    before_kickoff: DataFrame,
) -> MarketCloseoutStatus:
    """Describe why exact-identity history has no eligible observation."""
    if not before_kickoff.empty and before_kickoff["is_live"].eq(True).all():
        return MarketCloseoutStatus.LIVE_ONLY
    non_live = matching.loc[matching["is_live"].eq(False), :]
    if not non_live.empty and non_live["fetched_at"].ge(non_live["commence_time"]).all():
        return MarketCloseoutStatus.POST_KICKOFF_ONLY
    return MarketCloseoutStatus.NO_ELIGIBLE_PREGAME_OBSERVATION


def _multiple_maximum_status(rows: DataFrame) -> MarketCloseoutStatus:
    """Distinguish duplicate from conflicting maximum-fetch evidence."""
    terms = ["odds", "line", "sportsbook_updated_at", "commence_time", "is_live"]
    unique = rows.loc[:, terms].drop_duplicates()
    return (
        MarketCloseoutStatus.LATEST_OBSERVATION_AMBIGUOUS
        if len(unique) == 1
        else MarketCloseoutStatus.LATEST_OBSERVATION_CONFLICT
    )


def _calculate_clv(
    reference: MarketCloseoutReference,
    selected: Series,
) -> tuple[MarketCloseoutStatus, MarketClvKind | None, float | None]:
    """Calculate market-specific CLV only from complete validated terms."""
    close_price = _optional_int(selected["odds"])
    close_line = _optional_float(selected["line"])
    status: MarketCloseoutStatus
    kind: MarketClvKind | None = None
    value: float | None = None
    if reference.market == "moneyline":
        if reference.reference_american_price is None:
            status = MarketCloseoutStatus.REFERENCE_PRICE_UNAVAILABLE
        elif close_price is None:
            status = MarketCloseoutStatus.CLOSEOUT_PRICE_UNAVAILABLE
        else:
            status = MarketCloseoutStatus.AVAILABLE
            kind = MarketClvKind.MONEYLINE_PRICE
            value = closing_line_value(
                american_to_implied_prob(reference.reference_american_price),
                american_to_implied_prob(close_price),
            )
    elif reference.reference_line is None:
        status = MarketCloseoutStatus.REFERENCE_LINE_UNAVAILABLE
    elif close_line is None:
        status = MarketCloseoutStatus.CLOSEOUT_LINE_UNAVAILABLE
    elif reference.market == "spread":
        status = MarketCloseoutStatus.AVAILABLE
        kind = MarketClvKind.SPREAD_POINTS
        value = spread_clv(reference.reference_line, close_line, reference.side)
    else:
        status = MarketCloseoutStatus.AVAILABLE
        kind = MarketClvKind.TOTAL_POINTS
        value = total_clv(reference.reference_line, close_line, reference.side)
    return status, kind, value


def _selected_evidence(row: Series) -> dict[str, Any]:
    """Preserve one exact selected latest-eligible-pregame observation."""
    return {
        "fetched_at": _datetime(row["fetched_at"]),
        "sportsbook_updated_at": _optional_datetime(row["sportsbook_updated_at"]),
        "kickoff": _optional_datetime(row["commence_time"]),
        "is_live": bool(row["is_live"]),
        "american_price": _optional_int(row["odds"]),
        "line": _optional_float(row["line"]),
    }


def _unavailable(
    reference: MarketCloseoutReference,
    status: MarketCloseoutStatus,
) -> MarketCloseoutResult:
    return MarketCloseoutResult(reference=reference, status=status)


def _nullable_identity_mask(values: Series, expected: str | None) -> Series:
    return values.isna() if expected is None else values.eq(expected)


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value


def _datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(cast(Any, value))
    result = timestamp.to_pydatetime()
    return _require_utc(result, label="quote timestamp")


def _optional_datetime(value: object) -> datetime | None:
    return None if pd.isna(cast(Any, value)) else _datetime(value)


def _optional_float(value: object) -> float | None:
    return None if pd.isna(cast(Any, value)) else float(cast(float | int, value))


def _optional_int(value: object) -> int | None:
    return None if pd.isna(cast(Any, value)) else int(float(cast(float | int, value)))
