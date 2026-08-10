"""Tests for exact bet-reference matching diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS, empty_quote_frame
from gridiron_edge.market.bet_reference_matching import (
    BetReferenceMatch,
    BetReferenceMatchStatus,
    match_bet_references,
)
from gridiron_edge.market.history_boundaries import SelectedQuoteObservation

FETCHED = datetime(2026, 9, 1, 12, tzinfo=UTC)
UPDATED = datetime(2026, 9, 1, 11, 59, tzinfo=UTC)
KICKOFF = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


def _bet(**overrides: object) -> dict[str, object]:
    """Build one minimal reference-backed bet row."""
    row: dict[str, object] = {
        "bet_id": "bet-1",
        "game_id": "2026_01_KC_LAC",
        "market_type": "spread",
        "side": "away",
        "book": "fanduel",
        "odds": -105,
        "line": 4.0,
        "placed_at": datetime(2026, 9, 1, 12, 5, tzinfo=UTC),
        "reference_provider": "the_odds_api",
        "reference_provider_event_id": "event-1",
        "reference_sportsbook": "draftkings",
        "reference_market_fetched_at": FETCHED,
        "reference_sportsbook_updated_at": UPDATED,
        "reference_commence_time": KICKOFF,
        "reference_american_odds": -110,
        "reference_line": 3.5,
    }
    row.update(overrides)
    return row


def _manual_bet(**overrides: object) -> dict[str, object]:
    """Build one manual bet with explicit null reference provenance."""
    row = _bet(
        reference_provider=None,
        reference_provider_event_id=None,
        reference_sportsbook=None,
        reference_market_fetched_at=None,
        reference_sportsbook_updated_at=None,
        reference_commence_time=None,
        reference_american_odds=None,
        reference_line=None,
    )
    row.update(overrides)
    return row


def _quote(**overrides: object) -> dict[str, object]:
    """Build one canonical matching quote row."""
    row: dict[str, object] = {
        "fetched_at": FETCHED,
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook": "draftkings",
        "sportsbook_updated_at": UPDATED,
        "commence_time": KICKOFF,
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": "spread",
        "side": "away",
        "odds": -110.0,
        "line": 3.5,
    }
    row.update(overrides)
    return row


def _bets(*rows: dict[str, object]) -> DataFrame:
    """Build matcher input bets without requiring the full ledger schema."""
    return DataFrame(rows)


def _quotes(*rows: dict[str, object]) -> DataFrame:
    """Build canonical quote observations."""
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_empty_bets_return_no_results() -> None:
    """No recorded bets produce no diagnostics."""
    empty_bets = DataFrame(columns=list(_bet()))
    assert match_bet_references(empty_bets, empty_quote_frame()) == ()


def test_manual_bet_requires_no_quote_lookup() -> None:
    """Manual wagers remain explicit without historical evidence."""
    result = match_bet_references(
        _bets(_manual_bet()),
        empty_quote_frame(),
    )[0]
    assert result.status is BetReferenceMatchStatus.MANUAL_BET
    assert result.provider is None
    assert result.reference_fetched_at is None
    assert result.matched_observation is None
    assert result.mismatched_fields == ()


def test_exact_reference_matches_immutable_observation() -> None:
    """Exact source identity and terms establish one matched observation."""
    result = match_bet_references(_bets(_bet()), _quotes(_quote()))[0]
    assert result.status is BetReferenceMatchStatus.MATCHED
    assert result.provider == "the_odds_api"
    assert result.provider_event_id == "event-1"
    assert result.sportsbook == "draftkings"
    assert result.reference_fetched_at == FETCHED
    assert result.mismatched_fields == ()
    assert result.matched_observation == SelectedQuoteObservation(
        fetched_at=FETCHED,
        sportsbook_updated_at=UPDATED,
        commence_time=KICKOFF,
        is_live=False,
        odds=-110.0,
        line=3.5,
    )


@pytest.mark.parametrize(
    ("quote_override", "bet_override"),
    [
        ({"provider": "other"}, {}),
        ({"provider_event_id": "event-2"}, {}),
        ({"sportsbook": "fanduel"}, {}),
        ({"game_id": "2026_01_BUF_MIA"}, {}),
        ({"market": "moneyline", "line": None}, {}),
        ({"side": "home"}, {}),
        ({"fetched_at": datetime(2026, 9, 1, 13, tzinfo=UTC)}, {}),
    ],
)
def test_every_identity_field_is_exact(
    quote_override: dict[str, object],
    bet_override: dict[str, object],
) -> None:
    """Every persisted source-aware identity field participates in lookup."""
    result = match_bet_references(
        _bets(_bet(**bet_override)),
        _quotes(_quote(**quote_override)),
    )[0]
    assert result.status is BetReferenceMatchStatus.OBSERVATION_NOT_FOUND


def test_null_identities_match_only_null() -> None:
    """Source-neutral null identity is exact rather than a wildcard."""
    bet = _bet(
        reference_provider="nflverse",
        reference_provider_event_id=None,
        reference_sportsbook=None,
        reference_sportsbook_updated_at=None,
        reference_commence_time=None,
        reference_american_odds=None,
        reference_line=None,
    )
    quote = _quote(
        provider="nflverse",
        provider_event_id=None,
        sportsbook=None,
        sportsbook_updated_at=None,
        commence_time=None,
        odds=None,
        line=None,
    )
    matched = match_bet_references(_bets(bet), _quotes(quote))[0]
    assert matched.status is BetReferenceMatchStatus.MATCHED

    nonnull_book = _quote(
        provider="nflverse",
        provider_event_id=None,
        sportsbook="draftkings",
        sportsbook_updated_at=None,
        commence_time=None,
        odds=None,
        line=None,
    )
    missing = match_bet_references(_bets(bet), _quotes(nonnull_book))[0]
    assert missing.status is BetReferenceMatchStatus.OBSERVATION_NOT_FOUND


def test_duplicate_exact_candidates_are_ambiguous() -> None:
    """Malformed duplicate candidates are never selected arbitrarily."""
    result = match_bet_references(
        _bets(_bet()),
        _quotes(_quote(), _quote()),
    )[0]
    assert result.status is BetReferenceMatchStatus.AMBIGUOUS_OBSERVATION
    assert result.matched_observation is None


@pytest.mark.parametrize(
    ("bet_override", "expected"),
    [
        (
            {"reference_sportsbook_updated_at": datetime(2026, 9, 1, 11, 58, tzinfo=UTC)},
            ("reference_sportsbook_updated_at",),
        ),
        (
            {"reference_commence_time": datetime(2026, 9, 10, 1, 20, tzinfo=UTC)},
            ("reference_commence_time",),
        ),
        (
            {"reference_american_odds": -105},
            ("reference_american_odds",),
        ),
        ({"reference_line": 4.0}, ("reference_line",)),
    ],
)
def test_each_reference_term_conflict_is_named(
    bet_override: dict[str, object],
    expected: tuple[str, ...],
) -> None:
    """Each immutable term conflict reports its canonical field name."""
    result = match_bet_references(
        _bets(_bet(**bet_override)),
        _quotes(_quote()),
    )[0]
    assert result.status is BetReferenceMatchStatus.REFERENCE_TERMS_CONFLICT
    assert result.matched_observation is None
    assert result.mismatched_fields == expected


def test_multiple_conflicts_use_canonical_order() -> None:
    """Term diagnostics are stable and independent of row ordering."""
    result = match_bet_references(
        _bets(
            _bet(
                reference_sportsbook_updated_at=None,
                reference_commence_time=None,
                reference_american_odds=-105,
                reference_line=4.0,
            )
        ),
        _quotes(_quote()),
    )[0]
    assert result.mismatched_fields == (
        "reference_sportsbook_updated_at",
        "reference_commence_time",
        "reference_american_odds",
        "reference_line",
    )


def test_null_terms_match_only_null() -> None:
    """Missing immutable terms are exact values, not wildcards."""
    bet = _bet(
        reference_sportsbook_updated_at=None,
        reference_commence_time=None,
        reference_line=None,
    )
    quote = _quote(
        sportsbook_updated_at=None,
        commence_time=None,
        line=None,
    )
    matched = match_bet_references(_bets(bet), _quotes(quote))[0]
    assert matched.status is BetReferenceMatchStatus.MATCHED

    conflict = match_bet_references(_bets(bet), _quotes(_quote()))[0]
    assert conflict.mismatched_fields == (
        "reference_sportsbook_updated_at",
        "reference_commence_time",
        "reference_line",
    )


def test_actual_wager_terms_do_not_affect_reference_match() -> None:
    """Actual book, odds, line, and placement time are not reference identity."""
    changed_actual = _bet(
        book="betmgm",
        odds=125,
        line=7.5,
        placed_at=datetime(2026, 9, 2, 15, tzinfo=UTC),
    )
    result = match_bet_references(
        _bets(changed_actual),
        _quotes(_quote()),
    )[0]
    assert result.status is BetReferenceMatchStatus.MATCHED


def test_equivalent_utc_timestamp_types_match() -> None:
    """Equivalent UTC timestamp representations resolve to one instant."""
    bet = _bet(
        reference_market_fetched_at=pd.Timestamp(FETCHED),
        reference_sportsbook_updated_at=pd.Timestamp(UPDATED),
        reference_commence_time=pd.Timestamp(KICKOFF),
    )
    assert (
        match_bet_references(_bets(bet), _quotes(_quote()))[0].status
        is BetReferenceMatchStatus.MATCHED
    )


def test_results_are_sorted_by_bet_id_without_mutating_inputs() -> None:
    """Result order is deterministic and both input frames remain unchanged."""
    bets = _bets(_manual_bet(bet_id="z-bet"), _bet(bet_id="a-bet"))
    quotes = _quotes(_quote())
    bets_before = bets.copy(deep=True)
    quotes_before = quotes.copy(deep=True)

    result = match_bet_references(bets, quotes)

    assert tuple(item.bet_id for item in result) == ("a-bet", "z-bet")
    pd.testing.assert_frame_equal(bets, bets_before)
    pd.testing.assert_frame_equal(quotes, quotes_before)


@pytest.mark.parametrize(
    "bets",
    [
        _bets(_manual_bet(bet_id="")),
        _bets(_manual_bet(bet_id="same"), _manual_bet(bet_id="same")),
    ],
)
def test_invalid_bet_ids_are_rejected(bets: DataFrame) -> None:
    """Result identity requires unique nonempty bet IDs."""
    with pytest.raises(ValueError, match="bet_id"):
        match_bet_references(bets, empty_quote_frame())


def test_missing_required_bet_column_is_rejected() -> None:
    """The matcher validates its narrow required bet contract."""
    bets = _bets(_bet()).drop(columns="reference_line")
    with pytest.raises(ValueError, match="reference_line"):
        match_bet_references(bets, _quotes(_quote()))


def test_missing_quote_column_is_rejected() -> None:
    """Quote validation remains owned by the canonical store contract."""
    quotes = _quotes(_quote()).drop(columns="provider")
    with pytest.raises(ValueError, match="missing columns"):
        match_bet_references(_bets(_bet()), quotes)


def test_match_contract_is_frozen() -> None:
    """Reference diagnostics and matched observations are immutable."""
    result = match_bet_references(_bets(_bet()), _quotes(_quote()))[0]
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        result.status = BetReferenceMatchStatus.MANUAL_BET
    assert isinstance(result, BetReferenceMatch)
    assert isinstance(result.matched_observation, SelectedQuoteObservation)
