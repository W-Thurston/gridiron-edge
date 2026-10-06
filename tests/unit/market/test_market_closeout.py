"""Tests for validated latest-eligible-pregame market closeout."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS
from gridiron_edge.market.candidate_issuance import (
    CANDIDATE_ISSUANCE_SCHEMA_VERSION,
    CandidateIssuance,
    CandidateIssuanceReason,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_id,
    candidate_issuance_row_id,
)
from gridiron_edge.market.market_closeout import (
    MarketCloseoutReference,
    MarketCloseoutReferenceKind,
    MarketCloseoutResult,
    MarketCloseoutStatus,
    MarketClvKind,
    close_candidate_issuance,
    close_candidate_rows,
    close_market_reference,
    close_recorded_wagers,
)

REFERENCE_FETCH = datetime(2026, 9, 1, 12, tzinfo=UTC)
KICKOFF = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)
GAME_ID = "2026_01_KC_LAC"


def _reference(**overrides: object) -> MarketCloseoutReference:
    values: dict[str, object] = {
        "reference_id": "reference-1",
        "reference_kind": MarketCloseoutReferenceKind.CANDIDATE_ISSUANCE,
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook": "draftkings",
        "game_id": GAME_ID,
        "market": "spread",
        "side": "away",
        "reference_fetched_at": REFERENCE_FETCH,
        "reference_sportsbook_updated_at": REFERENCE_FETCH,
        "reference_kickoff": KICKOFF,
        "reference_is_live": False,
        "reference_american_price": -110,
        "reference_line": 3.5,
    }
    values.update(overrides)
    return MarketCloseoutReference(**values)  # pyrefly: ignore [bad-argument-type]


def _quote(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "fetched_at": datetime(2026, 9, 9, 23, tzinfo=UTC),
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook": "draftkings",
        "sportsbook_updated_at": datetime(2026, 9, 9, 22, 59, tzinfo=UTC),
        "commence_time": KICKOFF,
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": GAME_ID,
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": "spread",
        "side": "away",
        "odds": -110.0,
        "line": 2.5,
    }
    row.update(overrides)
    return row


def _quotes(*rows: dict[str, object]) -> DataFrame:
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_spread_closeout_selects_maximum_eligible_fetch() -> None:
    result = close_market_reference(
        _reference(),
        _quotes(
            _quote(fetched_at=datetime(2026, 9, 1, 13, tzinfo=UTC), line=4.0),
            _quote(line=2.5),
            _quote(
                fetched_at=KICKOFF,
                line=1.5,
            ),
            _quote(
                fetched_at=datetime(2026, 9, 9, 23, 30, tzinfo=UTC),
                line=1.0,
                is_live=True,
            ),
        ),
    )
    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.closeout_line == pytest.approx(2.5)
    assert result.closeout_fetched_at == datetime(2026, 9, 9, 23, tzinfo=UTC)
    assert result.clv_kind is MarketClvKind.SPREAD_POINTS
    assert result.clv == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "other"),
        ("provider_event_id", "event-2"),
        ("sportsbook", "fanduel"),
        ("game_id", "other-game"),
        ("market", "total"),
        ("side", "home"),
    ],
)
def test_every_closeout_identity_field_is_exact(field: str, value: object) -> None:
    overrides = {field: value}
    if field == "market":
        overrides["side"] = "over"
    result = close_market_reference(_reference(), _quotes(_quote(**overrides)))
    assert result.status is MarketCloseoutStatus.CLOSEOUT_MISSING


@pytest.mark.parametrize(
    ("side", "reference_line", "close_line", "expected"),
    [
        ("home", -3.0, -7.0, 4.0),
        ("away", 7.0, 3.0, -4.0),
    ],
)
def test_spread_point_clv_orientation(
    side: str,
    reference_line: float,
    close_line: float,
    expected: float,
) -> None:
    result = close_market_reference(
        _reference(side=side, reference_line=reference_line),
        _quotes(_quote(side=side, line=close_line)),
    )
    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.clv == pytest.approx(expected)


@pytest.mark.parametrize(
    ("side", "reference_line", "close_line", "expected"),
    [
        ("over", 42.0, 45.0, 3.0),
        ("under", 48.0, 45.0, 3.0),
    ],
)
def test_total_point_clv_orientation(
    side: str,
    reference_line: float,
    close_line: float,
    expected: float,
) -> None:
    result = close_market_reference(
        _reference(market="total", side=side, reference_line=reference_line),
        _quotes(_quote(market="total", side=side, line=close_line)),
    )
    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.clv_kind is MarketClvKind.TOTAL_POINTS
    assert result.clv == pytest.approx(expected)


def test_moneyline_price_clv_uses_validated_american_prices() -> None:
    result = close_market_reference(
        _reference(
            market="moneyline",
            side="home",
            reference_line=None,
            reference_american_price=-110,
        ),
        _quotes(_quote(market="moneyline", side="home", line=None, odds=-150.0)),
    )
    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.clv_kind is MarketClvKind.MONEYLINE_PRICE
    assert result.closeout_american_price == -150
    assert result.clv is not None
    assert result.clv > 0.0


@pytest.mark.parametrize(
    ("rows", "status"),
    [
        ((_quote(commence_time=None),), MarketCloseoutStatus.KICKOFF_UNAVAILABLE),
        (
            (
                _quote(),
                _quote(
                    fetched_at=datetime(2026, 9, 9, 22, tzinfo=UTC),
                    commence_time=datetime(2026, 9, 10, 1, 20, tzinfo=UTC),
                ),
            ),
            MarketCloseoutStatus.KICKOFF_CONFLICT,
        ),
        ((_quote(is_live=True),), MarketCloseoutStatus.LIVE_ONLY),
        ((_quote(fetched_at=KICKOFF),), MarketCloseoutStatus.POST_KICKOFF_ONLY),
    ],
)
def test_unavailable_history_states_are_explicit(
    rows: tuple[dict[str, object], ...],
    status: MarketCloseoutStatus,
) -> None:
    result = close_market_reference(_reference(), _quotes(*rows))
    assert result.status is status
    assert result.clv is None


def test_duplicate_maximum_fetch_is_ambiguous() -> None:
    result = close_market_reference(_reference(), _quotes(_quote(), _quote()))
    assert result.status is MarketCloseoutStatus.LATEST_OBSERVATION_AMBIGUOUS


def test_conflicting_maximum_fetch_is_explicit() -> None:
    result = close_market_reference(
        _reference(),
        _quotes(_quote(line=2.5), _quote(line=3.0)),
    )
    assert result.status is MarketCloseoutStatus.LATEST_OBSERVATION_CONFLICT


def test_result_is_input_order_independent_and_inputs_are_not_mutated() -> None:
    reference = _reference()
    observations = _quotes(
        _quote(fetched_at=datetime(2026, 9, 1, 13, tzinfo=UTC), line=4.0),
        _quote(line=2.5),
    )
    before = observations.copy(deep=True)
    first = close_market_reference(reference, observations)
    second = close_market_reference(
        reference,
        DataFrame(observations.iloc[::-1].reset_index(drop=True)),
    )
    assert first == second
    pd.testing.assert_frame_equal(observations, before)


def _issuance() -> CandidateIssuance:
    issuance_id = candidate_issuance_id(
        product_id="product-1",
        product_run_id="run-1",
        season="2026-2027",
        week=1,
        evaluated_at=REFERENCE_FETCH,
    )
    row = CandidateIssuanceRow(
        game_id=GAME_ID,
        market="spread",
        side="away",
        provider="the_odds_api",
        provider_event_id="event-1",
        sportsbook="draftkings",
        line=3.5,
        american_price=-110,
        fetched_at=REFERENCE_FETCH,
        sportsbook_updated_at=REFERENCE_FETCH,
        kickoff=KICKOFF,
        is_live=False,
        forecast_event_id="forecast-1",
        forecast_run_id="run-1",
        forecast_role="live",
        forecast_generated_at=REFERENCE_FETCH,
        model_name="win_prob",
        model_type="elo",
        model_probability=0.55,
        expected_value=0.05,
        state=CandidateIssuanceState.CANDIDATE,
        reason=CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
    )
    return CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        "product-1",
        "run-1",
        REFERENCE_FETCH,
        "2026-2027",
        1,
        REFERENCE_FETCH,
        (row,),
    )


def test_candidate_adapter_preserves_issuance_reference() -> None:
    issuance = _issuance()
    result = close_candidate_issuance(issuance, _quotes(_quote()))[0]
    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.reference.reference_kind is MarketCloseoutReferenceKind.CANDIDATE_ISSUANCE
    assert result.reference.reference_id.startswith(f"{issuance.issuance_id}:")
    assert result.reference.reference_line == pytest.approx(3.5)
    assert result.clv == pytest.approx(-1.0)


def test_chosen_rows_close_exactly_as_the_whole_issuance_does() -> None:
    issuance = _issuance()
    quotes = _quotes(
        _quote(),
        _quote(game_id="2026_01_BUF_NYJ", provider_event_id="event-2", line=1.5),
    )

    chosen = close_candidate_rows(issuance, issuance.rows, quotes)

    assert chosen == close_candidate_issuance(issuance, quotes)
    assert close_candidate_rows(issuance, (), quotes) == ()


def test_contracts_are_frozen() -> None:
    result = close_market_reference(_reference(), _quotes(_quote()))
    with pytest.raises(FrozenInstanceError):
        result.status = MarketCloseoutStatus.CLOSEOUT_MISSING  # pyrefly: ignore [read-only]
    assert isinstance(result, MarketCloseoutResult)


def test_module_does_not_use_storage_order_as_market_state() -> None:
    source = Path("src/gridiron_edge/market/market_closeout.py").read_text()
    forbidden = ("iloc[-1]", "tail(1)", "first stored", "last stored", "opening line")
    assert not any(token in source.lower() for token in forbidden)


def _bet(**overrides: object) -> dict[str, object]:
    """Build one minimal recorded wager with immutable reference evidence."""
    row: dict[str, object] = {
        "bet_id": "bet-1",
        "game_id": GAME_ID,
        "market_type": "spread",
        "side": "away",
        "reference_provider": "the_odds_api",
        "reference_provider_event_id": "event-1",
        "reference_sportsbook": "draftkings",
        "reference_market_fetched_at": REFERENCE_FETCH,
        "reference_sportsbook_updated_at": REFERENCE_FETCH,
        "reference_commence_time": KICKOFF,
        "reference_american_odds": -110,
        "reference_line": 3.5,
    }
    row.update(overrides)
    return row


def _bets(*rows: dict[str, object]) -> DataFrame:
    """Build narrow recorded-wager closeout input."""
    return DataFrame(rows)


def _reference_quote() -> dict[str, object]:
    """Return the exact quote observation recorded by the wager."""
    return _quote(
        fetched_at=REFERENCE_FETCH,
        sportsbook_updated_at=REFERENCE_FETCH,
        line=3.5,
    )


def test_recorded_wager_adapter_validates_reference_then_closes() -> None:
    """A matched immutable wager reference closes against later evidence."""
    bets = _bets(_bet())
    observations = _quotes(_reference_quote(), _quote(line=2.5))
    bets_before = bets.copy(deep=True)
    observations_before = observations.copy(deep=True)

    result = close_recorded_wagers(bets, observations)[0]

    assert result.status is MarketCloseoutStatus.AVAILABLE
    assert result.reference.reference_kind is MarketCloseoutReferenceKind.RECORDED_WAGER
    assert result.reference.reference_id == "bet-1"
    assert result.reference.reference_line == pytest.approx(3.5)
    assert result.closeout_line == pytest.approx(2.5)
    assert result.clv == pytest.approx(-1.0)
    pd.testing.assert_frame_equal(bets, bets_before)
    pd.testing.assert_frame_equal(observations, observations_before)


@pytest.mark.parametrize(
    ("bet", "observations", "status"),
    [
        (
            _bet(
                reference_provider=None,
                reference_provider_event_id=None,
                reference_sportsbook=None,
                reference_market_fetched_at=None,
                reference_sportsbook_updated_at=None,
                reference_commence_time=None,
                reference_american_odds=None,
                reference_line=None,
            ),
            _quotes(),
            MarketCloseoutStatus.REFERENCE_UNAVAILABLE,
        ),
        (
            _bet(),
            _quotes(_quote(provider="other")),
            MarketCloseoutStatus.REFERENCE_MISSING,
        ),
        (
            _bet(),
            _quotes(_reference_quote(), _reference_quote()),
            MarketCloseoutStatus.REFERENCE_AMBIGUOUS,
        ),
        (
            _bet(reference_line=4.0),
            _quotes(_reference_quote()),
            MarketCloseoutStatus.REFERENCE_CONFLICT,
        ),
    ],
)
def test_recorded_wager_reference_states_remain_explicit(
    bet: dict[str, object],
    observations: DataFrame,
    status: MarketCloseoutStatus,
) -> None:
    """Reference failures cannot silently proceed to CLV calculation."""
    result = close_recorded_wagers(_bets(bet), observations)[0]
    assert result.status is status
    assert result.clv is None
    assert result.closeout_fetched_at is None


def test_recorded_wager_results_are_sorted_by_bet_id() -> None:
    """Recorded closeout is deterministic and independent of input bet order."""
    bets = _bets(
        _bet(bet_id="z-bet"),
        _bet(bet_id="a-bet"),
    )
    results = close_recorded_wagers(
        bets,
        _quotes(_reference_quote(), _quote(line=2.5)),
    )
    assert tuple(result.reference.reference_id for result in results) == (
        "a-bet",
        "z-bet",
    )


@pytest.mark.parametrize(
    ("reference", "quote", "status"),
    [
        (
            _reference(
                market="moneyline",
                side="home",
                reference_line=None,
                reference_american_price=None,
            ),
            _quote(market="moneyline", side="home", line=None, odds=-150.0),
            MarketCloseoutStatus.REFERENCE_PRICE_UNAVAILABLE,
        ),
        (
            _reference(
                market="moneyline",
                side="home",
                reference_line=None,
                reference_american_price=-110,
            ),
            _quote(market="moneyline", side="home", line=None, odds=None),
            MarketCloseoutStatus.CLOSEOUT_PRICE_UNAVAILABLE,
        ),
        (
            _reference(reference_line=None),
            _quote(line=2.5),
            MarketCloseoutStatus.REFERENCE_LINE_UNAVAILABLE,
        ),
        (
            _reference(),
            _quote(line=None),
            MarketCloseoutStatus.CLOSEOUT_LINE_UNAVAILABLE,
        ),
    ],
)
def test_missing_clv_terms_remain_explicit(
    reference: MarketCloseoutReference,
    quote: dict[str, object],
    status: MarketCloseoutStatus,
) -> None:
    """Missing price or line evidence never becomes zero CLV."""
    result = close_market_reference(reference, _quotes(quote))
    assert result.status is status
    assert result.clv_kind is None
    assert result.clv is None


def test_live_before_kickoff_with_post_kickoff_non_live_evidence_is_live_only() -> None:
    """Only live evidence exists inside the strictly pre-kickoff window."""
    result = close_market_reference(
        _reference(),
        _quotes(
            _quote(is_live=True),
            _quote(fetched_at=KICKOFF),
        ),
    )
    assert result.status is MarketCloseoutStatus.LIVE_ONLY
    assert result.clv is None


def test_candidate_unavailable_reference_does_not_calculate_clv() -> None:
    """Incomplete immutable candidate evidence remains reference unavailable."""
    issuance = _issuance()
    incomplete = CandidateIssuanceRow(
        game_id=GAME_ID,
        market="spread",
        side="away",
        provider="the_odds_api",
        provider_event_id="event-1",
        sportsbook="draftkings",
        line=3.5,
        american_price=-110,
        fetched_at=REFERENCE_FETCH,
        sportsbook_updated_at=REFERENCE_FETCH,
        kickoff=KICKOFF,
        is_live=True,
        forecast_event_id="forecast-1",
        forecast_run_id="run-1",
        forecast_role="live",
        forecast_generated_at=REFERENCE_FETCH,
        model_name="win_prob",
        model_type="elo",
        model_probability=None,
        expected_value=None,
        state=CandidateIssuanceState.UNAVAILABLE,
        reason=CandidateIssuanceReason.QUOTE_LIVE,
    )
    changed = CandidateIssuance(
        issuance.schema_version,
        issuance.issuance_id,
        issuance.product_id,
        issuance.product_run_id,
        issuance.product_generated_at,
        issuance.season,
        issuance.week,
        issuance.evaluated_at,
        (incomplete,),
    )
    result = close_candidate_issuance(changed, _quotes(_quote()))[0]
    assert result.status is MarketCloseoutStatus.REFERENCE_UNAVAILABLE
    assert result.clv is None


def test_candidate_closeout_reference_uses_shared_row_identity() -> None:
    issuance = _issuance()
    result = close_candidate_issuance(issuance, _quotes(_quote()))[0]
    candidate = next(row for row in issuance.rows if row.state is CandidateIssuanceState.CANDIDATE)
    assert result.reference.reference_id == candidate_issuance_row_id(
        issuance.issuance_id, candidate
    )
