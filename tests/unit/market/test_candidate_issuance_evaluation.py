"""Acceptance tests for pure pregame candidate issuance evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS
from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS
from gridiron_edge.market.candidate_issuance import (
    CandidateIssuanceReason,
    CandidateIssuanceState,
    issue_pregame_candidates,
)

EVALUATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)
PRODUCT_GENERATED_AT = datetime(2026, 9, 1, 11, tzinfo=UTC)
KICKOFF = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)
FETCHED_AT = datetime(2026, 9, 1, 11, 30, tzinfo=UTC)
UPDATED_AT = datetime(2026, 9, 1, 11, 29, tzinfo=UTC)
GAME_ID = "2026_01_KC_LAC"


def _product(**overrides: object) -> DataFrame:
    """Return one fully validated stored weekly-product row."""
    row: dict[str, object] = {
        "product_schema_version": 1,
        "product_id": "product-1",
        "product_run_id": "run-1",
        "product_generated_at": PRODUCT_GENERATED_AT,
        "season": "2026-2027",
        "week": 1,
        "game_id": GAME_ID,
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "neutral_site": False,
        "win_status": "available",
        "win_selection_status": "selected",
        "away_win_prob": 0.40,
        "home_win_prob": 0.60,
        "win_model_name": "win_prob",
        "win_model_type": "elo",
        "win_event_id": "win-1",
        "win_run_id": "run-1",
        "win_generated_at": PRODUCT_GENERATED_AT,
        "win_role": "live",
        "spread_status": "available",
        "model_spread": -3.0,
        "spread_uncertainty": 13.5,
        "spread_source_event_id": "win-1",
        "spread_model_name": "win_prob",
        "spread_model_type": "elo",
        "spread_calibration_key": "win_prob_elo",
        "spread_calibration_updated_at": "2026-07-30T12:00:00+00:00",
        "total_status": "available",
        "model_total": 44.0,
        "total_uncertainty": 12.8,
        "total_model_name": "total",
        "total_model_type": "xgboost",
        "total_event_id": "total-1",
        "total_run_id": "run-1",
        "total_generated_at": PRODUCT_GENERATED_AT,
        "total_role": "live",
        "total_selection_status": "selected",
        "total_uncertainty_trained_at": "2026-07-01T14:20:00",
        "projected_score_status": "available",
        "projected_home_score": 23.5,
        "projected_away_score": 20.5,
    }
    row.update(overrides)
    return DataFrame([row])


def _event(
    *,
    event_id: str,
    model_name: str,
    model_type: str,
    home_win_prob: float | None = None,
    model_total: float | None = None,
) -> dict[str, object]:
    """Return one exact canonical forecast-event row."""
    return {
        "event_id": event_id,
        "run_id": "run-1",
        "role": "live",
        "generated_at": PRODUCT_GENERATED_AT,
        "season": "2026-2027",
        "week": 1,
        "game_id": GAME_ID,
        "model_name": model_name,
        "model_type": model_type,
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "away_elo": None,
        "home_elo": None,
        "away_win_prob": (None if home_win_prob is None else 1.0 - home_win_prob),
        "home_win_prob": home_win_prob,
        "model_spread": None,
        "model_total": model_total,
        "projected_home_score": None,
        "projected_away_score": None,
        "margin_std": None,
        "win_prob_lo": None,
        "win_prob_hi": None,
        "confidence_tier": None,
    }


def _events(*, include_win: bool = True, include_total: bool = True) -> DataFrame:
    """Return exact forecast events selected by the product."""
    rows: list[dict[str, object]] = []
    if include_win:
        rows.append(
            _event(
                event_id="win-1",
                model_name="win_prob",
                model_type="elo",
                home_win_prob=0.60,
            )
        )
    if include_total:
        rows.append(
            _event(
                event_id="total-1",
                model_name="total",
                model_type="xgboost",
                model_total=44.0,
            )
        )
    return DataFrame(rows, columns=FORECAST_EVENT_COLUMNS)


def _quote(**overrides: object) -> dict[str, object]:
    """Return one exact canonical provider-aware quote observation."""
    row: dict[str, object] = {
        "fetched_at": FETCHED_AT,
        "provider": "the_odds_api",
        "provider_event_id": "provider-event-1",
        "sportsbook": "draftkings",
        "sportsbook_updated_at": UPDATED_AT,
        "commence_time": KICKOFF,
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": GAME_ID,
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": "moneyline",
        "side": "home",
        "odds": -110.0,
        "line": None,
    }
    row.update(overrides)
    return row


def _quotes(*rows: dict[str, object]) -> DataFrame:
    """Return canonical quote observations in exact schema order."""
    selected = rows or (_quote(),)
    return DataFrame(selected, columns=list(QUOTE_COLUMNS))


def _issue(
    *,
    product: DataFrame | None = None,
    events: DataFrame | None = None,
    quotes: DataFrame | None = None,
    evaluated_at: datetime = EVALUATED_AT,
):
    """Issue candidates from canonical fixtures."""
    return issue_pregame_candidates(
        product=_product() if product is None else product,
        forecast_events=_events() if events is None else events,
        quotes=_quotes() if quotes is None else quotes,
        evaluated_at=evaluated_at,
    )


def test_positive_ev_preserves_complete_candidate_evidence() -> None:
    """Positive EV freezes exact quote, product, forecast, and decision evidence."""
    result = _issue()
    assert result.product_id == "product-1"
    assert result.product_run_id == "run-1"
    assert result.product_generated_at == PRODUCT_GENERATED_AT
    assert result.season == "2026-2027"
    assert result.week == 1
    assert result.evaluated_at == EVALUATED_AT
    assert len(result.rows) == 1

    row = result.rows[0]
    assert row.state is CandidateIssuanceState.CANDIDATE
    assert row.reason is CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE
    assert row.provider == "the_odds_api"
    assert row.provider_event_id == "provider-event-1"
    assert row.sportsbook == "draftkings"
    assert row.game_id == GAME_ID
    assert (row.market, row.side) == ("moneyline", "home")
    assert row.line is None
    assert row.american_price == -110
    assert row.fetched_at == FETCHED_AT
    assert row.sportsbook_updated_at == UPDATED_AT
    assert row.kickoff == KICKOFF
    assert row.is_live is False
    assert row.forecast_event_id == "win-1"
    assert row.forecast_run_id == "run-1"
    assert row.forecast_role == "live"
    assert row.forecast_generated_at == PRODUCT_GENERATED_AT
    assert (row.model_name, row.model_type) == ("win_prob", "elo")
    assert row.model_probability == pytest.approx(0.60)
    assert row.expected_value is not None
    assert row.expected_value > 0.0


@pytest.mark.parametrize(
    ("side", "odds", "expected_ev"),
    [
        ("away", 120.0, -0.12),
        ("away", 150.0, 0.0),
    ],
)
def test_nonpositive_ev_is_not_candidate(
    side: str,
    odds: float,
    expected_ev: float,
) -> None:
    """Negative and break-even EV remain explicit non-candidates."""
    row = _issue(quotes=_quotes(_quote(side=side, odds=odds))).rows[0]
    assert row.state is CandidateIssuanceState.NOT_CANDIDATE
    assert row.reason is CandidateIssuanceReason.EXPECTED_VALUE_NOT_POSITIVE
    assert row.expected_value == pytest.approx(expected_ev, abs=1e-12)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"commence_time": None}, CandidateIssuanceReason.KICKOFF_UNAVAILABLE),
        ({"is_live": True}, CandidateIssuanceReason.QUOTE_LIVE),
        (
            {"fetched_at": KICKOFF},
            CandidateIssuanceReason.QUOTE_NOT_PREGAME,
        ),
        (
            {"fetched_at": KICKOFF + timedelta(seconds=1)},
            CandidateIssuanceReason.QUOTE_NOT_PREGAME,
        ),
    ],
)
def test_ineligible_quote_is_persisted_as_unavailable(
    overrides: dict[str, object],
    reason: CandidateIssuanceReason,
) -> None:
    """Unusable quote evidence remains visible without becoming a candidate."""
    row = _issue(quotes=_quotes(_quote(**overrides))).rows[0]
    assert row.state is CandidateIssuanceState.UNAVAILABLE
    assert row.reason is reason
    assert row.model_probability is None
    assert row.expected_value is None


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=1)])
def test_issuance_at_or_after_kickoff_is_rejected(offset: timedelta) -> None:
    """The invocation itself must occur strictly before every known kickoff."""
    with pytest.raises(ValueError, match="strictly before kickoff"):
        _issue(evaluated_at=KICKOFF + offset)


def test_missing_selected_forecast_is_unavailable() -> None:
    """The product reference is not reconstructed from another event."""
    row = _issue(events=_events(include_win=False)).rows[0]
    assert row.state is CandidateIssuanceState.UNAVAILABLE
    assert row.reason is CandidateIssuanceReason.FORECAST_EVENT_UNAVAILABLE
    assert row.forecast_event_id is None


def test_total_without_uncertainty_is_unavailable() -> None:
    """Missing uncertainty remains evidence, not a candidate-policy decision."""
    product = _product(
        total_status="uncertainty_unavailable",
        total_uncertainty=None,
        total_uncertainty_trained_at=None,
    )
    row = _issue(
        product=product,
        quotes=_quotes(_quote(market="total", side="over", line=44.0)),
    ).rows[0]
    assert row.state is CandidateIssuanceState.UNAVAILABLE
    assert row.reason is CandidateIssuanceReason.UNCERTAINTY_UNAVAILABLE
    assert row.forecast_event_id == "total-1"
    assert (row.model_name, row.model_type) == ("total", "xgboost")


def test_market_family_uses_exact_product_selected_event() -> None:
    """Moneyline, spread, and total preserve their exact selected provenance."""
    result = _issue(
        quotes=_quotes(
            _quote(market="moneyline", side="home", odds=-110.0, line=None),
            _quote(
                sportsbook="fanduel",
                market="spread",
                side="home",
                odds=-110.0,
                line=-3.0,
            ),
            _quote(
                sportsbook="bovada",
                market="total",
                side="over",
                odds=-110.0,
                line=44.0,
            ),
        )
    )
    by_market = {row.market: row for row in result.rows}
    assert by_market["moneyline"].forecast_event_id == "win-1"
    assert by_market["spread"].forecast_event_id == "win-1"
    assert by_market["total"].forecast_event_id == "total-1"


def test_duplicate_quote_observation_identity_is_rejected() -> None:
    """Duplicate immutable quote evidence cannot be silently collapsed."""
    quote = _quote()
    with pytest.raises(ValueError, match="duplicate quote observation"):
        _issue(quotes=_quotes(quote, quote.copy()))


def test_quote_scope_must_match_selected_product() -> None:
    """Issuance cannot mix a selected product with another weekly quote scope."""
    with pytest.raises(ValueError, match="selected product scope"):
        _issue(quotes=_quotes(_quote(week=2)))


def test_non_live_forecast_role_is_rejected() -> None:
    """A development or backfilled forecast can never back a real recommendation."""
    events = _events().assign(role="development")
    with pytest.raises(ValueError, match="exclusively live forecast events"):
        _issue(events=events)


def test_forecast_generated_after_evaluated_at_is_rejected() -> None:
    """Evaluating against a forecast generated after the fact is look-ahead leakage."""
    events = _events().assign(generated_at=EVALUATED_AT + timedelta(minutes=1))
    with pytest.raises(ValueError, match="generated after evaluated_at"):
        _issue(events=events)


def test_output_is_deterministic_and_inputs_are_not_mutated() -> None:
    """Input order cannot change persisted ordering or source frames."""
    product = _product()
    events = _events()
    quotes = _quotes(
        _quote(sportsbook="fanduel", odds=-105.0),
        _quote(sportsbook="draftkings", odds=-110.0),
    )
    original_product = product.copy(deep=True)
    original_events = events.copy(deep=True)
    original_quotes = quotes.copy(deep=True)

    first = _issue(product=product, events=events, quotes=quotes)
    reordered_quotes = cast(
        DataFrame,
        quotes.iloc[::-1].reset_index(drop=True),
    )
    second = _issue(
        product=product,
        events=events,
        quotes=reordered_quotes,
    )

    assert first == second
    assert [row.sportsbook for row in first.rows] == ["draftkings", "fanduel"]
    pd.testing.assert_frame_equal(product, original_product)
    pd.testing.assert_frame_equal(events, original_events)
    pd.testing.assert_frame_equal(quotes, original_quotes)
