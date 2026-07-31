# tests/unit/market/test_edge_result.py

"""Tests for recommendation rows paired with structured diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import (
    EdgeResult,
    build_edge_result,
)

SEASON = "2026-2027"
WEEK = 1
GAME_ID = "2026_01_KC_LAC"


def _predictions(
    *,
    season: str = SEASON,
    week: int = WEEK,
    game_id: str = GAME_ID,
    home_win_prob: float = 0.70,
    model_spread: float = -7.0,
    model_total: float = 52.0,
) -> DataFrame:
    return DataFrame(
        [
            {
                "season": season,
                "week": week,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "home_win_prob": home_win_prob,
                "model_spread": model_spread,
                "model_total": model_total,
                "model_name": "win_prob",
                "model_type": "random_forest",
                "win_event_id": "win-event",
                "total_event_id": "total-event",
                "product_id": "weekly-product",
            }
        ]
    )


def _markets(
    *,
    season: str = SEASON,
    week: int = WEEK,
    game_id: str = GAME_ID,
    fetched_at: datetime | None = None,
) -> DataFrame:
    timestamp = fetched_at or datetime(2026, 9, 5, 12, tzinfo=UTC)
    base: dict[str, object] = {
        "season": season,
        "week": week,
        "game_id": game_id,
        "game_date": "2026-09-05",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "sportsbook": "nflverse_schedule",
        "fetched_at": timestamp,
    }
    return DataFrame(
        [
            {**base, "market": "moneyline", "side": "home", "odds": -150.0, "line": None},
            {**base, "market": "moneyline", "side": "away", "odds": 130.0, "line": None},
            {**base, "market": "spread", "side": "home", "odds": -110.0, "line": -3.0},
            {**base, "market": "spread", "side": "away", "odds": -110.0, "line": 3.0},
            {**base, "market": "total", "side": "over", "odds": -110.0, "line": 45.0},
            {**base, "market": "total", "side": "under", "odds": -110.0, "line": 45.0},
        ]
    )


def _result(
    predictions: DataFrame | None = None,
    markets: DataFrame | None = None,
    **kwargs: object,
) -> EdgeResult:
    return build_edge_result(
        _predictions() if predictions is None else predictions,
        _markets() if markets is None else markets,
        season=SEASON,
        week=WEEK,
        margin_std=13.0,
        total_std=13.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_positive_result_pairs_rows_and_diagnostics() -> None:
    result = _result()

    assert not result.rows.empty
    assert result.diagnostics.state is EdgeResultState.POSITIVE_EDGES
    assert result.diagnostics.filtered_edge_count == len(result.rows)
    assert result.diagnostics.calculated_edge_count >= len(result.rows)
    assert result.diagnostics.eligible_market_count == 3
    assert result.diagnostics.provenance.win_event_ids == ("win-event",)
    assert result.diagnostics.provenance.total_event_ids == ("total-event",)
    assert result.diagnostics.provenance.product_ids == ("weekly-product",)
    assert result.diagnostics.provenance.market_sources == ("nflverse_schedule",)


def test_no_predictions_returns_explicit_empty_reason() -> None:
    result = _result(predictions=DataFrame())

    assert result.rows.empty
    assert result.diagnostics.state is EdgeResultState.BLOCKED
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.NO_PREDICTIONS,)


def test_no_market_data_returns_explicit_empty_reason() -> None:
    result = _result(markets=DataFrame())

    assert result.rows.empty
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.NO_MARKET_DATA,)


def test_wrong_scope_market_data_returns_explicit_empty_reason() -> None:
    result = _result(markets=_markets(week=2))

    assert result.rows.empty
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE,)


def test_zero_match_returns_explicit_empty_reason() -> None:
    result = _result(markets=_markets(game_id="2026_01_BUF_MIA"))

    assert result.rows.empty
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES,)


def test_incomplete_market_result_retains_reason() -> None:
    markets = _markets()
    markets.loc[
        (markets["market"] == "total") & (markets["side"] == "under"),
        "odds",
    ] = None

    result = _result(markets=markets)

    assert result.diagnostics.state is EdgeResultState.BLOCKED
    assert EdgeDiagnosticBlocker.INCOMPLETE_MARKETS in (result.diagnostics.blockers)
    assert result.diagnostics.complete_total_count == 0


def test_no_positive_edges_is_explicit() -> None:
    markets = _markets()
    markets.loc[
        markets["market"] == "moneyline",
        "odds",
    ] = -110.0

    result = _result(
        predictions=_predictions(
            home_win_prob=0.50,
            model_spread=-3.0,
            model_total=45.0,
        ),
        markets=markets,
    )

    assert result.rows.empty
    assert result.diagnostics.state is (EdgeResultState.NO_CALCULABLE_EDGES)
    assert result.diagnostics.calculated_edge_count == 0
    assert result.diagnostics.positive_edge_count == 0
    assert result.diagnostics.filtered_edge_count == 0
    assert not result.diagnostics.blockers


def test_rejects_negative_minimum_ev() -> None:
    with pytest.raises(ValueError, match="min_ev"):
        _result(min_ev=-0.01)


def test_custom_threshold_can_empty_rows_without_losing_positive_count() -> None:
    result = _result(min_ev=1.0)

    assert result.rows.empty
    assert result.diagnostics.state is EdgeResultState.POSITIVE_EDGES
    assert result.diagnostics.positive_edge_count > 0
    assert result.diagnostics.filtered_edge_count == 0


def test_stale_market_blocker_flows_through_result() -> None:
    result = _result(
        markets=_markets(
            fetched_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
        ),
        as_of=datetime(2026, 9, 7, 12, tzinfo=UTC),
        max_market_age=timedelta(hours=24),
    )

    assert EdgeDiagnosticBlocker.MARKET_STALE in result.diagnostics.blockers


def test_inputs_are_not_mutated() -> None:
    predictions = _predictions()
    markets = _markets()
    expected_predictions = predictions.copy(deep=True)
    expected_markets = markets.copy(deep=True)

    _result(predictions=predictions, markets=markets)

    pd.testing.assert_frame_equal(predictions, expected_predictions)
    pd.testing.assert_frame_equal(markets, expected_markets)


def test_result_contract_is_frozen() -> None:
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.rows = DataFrame()  # type: ignore[misc]


def test_result_rejects_row_count_mismatch() -> None:
    diagnostics = EdgeDiagnostics(
        season=SEASON,
        week=WEEK,
        prediction_game_count=1,
        market_game_count=1,
        matched_game_count=1,
        complete_moneyline_count=1,
        complete_spread_count=1,
        complete_total_count=1,
        eligible_market_count=3,
        calculated_edge_count=1,
        positive_edge_count=1,
        filtered_edge_count=1,
        state=EdgeResultState.POSITIVE_EDGES,
    )
    with pytest.raises(ValueError, match="filtered_edge_count"):
        EdgeResult(rows=DataFrame(), diagnostics=diagnostics)
