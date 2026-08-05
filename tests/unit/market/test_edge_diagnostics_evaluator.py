# tests/unit/market/test_edge_diagnostics_evaluator.py

"""Tests for pure weekly edge diagnostic evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeResultState,
    evaluate_edge_diagnostics,
)

SEASON = "2026-2027"
WEEK = 1
GAME_ID = "2026_01_KC_LAC"


def _predictions(**overrides: object) -> DataFrame:
    row: dict[str, object] = {
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_ID,
        "home_win_prob": 0.62,
        "model_spread": -3.5,
        "model_total": 46.5,
        "win_event_id": "win-event",
        "win_run_id": "win-run",
        "win_model_name": "win_prob",
        "win_model_type": "random_forest",
        "total_event_id": "total-event",
        "total_run_id": "total-run",
        "total_model_name": "total",
        "total_model_type": "random_forest",
        "product_id": "weekly-product",
        "product_run_id": "product-run",
    }
    row.update(overrides)
    return DataFrame([row])


def _markets(
    *,
    fetched_at: datetime | None = None,
    source: str = "nflverse",
) -> DataFrame:
    timestamp = fetched_at or datetime(2026, 9, 5, 12, tzinfo=UTC)
    base: dict[str, object] = {
        "fetched_at": timestamp,
        "provider": source,
        "sportsbook": None,
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_ID,
    }
    return DataFrame(
        [
            {**base, "market": "moneyline", "side": "home", "odds": -120.0, "line": None},
            {**base, "market": "moneyline", "side": "away", "odds": 105.0, "line": None},
            {**base, "market": "spread", "side": "home", "odds": -110.0, "line": -2.5},
            {**base, "market": "spread", "side": "away", "odds": -110.0, "line": 2.5},
            {**base, "market": "total", "side": "over", "odds": -110.0, "line": 45.5},
            {**base, "market": "total", "side": "under", "odds": -110.0, "line": 45.5},
        ]
    )


def _edges(*evs: float) -> DataFrame:
    return DataFrame(
        [
            {
                "season": SEASON,
                "week": WEEK,
                "ev": ev,
            }
            for ev in evs
        ]
    )


def _evaluate(
    *,
    predictions: DataFrame | None = None,
    markets: DataFrame | None = None,
    calculated: DataFrame | None = None,
    filtered: DataFrame | None = None,
    **kwargs: object,
):
    return evaluate_edge_diagnostics(
        _predictions() if predictions is None else predictions,
        _markets() if markets is None else markets,
        _edges(0.08, -0.01) if calculated is None else calculated,
        _edges(0.08) if filtered is None else filtered,
        season=SEASON,
        week=WEEK,
        **kwargs,  # type: ignore[arg-type]
    )


def test_no_predictions_is_blocked() -> None:
    result = _evaluate(predictions=DataFrame())
    assert result.state is EdgeResultState.BLOCKED
    assert result.blockers == (EdgeDiagnosticBlocker.NO_PREDICTIONS,)
    assert result.prediction_game_count == 0


def test_no_market_data_is_blocked() -> None:
    result = _evaluate(markets=DataFrame())
    assert result.state is EdgeResultState.BLOCKED
    assert result.blockers == (EdgeDiagnosticBlocker.NO_MARKET_DATA,)
    assert result.market_game_count == 0


def test_simultaneous_input_gaps_retain_both_blockers() -> None:
    result = _evaluate(predictions=DataFrame(), markets=DataFrame())
    assert result.blockers == (
        EdgeDiagnosticBlocker.NO_PREDICTIONS,
        EdgeDiagnosticBlocker.NO_MARKET_DATA,
    )


def test_wrong_scope_market_data_is_distinct_from_missing() -> None:
    markets = _markets()
    markets["week"] = 2
    result = _evaluate(markets=markets)
    assert result.blockers == (EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE,)
    assert result.market_game_count == 0


def test_zero_matched_games_is_blocked() -> None:
    markets = _markets()
    markets["game_id"] = "2026_01_BUF_MIA"
    result = _evaluate(markets=markets)
    assert result.blockers == (EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES,)
    assert result.prediction_game_count == 1
    assert result.market_game_count == 1
    assert result.matched_game_count == 0


def test_stale_markets_use_explicit_policy() -> None:
    result = _evaluate(
        markets=_markets(
            fetched_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
        ),
        as_of=datetime(2026, 9, 7, 12, tzinfo=UTC),
        max_market_age=timedelta(hours=24),
    )
    assert EdgeDiagnosticBlocker.MARKET_STALE in result.blockers


def test_market_age_is_not_inferred_without_policy() -> None:
    result = _evaluate(
        markets=_markets(
            fetched_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
    )
    assert EdgeDiagnosticBlocker.MARKET_STALE not in result.blockers


def test_freshness_policy_requires_both_inputs() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _evaluate(as_of=datetime(2026, 9, 7, tzinfo=UTC))
    with pytest.raises(ValueError, match="must be provided together"):
        _evaluate(max_market_age=timedelta(hours=1))


def test_freshness_reference_requires_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _evaluate(
            as_of=datetime(2026, 9, 7),
            max_market_age=timedelta(hours=1),
        )


def test_incomplete_markets_are_explicit() -> None:
    markets = _markets()
    markets.loc[
        (markets["market"] == "total") & (markets["side"] == "under"),
        "odds",
    ] = None
    result = _evaluate(markets=markets)
    assert result.blockers == (EdgeDiagnosticBlocker.INCOMPLETE_MARKETS,)
    assert result.complete_moneyline_count == 1
    assert result.complete_spread_count == 1
    assert result.complete_total_count == 0
    assert result.eligible_market_count == 2


def test_no_calculable_edges_is_explicit() -> None:
    result = _evaluate(calculated=DataFrame(), filtered=DataFrame())
    assert result.state is EdgeResultState.NO_CALCULABLE_EDGES
    assert result.calculated_edge_count == 0


def test_no_positive_edges_is_explicit() -> None:
    result = _evaluate(
        calculated=_edges(-0.01, 0.0),
        filtered=DataFrame(),
    )
    assert result.state is EdgeResultState.NO_POSITIVE_EDGES
    assert result.calculated_edge_count == 2
    assert result.positive_edge_count == 0


def test_positive_edges_and_counts_come_from_inputs() -> None:
    result = _evaluate(
        calculated=_edges(0.08, 0.03, -0.02),
        filtered=_edges(0.08),
    )
    assert result.state is EdgeResultState.POSITIVE_EDGES
    assert result.prediction_game_count == 1
    assert result.market_game_count == 1
    assert result.matched_game_count == 1
    assert result.complete_moneyline_count == 1
    assert result.complete_spread_count == 1
    assert result.complete_total_count == 1
    assert result.eligible_market_count == 3
    assert result.calculated_edge_count == 3
    assert result.positive_edge_count == 2
    assert result.filtered_edge_count == 1


def test_counts_distinct_games_not_duplicate_rows() -> None:
    predictions = pd.concat([_predictions(), _predictions()], ignore_index=True)
    markets = pd.concat([_markets(), _markets()], ignore_index=True)
    result = _evaluate(predictions=predictions, markets=markets)
    assert result.prediction_game_count == 1
    assert result.market_game_count == 1
    assert result.matched_game_count == 1


def test_provenance_retains_win_total_product_and_market_values() -> None:
    predictions = pd.concat(
        [
            _predictions(
                win_event_id="win-b",
                total_event_id="total-b",
                product_id="product-b",
            ),
            _predictions(
                win_event_id="win-a",
                total_event_id="total-a",
                product_id="product-a",
            ),
        ],
        ignore_index=True,
    )
    markets = pd.concat(
        [
            _markets(
                source="source-b",
                fetched_at=datetime(2026, 9, 5, 13, tzinfo=UTC),
            ),
            _markets(
                source="source-a",
                fetched_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
            ),
        ],
        ignore_index=True,
    )
    result = _evaluate(predictions=predictions, markets=markets)
    assert result.provenance.win_event_ids == ("win-a", "win-b")
    assert result.provenance.total_event_ids == ("total-a", "total-b")
    assert result.provenance.product_ids == ("product-a", "product-b")
    assert result.provenance.market_providers == ("source-a", "source-b")
    assert result.provenance.market_fetched_at == (
        datetime(2026, 9, 5, 12, tzinfo=UTC),
        datetime(2026, 9, 5, 13, tzinfo=UTC),
    )


def test_inputs_are_not_mutated() -> None:
    predictions = _predictions()
    markets = _markets()
    calculated = _edges(0.08)
    filtered = _edges(0.08)
    expected_predictions = predictions.copy(deep=True)
    expected_markets = markets.copy(deep=True)
    expected_calculated = calculated.copy(deep=True)
    expected_filtered = filtered.copy(deep=True)
    evaluate_edge_diagnostics(
        predictions,
        markets,
        calculated,
        filtered,
        season=SEASON,
        week=WEEK,
    )
    pd.testing.assert_frame_equal(predictions, expected_predictions)
    pd.testing.assert_frame_equal(markets, expected_markets)
    pd.testing.assert_frame_equal(calculated, expected_calculated)
    pd.testing.assert_frame_equal(filtered, expected_filtered)
