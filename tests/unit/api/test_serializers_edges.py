"""Tests for /edges serializers."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.api.schemas.edges import EdgeList, EdgeRow
from gridiron_edge.api.serializers.edges import (
    _none_if_nan,
    _row_to_edge,
    _serialize_diagnostics,
    serialize_edges_list,
)
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeProvenance,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import EdgeResult


def _valid_row() -> dict:
    """Return one canonical service recommendation row."""
    return {
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-05",
        "season": "2026-2027",
        "week": 1,
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "model_key": "win_prob_random_forest",
        "confidence_tier": "High",
        "market_type": "moneyline",
        "side": "away",
        "model_value": 0.30,
        "market_value": 0.37,
        "american_odds": -110,
        "point_edge": float("nan"),
        "cover_prob": float("nan"),
        "ev": 0.045,
        "edge_strength": "moderate",
        "kelly_frac": 0.023,
        "kelly_stake": 5.75,
    }


def _result(rows: pd.DataFrame | None = None) -> EdgeResult:
    provenance = EdgeProvenance(
        win_event_ids=("win-event-1",),
        win_run_ids=("win-run-1",),
        win_model_names=("win_prob",),
        win_model_types=("random_forest",),
        total_event_ids=("total-event-1",),
        total_run_ids=("total-run-1",),
        total_model_names=("total",),
        total_model_types=("xgboost",),
        product_ids=("weekly-product-1",),
        product_run_ids=("weekly-run-1",),
        market_sources=("nflverse_schedule",),
        market_fetched_at=(datetime(2026, 9, 5, 12, tzinfo=UTC),),
    )
    frame = pd.DataFrame([_valid_row()]) if rows is None else rows
    diagnostics = EdgeDiagnostics(
        season="2026-2027",
        week=1,
        prediction_game_count=1,
        market_game_count=1,
        matched_game_count=1,
        complete_moneyline_count=1,
        complete_spread_count=0,
        complete_total_count=0,
        eligible_market_count=1,
        calculated_edge_count=1,
        positive_edge_count=1,
        filtered_edge_count=len(frame),
        state=EdgeResultState.POSITIVE_EDGES,
        provenance=provenance,
    )
    return EdgeResult(rows=frame, diagnostics=diagnostics)


class TestNoneIfNan:
    def test_none_and_nan_return_none(self) -> None:
        assert _none_if_nan(None) is None
        assert _none_if_nan(float("nan")) is None
        assert _none_if_nan(np.nan) is None

    @pytest.mark.parametrize("value", [0.0, -3.5, "KC"])
    def test_nonmissing_value_is_preserved(self, value: object) -> None:
        assert _none_if_nan(value) == value


class TestRowToEdge:
    def test_preserves_service_row_and_normalizes_nan(self) -> None:
        row: EdgeRow = _row_to_edge(_valid_row())
        assert row.game_id == "2026_01_KC_LAC"
        assert row.away_team == "Kansas City Chiefs"
        assert row.home_team == "Los Angeles Chargers"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.edge_strength == "moderate"


class TestSerializeDiagnostics:
    def test_preserves_counts_state_and_complete_provenance(self) -> None:
        diagnostics = _serialize_diagnostics(_result())
        assert diagnostics.state is EdgeResultState.POSITIVE_EDGES
        assert diagnostics.filtered_edge_count == 1
        assert diagnostics.provenance.win_event_ids == ("win-event-1",)
        assert diagnostics.provenance.total_model_types == ("xgboost",)
        assert diagnostics.provenance.product_ids == ("weekly-product-1",)
        assert diagnostics.provenance.market_sources == ("nflverse_schedule",)
        assert diagnostics.provenance.market_fetched_at == (datetime(2026, 9, 5, 12, tzinfo=UTC),)

    def test_preserves_every_blocker_without_collapse(self) -> None:
        blockers = tuple(EdgeDiagnosticBlocker)
        result = EdgeResult(
            rows=pd.DataFrame(),
            diagnostics=EdgeDiagnostics(
                season="2026-2027",
                week=1,
                prediction_game_count=0,
                market_game_count=0,
                matched_game_count=0,
                complete_moneyline_count=0,
                complete_spread_count=0,
                complete_total_count=0,
                eligible_market_count=0,
                calculated_edge_count=0,
                positive_edge_count=0,
                filtered_edge_count=0,
                state=EdgeResultState.BLOCKED,
                blockers=blockers,
            ),
        )
        assert _serialize_diagnostics(result).blockers == blockers


class TestSerializeEdgesList:
    def test_serializes_complete_result(self) -> None:
        response: EdgeList = serialize_edges_list(
            _result(),
            min_ev=0.0,
            bankroll=2500.0,
            kelly_multiplier=0.1,
        )
        assert response.total == 1
        assert response.season == "2026-2027"
        assert response.week == 1
        assert response.items[0].away_team == "Kansas City Chiefs"
        assert response.diagnostics.filtered_edge_count == 1
        assert response.diagnostics.provenance.market_sources == ("nflverse_schedule",)

    def test_scope_comes_from_service_diagnostics(self) -> None:
        response = serialize_edges_list(
            _result(),
            min_ev=None,
            bankroll=None,
            kelly_multiplier=None,
        )
        assert response.season == "2026-2027"
        assert response.week == 1
        assert response.min_ev is None
        assert response.bankroll is None
        assert response.kelly_multiplier is None
