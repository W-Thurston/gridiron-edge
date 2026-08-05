# tests/unit/api/test_schemas_edges.py

"""Tests for /edges response schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.edges import (
    EdgeDiagnosticsResponse,
    EdgeList,
    EdgeProvenanceResponse,
    EdgeRow,
)
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeResultState,
)


def _valid_edge_row() -> dict:
    return {
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-05",
        "season": "2026-2027",
        "week": 1,
        "away_team": "KC",
        "home_team": "LAC",
        "model_key": "win_prob_random_forest",
        "confidence_tier": "High",
        "market_type": "moneyline",
        "side": "away",
        "model_value": 0.30,
        "market_value": 0.37,
        "american_odds": -110,
        "point_edge": None,
        "cover_prob": None,
        "ev": 0.045,
        "edge_strength": "moderate",
        "kelly_frac": 0.023,
        "kelly_stake": 5.75,
    }


class TestEdgeRow:
    def test_minimum_shape(self) -> None:
        row = EdgeRow(
            game_id="2026_01_KC_LAC",
            away_team="KC",
            home_team="LAC",
            model_key="win_prob_elo",
            market_type="moneyline",
            side="away",
            american_odds=125,
            ev=0.02,
            edge_strength="lean",
        )
        assert row.game_id == "2026_01_KC_LAC"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.kelly_stake is None
        assert row.american_odds == 125

    def test_full_moneyline_row(self) -> None:
        row = EdgeRow(**_valid_edge_row())
        assert row.market_type == "moneyline"
        assert row.side == "away"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.ev == 0.045
        assert row.edge_strength == "moderate"
        assert row.american_odds == -110

    def test_full_spread_row(self) -> None:
        row = EdgeRow(
            **{
                **_valid_edge_row(),
                "market_type": "spread",
                "side": "home",
                "model_value": -7.0,
                "market_value": -3.5,
                "american_odds": -108,
                "point_edge": -3.5,
                "cover_prob": 0.62,
            }
        )
        assert row.market_type == "spread"
        assert row.point_edge == -3.5
        assert row.cover_prob == 0.62
        assert row.market_value == -3.5
        assert row.american_odds == -108

    def test_full_total_row(self) -> None:
        row = EdgeRow(
            **{
                **_valid_edge_row(),
                "market_type": "total",
                "side": "over",
                "model_value": 50.0,
                "market_value": 44.0,
                "american_odds": 102,
                "point_edge": 6.0,
                "cover_prob": 0.68,
            }
        )
        assert row.market_type == "total"
        assert row.point_edge == 6.0
        assert row.cover_prob == 0.68
        assert row.market_value == 44.0
        assert row.american_odds == 102

    @pytest.mark.parametrize(
        "edge_strength",
        ["strong", "moderate", "lean", "no_edge"],
    )
    def test_accepts_implementation_edge_strengths(
        self,
        edge_strength: str,
    ) -> None:
        row = EdgeRow(
            **{
                **_valid_edge_row(),
                "edge_strength": edge_strength,
            }
        )
        assert row.edge_strength == edge_strength

    def test_rejects_retired_weak_edge_strength(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRow(
                **{
                    **_valid_edge_row(),
                    "edge_strength": "weak",
                }
            )

    def test_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRow(game_id="2026_01_KC_LAC")  # type: ignore[call-arg]

    def test_rejects_missing_ev(self) -> None:
        row_data = _valid_edge_row()
        del row_data["ev"]
        with pytest.raises(ValidationError):
            EdgeRow(**row_data)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            EdgeRow(
                **{
                    **_valid_edge_row(),
                    "surprise_field": "oops",
                }
            )  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        row = EdgeRow(**_valid_edge_row())
        with pytest.raises(ValidationError):
            row.ev = 0.99  # type: ignore[misc]


class TestEdgeDiagnosticsResponse:
    def test_preserves_complete_service_contract(self) -> None:
        fetched_at = datetime(2026, 9, 5, 12, tzinfo=UTC)
        provenance = EdgeProvenanceResponse(
            win_event_ids=("win-event-1",),
            win_run_ids=("win-run-1",),
            win_model_names=("win_prob",),
            win_model_types=("elo",),
            total_event_ids=("total-event-1",),
            total_run_ids=("total-run-1",),
            total_model_names=("total",),
            total_model_types=("xgboost",),
            product_ids=("weekly-product-1",),
            product_run_ids=("weekly-run-1",),
            market_sources=("nflverse_schedule",),
            market_fetched_at=(fetched_at,),
        )
        diagnostics = EdgeDiagnosticsResponse(
            season="2026-2027",
            week=1,
            prediction_game_count=1,
            market_game_count=1,
            matched_game_count=1,
            complete_moneyline_count=1,
            complete_spread_count=1,
            complete_total_count=1,
            eligible_market_count=3,
            calculated_edge_count=3,
            positive_edge_count=2,
            filtered_edge_count=1,
            state=EdgeResultState.POSITIVE_EDGES,
            blockers=(),
            provenance=provenance,
        )

        assert diagnostics.state is EdgeResultState.POSITIVE_EDGES
        assert diagnostics.provenance.win_event_ids == ("win-event-1",)
        assert diagnostics.provenance.total_model_types == ("xgboost",)
        assert diagnostics.provenance.market_sources == ("nflverse_schedule",)
        assert diagnostics.provenance.market_fetched_at == (fetched_at,)

    @pytest.mark.parametrize("blocker", list(EdgeDiagnosticBlocker))
    def test_accepts_every_service_blocker(
        self,
        blocker: EdgeDiagnosticBlocker,
    ) -> None:
        diagnostics = EdgeDiagnosticsResponse(
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
            blockers=(blocker,),
        )
        assert diagnostics.blockers == (blocker,)

    @pytest.mark.parametrize("state", list(EdgeResultState))
    def test_accepts_every_service_result_state(
        self,
        state: EdgeResultState,
    ) -> None:
        diagnostics = EdgeDiagnosticsResponse(
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
            state=state,
        )
        assert diagnostics.state is state


def _edge_list_diagnostics() -> EdgeDiagnosticsResponse:
    """Return the required diagnostics for direct EdgeList construction."""
    return EdgeDiagnosticsResponse(
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
        state=EdgeResultState.NO_CALCULABLE_EDGES,
    )


class TestEdgeList:
    def test_empty_list(self) -> None:
        response = EdgeList(diagnostics=_edge_list_diagnostics())
        assert response.items == []
        assert response.total is None
        assert response.season is None
        assert response.week is None
        assert response.min_ev is None
        assert response.response_meta is None
        assert response.bankroll is None
        assert response.kelly_multiplier is None
        assert response.diagnostics == _edge_list_diagnostics()

    def test_with_edges(self) -> None:
        response = EdgeList(
            items=[
                EdgeRow(**_valid_edge_row()),
                EdgeRow(
                    **{
                        **_valid_edge_row(),
                        "game_id": "2026_01_BUF_MIA",
                    }
                ),
            ],
            season="2026-2027",
            week=1,
            min_ev=0.02,
            bankroll=2500.0,
            kelly_multiplier=0.1,
            diagnostics=_edge_list_diagnostics(),
        )
        assert len(response.items) == 2
        assert response.season == "2026-2027"
        assert response.min_ev == 0.02
        assert response.bankroll == 2500.0
        assert response.kelly_multiplier == 0.1

    def test_min_ev_zero_preserved(self) -> None:
        response = EdgeList(
            items=[],
            min_ev=0.0,
            diagnostics=_edge_list_diagnostics(),
        )
        # 0.0 is a valid value, not "unspecified".
        assert response.min_ev == 0.0
