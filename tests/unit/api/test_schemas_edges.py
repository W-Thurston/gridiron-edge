# tests/unit/api/test_schemas_edges.py

"""Tests for /edges response schemas (W8 Tier 2 Step 6b)."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.edges import EdgeList, EdgeRow


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
            ev=0.02,
            edge_strength="weak",
        )
        assert row.game_id == "2026_01_KC_LAC"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.kelly_stake is None

    def test_full_moneyline_row(self) -> None:
        row = EdgeRow(**_valid_edge_row())
        assert row.market_type == "moneyline"
        assert row.side == "away"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.ev == 0.045
        assert row.edge_strength == "moderate"

    def test_full_spread_row(self) -> None:
        row = EdgeRow(
            **{
                **_valid_edge_row(),
                "market_type": "spread",
                "side": "home",
                "model_value": -7.0,
                "market_value": -3.5,
                "point_edge": -3.5,
                "cover_prob": 0.62,
            }
        )
        assert row.market_type == "spread"
        assert row.point_edge == -3.5
        assert row.cover_prob == 0.62

    def test_full_total_row(self) -> None:
        row = EdgeRow(
            **{
                **_valid_edge_row(),
                "market_type": "total",
                "side": "over",
                "model_value": 50.0,
                "market_value": 44.0,
                "point_edge": 6.0,
                "cover_prob": 0.68,
            }
        )
        assert row.market_type == "total"
        assert row.point_edge == 6.0
        assert row.cover_prob == 0.68

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


class TestEdgeList:
    def test_empty_list(self) -> None:
        response = EdgeList()
        assert response.items == []
        assert response.total is None
        assert response.season is None
        assert response.week is None
        assert response.min_ev is None
        assert response.response_meta is None

    def test_with_edges(self) -> None:
        response = EdgeList(
            items=[
                EdgeRow(**_valid_edge_row()),
                EdgeRow(**{**_valid_edge_row(), "game_id": "2026_01_BUF_MIA"}),
            ],
            season="2026-2027",
            week=1,
            min_ev=0.02,
        )
        assert len(response.items) == 2
        assert response.season == "2026-2027"
        assert response.min_ev == 0.02

    def test_min_ev_zero_preserved(self) -> None:
        response = EdgeList(items=[], min_ev=0.0)
        # 0.0 is a valid value, not "unspecified".
        assert response.min_ev == 0.0
