# tests/unit/api/test_serializers_edges.py

"""Tests for /edges serializers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.api.schemas.edges import EdgeList, EdgeRow
from gridiron_edge.api.serializers.edges import (
    _none_if_nan,
    _row_to_edge,
    serialize_edges_list,
)


def _valid_row() -> dict:
    """A canonical valid loader row for reuse across tests."""
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
        "point_edge": float("nan"),
        "cover_prob": float("nan"),
        "ev": 0.045,
        "edge_strength": "moderate",
        "kelly_frac": 0.023,
        "kelly_stake": 5.75,
    }


class TestNoneIfNan:
    def test_none_returns_none(self) -> None:
        assert _none_if_nan(None) is None

    def test_nan_returns_none(self) -> None:
        assert _none_if_nan(float("nan")) is None
        assert _none_if_nan(np.nan) is None

    def test_zero_preserved(self) -> None:
        assert _none_if_nan(0.0) == 0.0

    def test_negative_preserved(self) -> None:
        assert _none_if_nan(-3.5) == -3.5

    def test_string_preserved(self) -> None:
        assert _none_if_nan("KC") == "KC"


class TestRowToEdge:
    def test_moneyline_row(self) -> None:
        row: EdgeRow = _row_to_edge(_valid_row())
        assert isinstance(row, EdgeRow)
        assert row.game_id == "2026_01_KC_LAC"
        assert row.market_type == "moneyline"
        assert row.point_edge is None
        assert row.cover_prob is None
        assert row.ev == 0.045
        assert row.market_value == pytest.approx(0.37)
        assert row.american_odds == -110

    def test_spread_row(self) -> None:
        base: dict = _valid_row()
        base.update(
            {
                "market_type": "spread",
                "side": "home",
                "model_value": -7.0,
                "market_value": -3.5,
                "american_odds": -108,
                "point_edge": -3.5,
                "cover_prob": 0.62,
            }
        )
        row: EdgeRow = _row_to_edge(base)
        assert row.market_type == "spread"
        assert row.point_edge == -3.5
        assert row.cover_prob == 0.62
        assert row.market_value == -3.5
        assert row.american_odds == -108

    def test_total_row(self) -> None:
        base: dict = _valid_row()
        base.update(
            {
                "market_type": "total",
                "side": "over",
                "model_value": 50.0,
                "market_value": 44.0,
                "american_odds": 102,
                "point_edge": 6.0,
                "cover_prob": 0.68,
            }
        )
        row: EdgeRow = _row_to_edge(base)
        assert row.market_type == "total"
        assert row.point_edge == 6.0
        assert row.cover_prob == 0.68
        assert row.market_value == 44.0
        assert row.american_odds == 102

    def test_nan_kelly_becomes_none(self) -> None:
        base: dict = _valid_row()
        base["kelly_frac"] = float("nan")
        base["kelly_stake"] = float("nan")
        row: EdgeRow = _row_to_edge(base)
        assert row.kelly_frac is None
        assert row.kelly_stake is None

    def test_nan_confidence_tier_becomes_none(self) -> None:
        base: dict = _valid_row()
        base["confidence_tier"] = float("nan")
        row: EdgeRow = _row_to_edge(base)
        assert row.confidence_tier is None


class TestSerializeEdgesList:
    def test_empty_dataframe_returns_empty_list(self) -> None:
        response: EdgeList = serialize_edges_list(
            pd.DataFrame(),
            season="2026-2027",
            week=1,
            min_ev=0.0,
            bankroll=2500.0,
            kelly_multiplier=0.1,
        )
        assert isinstance(response, EdgeList)
        assert response.items == []
        assert response.total == 0
        assert response.season == "2026-2027"
        assert response.week == 1
        assert response.min_ev == 0.0
        assert response.bankroll == 2500.0
        assert response.kelly_multiplier == 0.1

    def test_dataframe_of_rows_serializes_each(self) -> None:
        df = pd.DataFrame(
            [
                _valid_row(),
                {**_valid_row(), "game_id": "2026_01_BUF_MIA"},
                {**_valid_row(), "game_id": "2026_01_PHI_DAL"},
            ]
        )
        response: EdgeList = serialize_edges_list(
            df,
            season="2026-2027",
            week=1,
            min_ev=0.0,
            bankroll=2500.0,
            kelly_multiplier=0.1,
        )
        assert len(response.items) == 3
        assert response.total == 3
        game_ids: list[str] = [item.game_id for item in response.items]
        assert game_ids == [
            "2026_01_KC_LAC",
            "2026_01_BUF_MIA",
            "2026_01_PHI_DAL",
        ]
        assert response.bankroll == 2500.0
        assert response.kelly_multiplier == 0.1
        assert response.items[0].american_odds == -110

    def test_none_filter_params_pass_through(self) -> None:
        response: EdgeList = serialize_edges_list(
            pd.DataFrame(),
            season=None,
            week=None,
            min_ev=None,
            bankroll=None,
            kelly_multiplier=None,
        )
        assert response.season is None
        assert response.week is None
        assert response.min_ev is None
        assert response.bankroll is None
        assert response.kelly_multiplier is None

    def test_min_ev_zero_distinct_from_none(self) -> None:
        response: EdgeList = serialize_edges_list(
            pd.DataFrame(),
            season="2026-2027",
            week=1,
            min_ev=0.0,
            bankroll=0.0,
            kelly_multiplier=0.0,
        )
        # 0.0 is a valid filter value, not "unspecified".
        assert response.min_ev == 0.0
        assert response.min_ev == 0.0
        assert response.bankroll == 0.0
        assert response.kelly_multiplier == 0.0
