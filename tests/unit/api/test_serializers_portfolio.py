# tests/unit/api/test_serializers_portfolio.py

"""Unit tests for portfolio serializers."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.serializers.portfolio import (
    _compose_streak_label,
    serialize_bankroll_curve,
    serialize_bets,
    serialize_portfolio_summary,
    serialize_splits,
    serialize_transactions,
)


class TestPortfolioSummary:
    def test_empty_ledger(self) -> None:
        result = serialize_portfolio_summary(pd.DataFrame(), 500.0, {})
        assert result.bankroll == 500.0
        assert result.total_bets == 0
        assert result.settled_bets == 0
        assert result.open_bets == 0

    def test_populated(self) -> None:
        bets = pd.DataFrame({"status": ["open", "won", "lost"]})
        perf = {
            "wins": 1,
            "losses": 1,
            "pushes": 0,
            "win_pct": 0.5,
            "roi_pct": 5.2,
            "total_staked": 300.0,
            "total_pnl": 15.5,
            "mean_clv": 0.02,
            "pct_positive_clv": 0.6,
            "n_clv_bets": 2,
            "mean_ev_at_bet": 0.03,
            "ev_vs_actual_gap": 0.01,
            "n_model_bets": 2,
            "calibration_health": "healthy",
            "current_streak": 1,
            "current_streak_type": "win",
            "longest_win_streak": 3,
            "longest_loss_streak": 2,
        }
        result = serialize_portfolio_summary(bets, 750.0, perf)
        assert result.bankroll == 750.0
        assert result.total_bets == 3
        assert result.settled_bets == 2
        assert result.open_bets == 1
        assert result.wins == 1
        assert result.roi_pct == 5.2
        assert result.current_streak == "W1"
        assert result.longest_win_streak == 3


class TestComposeStreakLabel:
    """Unit tests for the streak-label composition helper."""

    def test_win_streak(self) -> None:
        assert _compose_streak_label(3, "win") == "W3"

    def test_loss_streak(self) -> None:
        assert _compose_streak_label(2, "loss") == "L2"

    def test_zero_streak_returns_none(self) -> None:
        assert _compose_streak_label(0, "none") is None

    def test_none_count(self) -> None:
        assert _compose_streak_label(None, "win") is None

    def test_unknown_type(self) -> None:
        assert _compose_streak_label(3, "mystery") is None


class TestSerializeBets:
    def test_empty(self) -> None:
        result = serialize_bets(pd.DataFrame())
        assert result.items == []
        assert result.total == 0

    def test_populated(self) -> None:
        bets = pd.DataFrame(
            {
                "bet_id": ["b1"],
                "game_id": ["2026_01_KC_LAC"],
                "placed_at": [pd.Timestamp("2026-09-07T18:00:00Z")],
                "market_type": ["spread"],
                "side": ["home"],
                "line": [-3.5],
                "odds": [-110],
                "stake": [100.0],
                "book": ["draftkings"],
                "status": ["open"],
                "pnl": [None],
                "closing_line": [None],
                "clv": [None],
                "model_name": ["win_prob"],
                "model_type": ["random_forest"],
            },
        )
        result = serialize_bets(bets)
        assert result.total == 1
        assert result.items[0].bet_id == "b1"
        assert result.items[0].stake == 100.0
        assert result.items[0].pnl is None


class TestSerializeBankrollCurve:
    def test_empty(self) -> None:
        result = serialize_bankroll_curve(pd.DataFrame(), None)
        assert result.items == []

    def test_populated(self) -> None:
        history = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
                "running_balance": [100.0, 150.0],
            },
        )
        result = serialize_bankroll_curve(history, "30d")
        assert result.total == 2
        assert result.period == "30d"
        assert result.items[1].bankroll == 150.0


class TestSerializeTransactions:
    def test_empty(self) -> None:
        result = serialize_transactions(pd.DataFrame())
        assert result.items == []


class TestSerializeSplits:
    def test_empty(self) -> None:
        result = serialize_splits(pd.DataFrame(), "market_type")
        assert result.items == []
        assert result.dimension == "market_type"

    def test_populated(self) -> None:
        splits_df = pd.DataFrame(
            {
                "market_type": ["spread", "total"],
                "wins": [3, 5],
                "losses": [2, 3],
                "pushes": [0, 0],
                "total": [5, 8],
                "win_pct": [0.6, 0.625],
                "roi": [0.05, 0.10],
            },
        )
        result = serialize_splits(splits_df, "market_type")
        assert result.total == 2
        assert result.items[0].dimension_value == "spread"
        assert result.items[0].roi == 0.05
