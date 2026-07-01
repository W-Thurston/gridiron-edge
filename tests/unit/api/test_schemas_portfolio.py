# tests/unit/api/test_schemas_portfolio.py

"""Unit tests for portfolio schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.portfolio import (
    BankrollCurve,
    BetRow,
    CurveBucket,
    PortfolioSplits,
    PortfolioSummary,
    SplitRow,
    TransactionRow,
)


class TestPortfolioSummary:
    def test_default_construction(self) -> None:
        s = PortfolioSummary()
        assert s.bankroll is None
        assert s.total_bets is None

    def test_populated(self) -> None:
        s = PortfolioSummary(bankroll=1000.0, total_bets=5, wins=3, roi_pct=5.2)
        assert s.bankroll == 1000.0
        assert s.wins == 3
        assert s.roi_pct == 5.2

    def test_is_frozen(self) -> None:
        s = PortfolioSummary()
        with pytest.raises(ValidationError):
            s.bankroll = 100.0

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioSummary(unexpected="x")


class TestBetRow:
    def test_default_construction(self) -> None:
        assert BetRow() is not None

    def test_populated(self) -> None:
        row = BetRow(bet_id="x", stake=100.0, odds=-110, status="won")
        assert row.stake == 100.0

    def test_is_frozen(self) -> None:
        row = BetRow()
        with pytest.raises(ValidationError):
            row.stake = 200.0


class TestBankrollCurve:
    def test_default_construction(self) -> None:
        curve = BankrollCurve()
        assert curve.items == []
        assert curve.period is None

    def test_populated(self) -> None:
        curve = BankrollCurve(
            items=[CurveBucket(timestamp="2025-01-01", bankroll=100.0)],
            total=1,
            period="30d",
        )
        assert curve.items[0].bankroll == 100.0
        assert curve.period == "30d"


class TestTransactionRow:
    def test_default_construction(self) -> None:
        assert TransactionRow() is not None


class TestPortfolioSplits:
    def test_construction(self) -> None:
        splits = PortfolioSplits(
            items=[
                SplitRow(dimension_value="spread", total=10, wins=5),
            ],
            dimension="market_type",
        )
        assert splits.dimension == "market_type"
        assert splits.items[0].dimension_value == "spread"
