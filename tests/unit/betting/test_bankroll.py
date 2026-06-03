# tests/unit/betting/test_bankroll.py
"""Unit tests for bankroll management."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from gridiron_edge.betting.bankroll import (
    balance_history,
    current_balance,
    deposit,
    load_transactions,
    record_bet_placed,
    record_bet_settled,
    withdraw,
)

# ---------------------------------------------------------------------------
# TestDeposit
# ---------------------------------------------------------------------------


class TestDeposit:
    """Tests for depositing funds."""

    def test_creates_txn_log(self, tmp_path: Path) -> None:
        """First deposit creates the transaction log file."""
        deposit(500.0, repo=tmp_path)
        assert (tmp_path / "data" / "betting" / "bankroll_txn.parquet").exists()

    def test_deposit_amount(self, tmp_path: Path) -> None:
        """Deposit records the correct amount and type."""
        deposit(500.0, repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["txn_type"] == "deposit"
        assert df.iloc[0]["amount"] == 500.0

    def test_deposit_returns_txn_id(self, tmp_path: Path) -> None:
        """Deposit returns a valid UUID."""
        txn_id = deposit(500.0, repo=tmp_path)
        UUID(txn_id)

    def test_invalid_amount_raises(self, tmp_path: Path) -> None:
        """Deposit with non-positive amount raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            deposit(0.0, repo=tmp_path)
        with pytest.raises(ValueError, match="positive"):
            deposit(-100.0, repo=tmp_path)


# ---------------------------------------------------------------------------
# TestWithdraw
# ---------------------------------------------------------------------------


class TestWithdraw:
    """Tests for withdrawing funds."""

    def test_withdraw_recorded(self, tmp_path: Path) -> None:
        """Withdrawal records correct type and amount."""
        withdraw(200.0, repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["txn_type"] == "withdraw"
        assert df.iloc[0]["amount"] == 200.0

    def test_withdraw_returns_txn_id(self, tmp_path: Path) -> None:
        """Withdrawal returns a valid UUID."""
        txn_id = withdraw(200.0, repo=tmp_path)
        UUID(txn_id)

    def test_invalid_amount_raises(self, tmp_path: Path) -> None:
        """Withdrawal with non-positive amount raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            withdraw(0.0, repo=tmp_path)


# ---------------------------------------------------------------------------
# TestRecordBetPlaced
# ---------------------------------------------------------------------------


class TestRecordBetPlaced:
    """Tests for recording bet placements."""

    def test_bet_placed_recorded(self, tmp_path: Path) -> None:
        """Bet placement records correct type and amount."""
        record_bet_placed(100.0, repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["txn_type"] == "bet_placed"
        assert df.iloc[0]["amount"] == 100.0

    def test_bet_placed_with_reference(self, tmp_path: Path) -> None:
        """Bet placement stores bet_id as reference_id."""
        record_bet_placed(100.0, bet_id="abc-123", repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert df.iloc[0]["reference_id"] == "abc-123"

    def test_reduces_balance(self, tmp_path: Path) -> None:
        """Placing a bet reduces the balance."""
        deposit(1000.0, repo=tmp_path)
        record_bet_placed(100.0, repo=tmp_path)
        assert current_balance(repo=tmp_path) == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# TestRecordBetSettled
# ---------------------------------------------------------------------------


class TestRecordBetSettled:
    """Tests for recording bet settlements."""

    def test_won_credits_return(self, tmp_path: Path) -> None:
        """Won bet: deposit 1000, bet 100, win +150 profit -> balance 1150."""
        deposit(1000.0, repo=tmp_path)
        record_bet_placed(100.0, repo=tmp_path)
        record_bet_settled(100.0, 150.0, repo=tmp_path)  # gross return = 250
        assert current_balance(repo=tmp_path) == pytest.approx(1150.0)

    def test_lost_credits_zero(self, tmp_path: Path) -> None:
        """Lost bet: deposit 1000, bet 100, lose -> balance 900."""
        deposit(1000.0, repo=tmp_path)
        record_bet_placed(100.0, repo=tmp_path)
        record_bet_settled(100.0, -100.0, repo=tmp_path)  # gross return = 0
        assert current_balance(repo=tmp_path) == pytest.approx(900.0)

    def test_push_credits_stake(self, tmp_path: Path) -> None:
        """Push: deposit 1000, bet 100, push -> balance 1000."""
        deposit(1000.0, repo=tmp_path)
        record_bet_placed(100.0, repo=tmp_path)
        record_bet_settled(100.0, 0.0, repo=tmp_path)  # gross return = 100
        assert current_balance(repo=tmp_path) == pytest.approx(1000.0)

    def test_with_reference_id(self, tmp_path: Path) -> None:
        """Settlement stores bet_id as reference_id."""
        record_bet_settled(100.0, 50.0, bet_id="xyz-456", repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert df.iloc[0]["reference_id"] == "xyz-456"


# ---------------------------------------------------------------------------
# TestCurrentBalance
# ---------------------------------------------------------------------------


class TestCurrentBalance:
    """Tests for balance calculation."""

    def test_empty(self, tmp_path: Path) -> None:
        """No transactions -> balance is 0."""
        assert current_balance(repo=tmp_path) == 0.0

    def test_deposit_only(self, tmp_path: Path) -> None:
        """Single deposit -> balance equals deposit."""
        deposit(500.0, repo=tmp_path)
        assert current_balance(repo=tmp_path) == pytest.approx(500.0)

    def test_deposit_and_withdraw(self, tmp_path: Path) -> None:
        """Deposit then withdraw -> correct balance."""
        deposit(500.0, repo=tmp_path)
        withdraw(200.0, repo=tmp_path)
        assert current_balance(repo=tmp_path) == pytest.approx(300.0)

    def test_full_cycle(self, tmp_path: Path) -> None:
        """Full cycle: deposit, bet, win -> correct balance."""
        deposit(1000.0, repo=tmp_path)
        record_bet_placed(100.0, repo=tmp_path)
        # Won at +150: pnl = 150, gross return = 250
        record_bet_settled(100.0, 150.0, repo=tmp_path)
        # 1000 - 100 + 250 = 1150
        assert current_balance(repo=tmp_path) == pytest.approx(1150.0)


# ---------------------------------------------------------------------------
# TestBalanceHistory
# ---------------------------------------------------------------------------


class TestBalanceHistory:
    """Tests for balance history."""

    def test_columns(self, tmp_path: Path) -> None:
        """History has the expected columns."""
        deposit(500.0, repo=tmp_path)
        df = balance_history(repo=tmp_path)
        expected: list[str] = [
            "timestamp",
            "txn_type",
            "amount",
            "signed_amount",
            "running_balance",
        ]
        assert list(df.columns) == expected

    def test_running_balance(self, tmp_path: Path) -> None:
        """Running balance accumulates correctly."""
        deposit(500.0, repo=tmp_path)
        withdraw(100.0, repo=tmp_path)
        df = balance_history(repo=tmp_path)
        assert list(df["running_balance"]) == pytest.approx([500.0, 400.0])

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """History is sorted chronologically."""
        deposit(100.0, repo=tmp_path)
        deposit(200.0, repo=tmp_path)
        withdraw(50.0, repo=tmp_path)
        df = balance_history(repo=tmp_path)
        timestamps: list = list(df["timestamp"])
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# TestLoadTransactions
# ---------------------------------------------------------------------------


class TestLoadTransactions:
    """Tests for loading and filtering transactions."""

    def test_load_all(self, tmp_path: Path) -> None:
        """No filter returns all transactions."""
        deposit(100.0, repo=tmp_path)
        withdraw(50.0, repo=tmp_path)
        df = load_transactions(repo=tmp_path)
        assert len(df) == 2

    def test_filter_type(self, tmp_path: Path) -> None:
        """Filtering by txn_type returns only matching rows."""
        deposit(100.0, repo=tmp_path)
        withdraw(50.0, repo=tmp_path)
        deposit(200.0, repo=tmp_path)
        df = load_transactions(txn_type="deposit", repo=tmp_path)
        assert len(df) == 2
        assert all(df["txn_type"] == "deposit")
