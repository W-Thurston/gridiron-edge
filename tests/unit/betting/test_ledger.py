# tests/unit/betting/test_ledger.py
"""Unit tests for the bet ledger."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from gridiron_edge.betting.ledger import (
    _BET_COLUMNS,
    compute_pnl,
    load_bets,
    log_bet,
    settle_bet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GAME_ID = "2026_01_KC_LAC"
_DEFAULTS: dict[str, float | int | str] = {
    "game_id": _GAME_ID,
    "market_type": "moneyline",
    "side": "home",
    "odds": -150,
    "stake": 100.0,
    "book": "draftkings",
}


def _log_one(repo: Path, **overrides) -> str:
    """Log a single bet with defaults, return bet_id."""
    kw: dict[str, float | int | str] = {**_DEFAULTS, **overrides}
    return log_bet(**kw, repo=repo)


# ---------------------------------------------------------------------------
# TestComputePnl
# ---------------------------------------------------------------------------


class TestComputePnl:
    """Tests for the pure PnL calculation."""

    def test_won_positive_odds(self) -> None:
        """Won bet at +150: stake 100 -> profit 150."""
        assert compute_pnl(100.0, 150, "won") == pytest.approx(150.0)

    def test_won_negative_odds(self) -> None:
        """Won bet at -150: stake 100 -> profit ~66.67."""
        assert compute_pnl(100.0, -150, "won") == pytest.approx(66.6667, rel=1e-3)

    def test_won_even_money(self) -> None:
        """Won bet at +100: stake 100 -> profit 100."""
        assert compute_pnl(100.0, 100, "won") == pytest.approx(100.0)

    def test_lost(self) -> None:
        """Lost bet: PnL = -stake regardless of odds."""
        assert compute_pnl(100.0, -110, "lost") == pytest.approx(-100.0)

    def test_push(self) -> None:
        """Push: PnL = 0."""
        assert compute_pnl(100.0, -110, "push") == pytest.approx(0.0)

    def test_open(self) -> None:
        """Open bet: PnL = 0 (not yet settled)."""
        assert compute_pnl(100.0, -110, "open") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestLogBet
# ---------------------------------------------------------------------------


class TestLogBet:
    """Tests for logging new bets."""

    def test_creates_ledger(self, tmp_path: Path) -> None:
        """First bet creates the ledger file and returns a UUID."""
        bet_id: str = _log_one(tmp_path)
        assert (tmp_path / "data" / "betting" / "bet_ledger.parquet").exists()
        UUID(bet_id)  # Validates it's a real UUID

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        """Second bet appends; ledger has 2 rows."""
        _log_one(tmp_path)
        _log_one(tmp_path, game_id="2026_01_SF_BAL")
        df = load_bets(repo=tmp_path)
        assert len(df) == 2

    def test_required_fields_populated(self, tmp_path: Path) -> None:
        """All required fields are set on the logged bet."""
        _log_one(tmp_path)
        df = load_bets(repo=tmp_path)
        row = df.iloc[0]
        assert row["game_id"] == _GAME_ID
        assert row["market_type"] == "moneyline"
        assert row["side"] == "home"
        assert row["odds"] == -150
        assert row["stake"] == 100.0
        assert row["book"] == "draftkings"

    def test_optional_fields_default_to_none(self, tmp_path: Path) -> None:
        """Model context fields are NaN/None when not provided."""
        _log_one(tmp_path)
        df = load_bets(repo=tmp_path)
        row = df.iloc[0]
        assert pd.isna(row["model_name"])
        assert pd.isna(row["model_type"])
        assert pd.isna(row["model_prob"])

    def test_status_is_open(self, tmp_path: Path) -> None:
        """New bets always have status 'open'."""
        _log_one(tmp_path)
        df = load_bets(repo=tmp_path)
        assert df.iloc[0]["status"] == "open"

    def test_returns_valid_uuid(self, tmp_path: Path) -> None:
        """log_bet returns a string that parses as a UUID."""
        bet_id: str = _log_one(tmp_path)
        parsed = UUID(bet_id)
        assert str(parsed) == bet_id


# ---------------------------------------------------------------------------
# TestLedgerSchema
# ---------------------------------------------------------------------------


class TestLedgerSchema:
    """Schema invariants for the bet ledger."""

    def test_includes_model_identity(self) -> None:
        assert "model_name" in _BET_COLUMNS
        assert "model_type" in _BET_COLUMNS

    def test_excludes_model_version(self) -> None:
        assert "model_version" not in _BET_COLUMNS


class TestLogBetModelIdentity:
    """Bet identity uses (model_name, model_type), not model_version."""

    def test_log_bet_records_model_identity(self, tmp_path: Path) -> None:
        bet_id: str = log_bet(
            game_id=_GAME_ID,
            market_type="moneyline",
            side="home",
            odds=-110,
            stake=100.0,
            book="draftkings",
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )

        df = load_bets(repo=tmp_path)
        row = df.loc[df["bet_id"] == bet_id].iloc[0]

        assert row["model_name"] == "win_prob"
        assert row["model_type"] == "random_forest"

    def test_log_bet_does_not_collapse_model_variants(self, tmp_path: Path) -> None:
        """Different algorithms for the same game must produce distinct rows."""
        log_bet(
            game_id=_GAME_ID,
            market_type="moneyline",
            side="home",
            odds=-110,
            stake=100.0,
            book="draftkings",
            model_name="win_prob",
            model_type="elasticnet",
            repo=tmp_path,
        )
        log_bet(
            game_id=_GAME_ID,
            market_type="moneyline",
            side="home",
            odds=-110,
            stake=100.0,
            book="draftkings",
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )

        df = load_bets(repo=tmp_path)
        assert len(df) == 2
        assert set(df["model_type"]) == {"elasticnet", "random_forest"}

    def test_log_bet_persists_both_identity_fields_independently(self, tmp_path: Path) -> None:
        """model_name and model_type are independently nullable."""
        log_bet(
            game_id=_GAME_ID,
            market_type="moneyline",
            side="home",
            odds=-110,
            stake=100.0,
            book="draftkings",
            model_name="qb_pass_yards",
            model_type=None,
            repo=tmp_path,
        )

        df = load_bets(repo=tmp_path)
        row = df.iloc[0]
        assert row["model_name"] == "qb_pass_yards"
        assert pd.isna(row["model_type"])


# ---------------------------------------------------------------------------
# TestSettleBet
# ---------------------------------------------------------------------------


class TestSettleBet:
    """Tests for settling bets."""

    def test_settle_won(self, tmp_path: Path) -> None:
        """Settling a won bet computes positive PnL."""
        bet_id: str = _log_one(tmp_path, odds=150, stake=100.0)
        row = settle_bet(bet_id, "won", repo=tmp_path)
        assert row["status"] == "won"
        assert row["pnl"] == pytest.approx(150.0)

    def test_settle_lost(self, tmp_path: Path) -> None:
        """Settling a lost bet computes negative PnL."""
        bet_id: str = _log_one(tmp_path, stake=50.0)
        row = settle_bet(bet_id, "lost", repo=tmp_path)
        assert row["status"] == "lost"
        assert row["pnl"] == pytest.approx(-50.0)

    def test_settle_push(self, tmp_path: Path) -> None:
        """Settling a push computes zero PnL."""
        bet_id: str = _log_one(tmp_path)
        row = settle_bet(bet_id, "push", repo=tmp_path)
        assert row["status"] == "push"
        assert row["pnl"] == pytest.approx(0.0)

    def test_settle_sets_timestamp(self, tmp_path: Path) -> None:
        """Settlement populates settled_at."""
        bet_id: str = _log_one(tmp_path)
        row = settle_bet(bet_id, "won", repo=tmp_path)
        assert row["settled_at"] is not None

    def test_not_found_raises(self, tmp_path: Path) -> None:
        """Settling a nonexistent bet_id raises ValueError."""
        _log_one(tmp_path)
        with pytest.raises(ValueError, match="Bet not found"):
            settle_bet("nonexistent-id", "won", repo=tmp_path)

    def test_already_settled_raises(self, tmp_path: Path) -> None:
        """Settling an already-settled bet raises ValueError."""
        bet_id: str = _log_one(tmp_path)
        settle_bet(bet_id, "won", repo=tmp_path)
        with pytest.raises(ValueError, match="already settled"):
            settle_bet(bet_id, "lost", repo=tmp_path)

    def test_returns_series(self, tmp_path: Path) -> None:
        """settle_bet returns a pd.Series with correct fields."""
        bet_id: str = _log_one(tmp_path)
        row = settle_bet(bet_id, "won", repo=tmp_path)
        assert isinstance(row, pd.Series)
        assert row["bet_id"] == bet_id


# ---------------------------------------------------------------------------
# TestLoadBets
# ---------------------------------------------------------------------------


class TestLoadBets:
    """Tests for loading and filtering bets."""

    def test_load_all(self, tmp_path: Path) -> None:
        """No filters returns all bets."""
        _log_one(tmp_path)
        _log_one(tmp_path, market_type="spread", side="home", line=-3.5)
        df = load_bets(repo=tmp_path)
        assert len(df) == 2

    def test_filter_status(self, tmp_path: Path) -> None:
        """Filtering by status returns only matching bets."""
        bet_id: str = _log_one(tmp_path)
        _log_one(tmp_path, game_id="2026_01_SF_BAL")
        settle_bet(bet_id, "won", repo=tmp_path)
        open_bets = load_bets(status="open", repo=tmp_path)
        assert len(open_bets) == 1
        won_bets = load_bets(status="won", repo=tmp_path)
        assert len(won_bets) == 1

    def test_filter_market_type(self, tmp_path: Path) -> None:
        """Filtering by market_type returns only matching bets."""
        _log_one(tmp_path, market_type="moneyline")
        _log_one(tmp_path, market_type="spread", side="home", line=-3.5)
        df = load_bets(market_type="spread", repo=tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["market_type"] == "spread"

    def test_empty_ledger(self, tmp_path: Path) -> None:
        """No ledger file returns empty DataFrame with correct columns."""
        df = load_bets(repo=tmp_path)
        assert df.empty
        assert list(df.columns) == _BET_COLUMNS

    def test_multiple_filters(self, tmp_path: Path) -> None:
        """Multiple filters combine correctly."""
        _log_one(tmp_path, market_type="spread", side="home", line=-3.5, book="draftkings")
        _log_one(tmp_path, market_type="spread", side="away", line=3.5, book="fanduel")
        _log_one(tmp_path, market_type="moneyline", side="home", book="draftkings")
        df = load_bets(market_type="spread", book="draftkings", repo=tmp_path)
        assert len(df) == 1
