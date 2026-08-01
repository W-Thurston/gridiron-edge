# tests/unit/betting/test_ledger.py
"""Unit tests for the bet ledger."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from gridiron_edge.betting.ledger import (
    _BET_COLUMNS,
    _read_ledger,
    _write_ledger,
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
    """Current bet-ledger schema invariants."""

    def test_model_identity_columns_are_canonical(self) -> None:
        assert _BET_COLUMNS[9:11] == ["model_name", "model_type"]


class TestLogBetModelIdentity:
    """Bet model identity is one optional model_name/model_type pair."""

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

    def test_model_identity_may_be_omitted(self, tmp_path: Path) -> None:
        bet_id: str = _log_one(tmp_path)

        df = load_bets(repo=tmp_path)
        row = df.loc[df["bet_id"] == bet_id].iloc[0]

        assert pd.isna(row["model_name"])
        assert pd.isna(row["model_type"])

    def test_log_bet_does_not_collapse_model_variants(
        self,
        tmp_path: Path,
    ) -> None:
        """Different algorithms for one game remain distinct rows."""
        log_bet(
            game_id=_GAME_ID,
            market_type="moneyline",
            side="home",
            odds=-110,
            stake=100.0,
            book="draftkings",
            model_name="win_prob",
            model_type="logistic",
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
        assert set(df["model_type"]) == {
            "logistic",
            "random_forest",
        }

    @pytest.mark.parametrize(
        ("model_name", "model_type"),
        [
            ("win_prob", None),
            (None, "random_forest"),
        ],
    )
    def test_rejects_incomplete_model_identity(
        self,
        tmp_path: Path,
        model_name: str | None,
        model_type: str | None,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="model_name and model_type must be provided together",
        ):
            log_bet(
                game_id=_GAME_ID,
                market_type="moneyline",
                side="home",
                odds=-110,
                stake=100.0,
                book="draftkings",
                model_name=model_name,
                model_type=model_type,
                repo=tmp_path,
            )

        assert not (tmp_path / "data" / "betting" / "bet_ledger.parquet").exists()

    @pytest.mark.parametrize(
        ("model_name", "model_type", "message"),
        [
            (
                "",
                "random_forest",
                "model_name must be a nonempty string",
            ),
            (
                "win_prob",
                "",
                "model_type must be a nonempty string",
            ),
            (
                "   ",
                "random_forest",
                "model_name must be a nonempty string",
            ),
            (
                "win_prob",
                "   ",
                "model_type must be a nonempty string",
            ),
        ],
    )
    def test_rejects_empty_model_identity_values(
        self,
        tmp_path: Path,
        model_name: str,
        model_type: str,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            log_bet(
                game_id=_GAME_ID,
                market_type="moneyline",
                side="home",
                odds=-110,
                stake=100.0,
                book="draftkings",
                model_name=model_name,
                model_type=model_type,
                repo=tmp_path,
            )

        assert not (tmp_path / "data" / "betting" / "bet_ledger.parquet").exists()


# ---------------------------------------------------------------------------
# TestPersistedLedgerSchema
# ---------------------------------------------------------------------------


class TestPersistedLedgerSchema:
    """Strict schema checks at the persisted ledger boundary."""

    @staticmethod
    def _ledger_path(repo: Path) -> Path:
        return repo / "data" / "betting" / "bet_ledger.parquet"

    def test_exact_current_ledger_loads(
        self,
        tmp_path: Path,
    ) -> None:
        bet_id = _log_one(tmp_path)

        loaded = _read_ledger(tmp_path)

        assert loaded["bet_id"].tolist() == [bet_id]
        assert loaded.columns.tolist() == _BET_COLUMNS

    def test_read_rejects_missing_column(
        self,
        tmp_path: Path,
    ) -> None:
        _log_one(tmp_path)
        path = self._ledger_path(tmp_path)

        malformed = pd.read_parquet(path).drop(columns=["model_name"])
        malformed.to_parquet(path, index=False)

        with pytest.raises(
            ValueError,
            match="missing columns: model_name",
        ):
            _read_ledger(tmp_path)

        assert pd.read_parquet(path).columns.tolist() == (malformed.columns.tolist())

    def test_read_rejects_extra_column(
        self,
        tmp_path: Path,
    ) -> None:
        _log_one(tmp_path)
        path = self._ledger_path(tmp_path)

        malformed = pd.read_parquet(path)
        malformed["unexpected_field"] = "unexpected"
        malformed.to_parquet(path, index=False)

        with pytest.raises(
            ValueError,
            match="extra columns: unexpected_field",
        ):
            _read_ledger(tmp_path)

        assert pd.read_parquet(path).columns.tolist() == (malformed.columns.tolist())

    def test_read_rejects_reordered_columns(
        self,
        tmp_path: Path,
    ) -> None:
        _log_one(tmp_path)
        path = self._ledger_path(tmp_path)

        reordered_columns = [
            _BET_COLUMNS[1],
            _BET_COLUMNS[0],
            *_BET_COLUMNS[2:],
        ]
        malformed = pd.read_parquet(path).loc[
            :,
            reordered_columns,
        ]
        malformed.to_parquet(path, index=False)

        with pytest.raises(
            ValueError,
            match="columns are not in canonical order",
        ):
            _read_ledger(tmp_path)

        assert pd.read_parquet(path).columns.tolist() == (reordered_columns)

    @pytest.mark.parametrize(
        "malformation",
        [
            "missing",
            "extra",
            "reordered",
        ],
    )
    def test_write_rejects_invalid_schema(
        self,
        tmp_path: Path,
        malformation: str,
    ) -> None:
        frame = pd.DataFrame(columns=_BET_COLUMNS)

        if malformation == "missing":
            frame = frame.drop(columns=["model_type"])
            message = "missing columns: model_type"
        elif malformation == "extra":
            frame["unexpected_field"] = pd.Series(dtype="object")
            message = "extra columns: unexpected_field"
        else:
            reordered_columns = [
                _BET_COLUMNS[1],
                _BET_COLUMNS[0],
                *_BET_COLUMNS[2:],
            ]
            frame = frame.loc[:, reordered_columns]
            message = "columns are not in canonical order"

        with pytest.raises(ValueError, match=message):
            _write_ledger(frame, tmp_path)

        assert not self._ledger_path(tmp_path).exists()

    def test_malformed_ledger_prevents_log_overwrite(
        self,
        tmp_path: Path,
    ) -> None:
        _log_one(tmp_path)
        path = self._ledger_path(tmp_path)

        malformed = pd.read_parquet(path).drop(columns=["model_type"])
        malformed.to_parquet(path, index=False)

        with pytest.raises(
            ValueError,
            match="missing columns: model_type",
        ):
            _log_one(
                tmp_path,
                game_id="2026_02_BUF_MIA",
            )

        stored = pd.read_parquet(path)
        assert stored.columns.tolist() == malformed.columns.tolist()
        assert len(stored) == 1

    def test_malformed_ledger_prevents_settlement_overwrite(
        self,
        tmp_path: Path,
    ) -> None:
        bet_id = _log_one(tmp_path)
        path = self._ledger_path(tmp_path)

        malformed = pd.read_parquet(path)
        malformed["unexpected_field"] = "unexpected"
        malformed.to_parquet(path, index=False)

        with pytest.raises(
            ValueError,
            match="extra columns: unexpected_field",
        ):
            settle_bet(
                bet_id,
                "won",
                repo=tmp_path,
            )

        stored = pd.read_parquet(path)
        assert stored.columns.tolist() == malformed.columns.tolist()
        assert stored.iloc[0]["status"] == "open"

    def test_missing_ledger_returns_canonical_empty_frame(
        self,
        tmp_path: Path,
    ) -> None:
        loaded = _read_ledger(tmp_path)

        assert loaded.empty
        assert loaded.columns.tolist() == _BET_COLUMNS


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


class TestMlClvUnification:
    """Verify the ledger's _ml_clv matches the canonical CLV helper."""

    def test_ml_clv_matches_canonical_helper(self) -> None:
        from gridiron_edge.betting.ledger import _ml_clv
        from gridiron_edge.market.clv import closing_line_value
        from gridiron_edge.market.odds_math import american_to_implied_prob

        bet_odds = -110
        closing_odds = -130

        bet_prob = american_to_implied_prob(bet_odds)
        close_prob = american_to_implied_prob(closing_odds)
        expected = closing_line_value(bet_prob, close_prob)

        actual = _ml_clv(bet_odds, closing_odds)
        assert actual is not None
        assert actual == pytest.approx(expected)

    def test_ml_clv_returns_none_on_invalid_odds(self) -> None:
        from gridiron_edge.betting.ledger import _ml_clv

        # american_to_implied_prob of 0 yields 0 prob -> returns None.
        assert _ml_clv(0, -130) is None
        assert _ml_clv(-110, 0) is None
