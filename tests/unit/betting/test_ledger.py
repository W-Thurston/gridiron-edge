# tests/unit/betting/test_ledger.py
"""Unit tests for the bet ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
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
    # pyrefly: ignore [bad-argument-type]
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
        assert _BET_COLUMNS[17:19] == ["model_name", "model_type"]


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


class TestBetReferenceProvenance:
    """Exact reference-offer evidence is optional, strict, and immutable."""

    _REFERENCE_COLUMNS: ClassVar[list[str]] = [
        "reference_provider",
        "reference_provider_event_id",
        "reference_sportsbook",
        "reference_market_fetched_at",
        "reference_sportsbook_updated_at",
        "reference_commence_time",
        "reference_american_odds",
        "reference_line",
    ]

    def test_reference_columns_are_canonical(self) -> None:
        """Reference evidence follows actual wager terms in schema order."""
        assert _BET_COLUMNS[9:17] == self._REFERENCE_COLUMNS
        assert _BET_COLUMNS[17:19] == ["model_name", "model_type"]

    def test_manual_bet_has_null_reference_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        """Manual wagers do not fabricate source evidence from the book."""
        bet_id = _log_one(tmp_path)
        row = load_bets(repo=tmp_path).set_index("bet_id").loc[bet_id]
        assert row[self._REFERENCE_COLUMNS].isna().all()
        assert row["book"] == "draftkings"

    def test_exact_reference_offer_survives_persistence(
        self,
        tmp_path: Path,
    ) -> None:
        """All exact source and market fields round-trip unchanged."""
        fetched_at = datetime(2026, 9, 1, 12, tzinfo=UTC)
        updated_at = datetime(2026, 9, 1, 11, 59, tzinfo=UTC)
        commence_time = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)
        bet_id = log_bet(
            game_id=_GAME_ID,
            market_type="spread",
            side="away",
            odds=-105,
            stake=100.0,
            book="fanduel",
            line=4.0,
            reference_provider="the_odds_api",
            reference_provider_event_id="event-1",
            reference_sportsbook="draftkings",
            reference_market_fetched_at=fetched_at,
            reference_sportsbook_updated_at=updated_at,
            reference_commence_time=commence_time,
            reference_american_odds=-110,
            reference_line=3.5,
            repo=tmp_path,
        )

        row = load_bets(repo=tmp_path).set_index("bet_id").loc[bet_id]
        assert row["book"] == "fanduel"
        assert row["odds"] == -105
        assert row["line"] == 4.0
        assert row["reference_provider"] == "the_odds_api"
        assert row["reference_provider_event_id"] == "event-1"
        assert row["reference_sportsbook"] == "draftkings"
        assert row["reference_market_fetched_at"] == fetched_at
        assert row["reference_sportsbook_updated_at"] == updated_at
        assert row["reference_commence_time"] == commence_time
        assert row["reference_american_odds"] == -110
        assert row["reference_line"] == 3.5

    @pytest.mark.parametrize(
        "orphan",
        [
            {"reference_provider_event_id": "event-1"},
            {"reference_sportsbook": "draftkings"},
            {"reference_american_odds": -110},
            {"reference_line": 3.5},
        ],
    )
    def test_orphaned_reference_fields_are_rejected(
        self,
        tmp_path: Path,
        orphan: dict[str, object],
    ) -> None:
        """Reference evidence requires an explicit provider identity."""
        with pytest.raises(ValueError, match="reference_provider"):
            # pyrefly: ignore [bad-argument-type]
            log_bet(**_DEFAULTS, **orphan, repo=tmp_path)
        assert not (tmp_path / "data/betting/bet_ledger.parquet").exists()

    def test_reference_fetch_time_is_required(self, tmp_path: Path) -> None:
        """Provider-backed evidence requires its local observation time."""
        with pytest.raises(ValueError, match="reference_market_fetched_at"):
            _log_one(
                tmp_path,
                reference_provider="the_odds_api",
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("reference_provider_event_id", " "),
            ("reference_sportsbook", " "),
        ],
    )
    def test_empty_optional_reference_text_is_rejected(
        self,
        tmp_path: Path,
        field: str,
        value: str,
    ) -> None:
        """Optional source text is either absent or nonempty."""
        values: dict[str, object] = {
            "reference_provider": "the_odds_api",
            "reference_market_fetched_at": datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
            field: value,
        }
        with pytest.raises(ValueError, match=field):
            # pyrefly: ignore [bad-argument-type]
            log_bet(**_DEFAULTS, **values, repo=tmp_path)

    @pytest.mark.parametrize(
        "field",
        [
            "reference_market_fetched_at",
            "reference_sportsbook_updated_at",
            "reference_commence_time",
        ],
    )
    def test_reference_timestamps_require_utc(
        self,
        tmp_path: Path,
        field: str,
    ) -> None:
        """Naive reference timestamps are rejected at the write boundary."""
        values: dict[str, object] = {
            "reference_provider": "the_odds_api",
            "reference_market_fetched_at": datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
            field: datetime(2026, 9, 1, 12),
        }
        with pytest.raises(ValueError, match="timezone-aware UTC"):
            # pyrefly: ignore [bad-argument-type]
            log_bet(**_DEFAULTS, **values, repo=tmp_path)

    @pytest.mark.parametrize("odds", [0, float("inf"), float("-inf")])
    def test_invalid_reference_odds_are_rejected(
        self,
        tmp_path: Path,
        odds: float,
    ) -> None:
        """Reference American odds must be finite and nonzero."""
        with pytest.raises(ValueError, match="finite and nonzero"):
            # pyrefly: ignore [bad-argument-type]
            log_bet(
                # pyrefly: ignore [bad-argument-type]
                **_DEFAULTS,
                reference_provider="the_odds_api",
                reference_market_fetched_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
                # pyrefly: ignore [bad-argument-type]
                reference_american_odds=odds,
                repo=tmp_path,
            )

    def test_nonfinite_reference_line_is_rejected(self, tmp_path: Path) -> None:
        """Reference point values must be finite."""
        with pytest.raises(ValueError, match="reference_line"):
            _log_one(
                tmp_path,
                reference_provider="the_odds_api",
                reference_market_fetched_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
                reference_line=float("nan"),
            )

    def test_settlement_preserves_reference_evidence(
        self,
        tmp_path: Path,
    ) -> None:
        """Settlement changes outcomes, not immutable reference evidence."""
        fetched_at = datetime(2026, 9, 1, 12, tzinfo=UTC)
        bet_id = log_bet(
            # pyrefly: ignore [bad-argument-type]
            **_DEFAULTS,
            reference_provider="the_odds_api",
            reference_provider_event_id="event-1",
            reference_sportsbook="draftkings",
            reference_market_fetched_at=fetched_at,
            reference_american_odds=-150,
            repo=tmp_path,
        )
        before = (
            load_bets(repo=tmp_path)
            .set_index("bet_id")
            .loc[
                bet_id,
                self._REFERENCE_COLUMNS,
            ]
            .copy()
        )
        settle_bet(bet_id, "won", repo=tmp_path)
        after = (
            load_bets(repo=tmp_path)
            .set_index("bet_id")
            .loc[
                bet_id,
                self._REFERENCE_COLUMNS,
            ]
        )
        pd.testing.assert_series_equal(before, after)
