# tests/unit/betting/test_history_import.py
"""Tests for validated historical betting imports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.betting.bankroll import current_balance, load_transactions
from gridiron_edge.betting.history_import import import_history
from gridiron_edge.betting.ledger import load_bets, log_bet

BET_COLUMNS = [
    "placed_at_local",
    "source_bet_id",
    "description",
    "market_type",
    "game_id",
    "side",
    "line",
    "american_odds",
    "stake",
    "status",
    "paid_amount",
    "potential_payout",
    "sportsbook",
    "funding_type",
    "source_transaction_id",
    "notes",
]
TXN_COLUMNS = [
    "timestamp_local",
    "txn_type",
    "amount",
    "balance_after",
    "source_transaction_id",
    "linked_source_bet_id",
    "notes",
]


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    bets = pd.DataFrame(
        [
            {
                "placed_at_local": "2026-09-07T14:48:13-06:00",
                "source_bet_id": "DK-WIN",
                "description": "Jacksonville Jaguars",
                "market_type": "moneyline",
                "game_id": "2026_01_CLE_JAX",
                "side": "home",
                "line": "",
                "american_odds": -440,
                "stake": 1.0,
                "status": "won",
                "paid_amount": 1.22,
                "potential_payout": "",
                "sportsbook": "draftkings",
                "funding_type": "cash",
                "source_transaction_id": "",
                "notes": "",
            },
            {
                "placed_at_local": "2026-09-12T08:00:20-06:00",
                "source_bet_id": "DK-UNRESOLVED",
                "description": "19-pick SGP parlay",
                "market_type": "same_game_parlay",
                "game_id": "",
                "side": "",
                "line": "",
                "american_odds": 145273632,
                "stake": 0.1,
                "status": "lost",
                "paid_amount": "",
                "potential_payout": "",
                "sportsbook": "draftkings",
                "funding_type": "bonus",
                "source_transaction_id": "",
                "notes": "Funded with sportsbook promotional currency.",
            },
            {
                "placed_at_local": "2026-09-16T12:00:35-06:00",
                "source_bet_id": "DK-OPEN",
                "description": "Denver Broncos",
                "market_type": "moneyline",
                "game_id": "2026_02_JAX_DEN",
                "side": "home",
                "line": "",
                "american_odds": -148,
                "stake": 1.0,
                "status": "open",
                "paid_amount": "",
                "potential_payout": 1.67,
                "sportsbook": "draftkings",
                "funding_type": "cash",
                "source_transaction_id": "",
                "notes": "",
            },
        ],
        columns=BET_COLUMNS,
    )
    transactions = pd.DataFrame(
        [
            {
                "timestamp_local": "2026-09-07T14:47:00-06:00",
                "txn_type": "opening_balance",
                "amount": 10.0,
                "balance_after": 10.0,
                "source_transaction_id": "",
                "linked_source_bet_id": "",
                "notes": "Opening cash balance",
            },
            {
                "timestamp_local": "2026-09-07T14:48:00-06:00",
                "txn_type": "bet_placed",
                "amount": 1.0,
                "balance_after": 9.0,
                "source_transaction_id": "TXN-PLACE-WIN",
                "linked_source_bet_id": "DK-WIN",
                "notes": "",
            },
            {
                "timestamp_local": "2026-09-13T13:45:00-06:00",
                "txn_type": "bet_settled",
                "amount": 1.22,
                "balance_after": 10.22,
                "source_transaction_id": "TXN-PAYOUT-WIN",
                "linked_source_bet_id": "DK-WIN",
                "notes": "",
            },
            {
                "timestamp_local": "2026-09-16T12:00:00-06:00",
                "txn_type": "bet_placed",
                "amount": 1.0,
                "balance_after": 9.22,
                "source_transaction_id": "TXN-PLACE-OPEN",
                "linked_source_bet_id": "DK-OPEN",
                "notes": "",
            },
        ],
        columns=TXN_COLUMNS,
    )
    bets_path = tmp_path / "bets.csv"
    transactions_path = tmp_path / "transactions.csv"
    bets.to_csv(bets_path, index=False)
    transactions.to_csv(transactions_path, index=False)
    return bets_path, transactions_path


def test_dry_run_validates_without_writing(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)

    result = import_history(
        bets_path=bets_path,
        transactions_path=transactions_path,
        repo=tmp_path,
    )

    assert result.bet_count == 3
    assert result.settled_bet_count == 2
    assert result.open_bet_count == 1
    assert result.won_bet_count == 1
    assert result.lost_bet_count == 1
    assert result.opening_balance == pytest.approx(10.0)
    assert result.cash_wager_outflow == pytest.approx(2.0)
    assert result.settlement_inflow == pytest.approx(1.22)
    assert result.final_bankroll == pytest.approx(9.22)
    assert result.bonus_funding == pytest.approx(0.1)
    assert result.unresolved_funding == pytest.approx(0.0)
    assert result.published is False
    assert not (tmp_path / "data/betting/bet_ledger.parquet").exists()
    assert not (tmp_path / "data/betting/bankroll_txn.parquet").exists()


def test_replace_publishes_both_canonical_ledgers(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)

    result = import_history(
        bets_path=bets_path,
        transactions_path=transactions_path,
        replace_existing=True,
        repo=tmp_path,
    )

    bets = load_bets(repo=tmp_path)
    transactions = load_transactions(repo=tmp_path)
    assert result.published is True
    assert len(bets) == 3
    assert len(transactions) == 4
    assert current_balance(repo=tmp_path) == pytest.approx(9.22)
    winner = bets.set_index("source_bet_id").loc["DK-WIN"]
    assert winner["paid_amount"] == pytest.approx(1.22)
    assert winner["pnl"] == pytest.approx(0.22)
    bonus_funded = bets.set_index("source_bet_id").loc["DK-UNRESOLVED"]
    assert bonus_funded["funding_type"] == "bonus"


def test_duplicate_source_bet_id_is_rejected(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    bets = pd.read_csv(bets_path, keep_default_na=False)
    bets.loc[1, "source_bet_id"] = bets.loc[0, "source_bet_id"]
    bets.to_csv(bets_path, index=False)

    with pytest.raises(ValueError, match="duplicate source_bet_id"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            repo=tmp_path,
        )


def test_duplicate_source_transaction_id_is_rejected(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    transactions = pd.read_csv(transactions_path, keep_default_na=False)
    transactions.loc[2, "source_transaction_id"] = transactions.loc[
        1,
        "source_transaction_id",
    ]
    transactions.to_csv(transactions_path, index=False)

    with pytest.raises(ValueError, match="duplicate source_transaction_id"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            repo=tmp_path,
        )


def test_running_balance_mismatch_is_rejected(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    transactions = pd.read_csv(transactions_path, keep_default_na=False)
    transactions.loc[1, "balance_after"] = 8.0
    transactions.to_csv(transactions_path, index=False)

    with pytest.raises(ValueError, match="running balance mismatch"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            repo=tmp_path,
        )


def test_naive_timestamp_is_rejected(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    bets = pd.read_csv(bets_path, keep_default_na=False)
    bets.loc[0, "placed_at_local"] = "2026-09-07T14:48:13"
    bets.to_csv(bets_path, index=False)

    with pytest.raises(ValueError, match="explicit timezone offset"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            repo=tmp_path,
        )


def test_moneyline_without_game_id_is_rejected(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    bets = pd.read_csv(bets_path, keep_default_na=False)
    bets.loc[0, "game_id"] = ""
    bets.to_csv(bets_path, index=False)

    with pytest.raises(ValueError, match="requires game_id"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            repo=tmp_path,
        )


def test_funding_type_is_not_inferred_from_notes(tmp_path: Path) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    bets = pd.read_csv(bets_path, keep_default_na=False)
    bets.loc[1, "funding_type"] = "unresolved"
    bets.loc[1, "notes"] = "Funded with sportsbook promotional currency."
    bets.to_csv(bets_path, index=False)

    result = import_history(
        bets_path=bets_path,
        transactions_path=transactions_path,
        repo=tmp_path,
    )

    assert result.bonus_funding == pytest.approx(0.0)
    assert result.unresolved_funding == pytest.approx(0.1)


def test_failed_second_write_restores_both_prior_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bets_path, transactions_path = _write_sources(tmp_path)
    original_bet_id = log_bet(
        "2026_01_KC_LAC",
        market_type="moneyline",
        side="home",
        odds=-150,
        stake=1.0,
        book="draftkings",
        repo=tmp_path,
    )
    from gridiron_edge.betting.bankroll import deposit

    deposit(5.0, repo=tmp_path)
    ledger_path = tmp_path / "data/betting/bet_ledger.parquet"
    transaction_path = tmp_path / "data/betting/bankroll_txn.parquet"
    old_ledger = ledger_path.read_bytes()
    old_transactions = transaction_path.read_bytes()

    def fail_write(_frame: pd.DataFrame, _repo: Path) -> Path:
        raise RuntimeError("transaction publication failed")

    monkeypatch.setattr(
        "gridiron_edge.betting.history_import._write_txn_log",
        fail_write,
    )

    with pytest.raises(RuntimeError, match="transaction publication failed"):
        import_history(
            bets_path=bets_path,
            transactions_path=transactions_path,
            replace_existing=True,
            repo=tmp_path,
        )

    assert ledger_path.read_bytes() == old_ledger
    assert transaction_path.read_bytes() == old_transactions
    assert load_bets(repo=tmp_path)["bet_id"].tolist() == [original_bet_id]
