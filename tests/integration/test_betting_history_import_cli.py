# tests/integration/test_betting_history_import_cli.py
"""Integration tests for the historical betting-import CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.betting import betting_app


class _MockSettings:
    """Minimal settings stub pointing repository I/O at a temporary path."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root


@pytest.fixture
def import_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Create one valid normalized import and redirect repository settings."""
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _MockSettings(tmp_path),
    )
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
            }
        ]
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
                "source_transaction_id": "TXN-PLACE",
                "linked_source_bet_id": "DK-WIN",
                "notes": "",
            },
            {
                "timestamp_local": "2026-09-13T13:45:00-06:00",
                "txn_type": "bet_settled",
                "amount": 1.22,
                "balance_after": 10.22,
                "source_transaction_id": "TXN-SETTLE",
                "linked_source_bet_id": "DK-WIN",
                "notes": "",
            },
        ]
    )
    bets_path = tmp_path / "bets.csv"
    transactions_path = tmp_path / "transactions.csv"
    bets.to_csv(bets_path, index=False)
    transactions.to_csv(transactions_path, index=False)
    return bets_path, transactions_path


def _arguments(paths: tuple[Path, Path]) -> list[str]:
    bets, transactions = paths
    return [
        "import-history",
        "--bets",
        str(bets),
        "--transactions",
        str(transactions),
    ]


def test_import_history_defaults_to_nonwriting_dry_run(
    import_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(betting_app, _arguments(import_sources))

    assert result.exit_code == 0, result.output
    assert "validated; no files changed" in result.output
    assert "Bets: 1" in result.output
    assert "Current bankroll: $10.22" in result.output
    assert not (tmp_path / "data/betting/bet_ledger.parquet").exists()
    assert not (tmp_path / "data/betting/bankroll_txn.parquet").exists()


def test_import_history_replace_publishes_both_ledgers(
    import_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        betting_app,
        [*_arguments(import_sources), "--replace"],
    )

    assert result.exit_code == 0, result.output
    assert "Historical portfolio imported" in result.output
    assert (tmp_path / "data/betting/bet_ledger.parquet").exists()
    assert (tmp_path / "data/betting/bankroll_txn.parquet").exists()


def test_import_history_reports_validation_failure(
    import_sources: tuple[Path, Path],
) -> None:
    bets_path, transactions_path = import_sources
    transactions = pd.read_csv(transactions_path, keep_default_na=False)
    transactions.loc[1, "balance_after"] = 8.0
    transactions.to_csv(transactions_path, index=False)

    result = CliRunner().invoke(
        betting_app,
        _arguments((bets_path, transactions_path)),
    )

    assert result.exit_code == 1
    assert "running balance mismatch" in result.output
