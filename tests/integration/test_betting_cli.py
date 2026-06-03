# tests/integration/test_betting_cli.py
"""Integration tests for the betting CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.betting import betting_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockSettings:
    """Minimal settings stub pointing repo_root at a temp directory."""

    def __init__(self, tmp_path: Path) -> None:
        self.repo_root = tmp_path


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all data I/O to *tmp_path* and seed a bankroll deposit."""
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _MockSettings(tmp_path),
    )
    # Seed the bankroll so bet placements have funds
    from gridiron_edge.betting.bankroll import deposit

    deposit(5000.0, repo=tmp_path)
    return tmp_path


runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOG_FLAGS: list[str] = [
    "log",
    "--game-id",
    "2026_01_KC_LAC",
    "--market",
    "moneyline",
    "--side",
    "home",
    "--odds",
    "-150",
    "--stake",
    "100",
    "--book",
    "draftkings",
]


def _extract_bet_id(output: str) -> str:
    """Pull the bet UUID from 'Bet logged: <uuid>' output."""
    for line in output.splitlines():
        if line.startswith("Bet logged:"):
            return line.split("Bet logged:")[1].strip()
    msg: str = f"Could not find bet_id in output:\n{output}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogCommand:
    """Tests for 'gridiron bet log'."""

    def test_log_basic(self, cli_env: Path) -> None:
        """Log command exits 0 and prints a bet_id."""
        result = runner.invoke(betting_app, _LOG_FLAGS)
        assert result.exit_code == 0, result.output
        assert "Bet logged:" in result.output

    def test_log_with_model_context(self, cli_env: Path) -> None:
        """Log command accepts optional model flags."""
        flags: list[str] = [
            *_LOG_FLAGS,
            "--model-version",
            "rf_v3",
            "--model-prob",
            "0.62",
            "--model-ev",
            "0.08",
        ]
        result = runner.invoke(betting_app, flags)
        assert result.exit_code == 0, result.output

    def test_log_spread(self, cli_env: Path) -> None:
        """Log command accepts spread bets with a line."""
        flags: list[str] = [
            "log",
            "--game-id",
            "2026_01_KC_LAC",
            "--market",
            "spread",
            "--side",
            "home",
            "--odds",
            "-110",
            "--stake",
            "50",
            "--book",
            "draftkings",
            "--line",
            "-3.5",
        ]
        result = runner.invoke(betting_app, flags)
        assert result.exit_code == 0, result.output


class TestSettleCommand:
    """Tests for 'gridiron bet settle'."""

    def test_settle_won(self, cli_env: Path) -> None:
        """Settle a won bet: exit 0, prints PnL."""
        log_result = runner.invoke(betting_app, _LOG_FLAGS)
        bet_id: str = _extract_bet_id(log_result.output)

        result = runner.invoke(betting_app, ["settle", bet_id, "won", "--no-clv"])
        assert result.exit_code == 0, result.output
        assert "Settled:" in result.output
        assert "PnL:" in result.output

    def test_settle_lost(self, cli_env: Path) -> None:
        """Settle a lost bet."""
        log_result = runner.invoke(betting_app, _LOG_FLAGS)
        bet_id: str = _extract_bet_id(log_result.output)

        result = runner.invoke(betting_app, ["settle", bet_id, "lost", "--no-clv"])
        assert result.exit_code == 0, result.output

    def test_settle_invalid_id(self, cli_env: Path) -> None:
        """Settling a nonexistent bet exits with error."""
        result = runner.invoke(betting_app, ["settle", "bad-id", "won", "--no-clv"])
        assert result.exit_code == 1


class TestListCommand:
    """Tests for 'gridiron bet list'."""

    def test_list_shows_bets(self, cli_env: Path) -> None:
        """List command shows logged bets."""
        runner.invoke(betting_app, _LOG_FLAGS)
        result = runner.invoke(betting_app, ["list"])
        assert result.exit_code == 0, result.output
        assert "KC_LAC" in result.output

    def test_list_empty(self, cli_env: Path) -> None:
        """List with no bets prints a message."""
        result = runner.invoke(betting_app, ["list"])
        assert result.exit_code == 0, result.output
        assert "No bets found" in result.output

    def test_list_filter_status(self, cli_env: Path) -> None:
        """List with status filter."""
        runner.invoke(betting_app, _LOG_FLAGS)
        result = runner.invoke(betting_app, ["list", "--status", "open"])
        assert result.exit_code == 0, result.output
        assert "1 bet(s)" in result.output


class TestSummaryCommand:
    """Tests for 'gridiron bet summary'."""

    def test_summary_with_bets(self, cli_env: Path) -> None:
        """Summary shows performance metrics after settling bets."""
        log_result = runner.invoke(betting_app, _LOG_FLAGS)
        bet_id: str = _extract_bet_id(log_result.output)
        runner.invoke(betting_app, ["settle", bet_id, "won", "--no-clv"])

        result = runner.invoke(betting_app, ["summary"])
        assert result.exit_code == 0, result.output
        assert "Record:" in result.output
        assert "1W" in result.output

    def test_summary_empty(self, cli_env: Path) -> None:
        """Summary with no bets prints a message."""
        result = runner.invoke(betting_app, ["summary"])
        assert result.exit_code == 0, result.output
        assert "No bets" in result.output


class TestBalanceCommand:
    """Tests for 'gridiron bet balance'."""

    def test_balance_shows_amount(self, cli_env: Path) -> None:
        """Balance command shows current balance."""
        result = runner.invoke(betting_app, ["balance"])
        assert result.exit_code == 0, result.output
        assert "$" in result.output


class TestDepositWithdraw:
    """Tests for deposit and withdraw commands."""

    def test_deposit(self, cli_env: Path) -> None:
        """Deposit command exits 0 and shows new balance."""
        result = runner.invoke(betting_app, ["deposit", "500"])
        assert result.exit_code == 0, result.output
        assert "Deposited" in result.output
        assert "$500.00" in result.output

    def test_withdraw(self, cli_env: Path) -> None:
        """Withdraw command exits 0 and shows new balance."""
        result = runner.invoke(betting_app, ["withdraw", "200"])
        assert result.exit_code == 0, result.output
        assert "Withdrew" in result.output

    def test_deposit_invalid(self, cli_env: Path) -> None:
        """Depositing 0 or negative amount exits with error."""
        result = runner.invoke(betting_app, ["deposit", "0"])
        assert result.exit_code == 1


class TestExportCommand:
    """Tests for 'gridiron bet export'."""

    def test_export_creates_csv(self, cli_env: Path) -> None:
        """Export creates a CSV file."""
        runner.invoke(betting_app, _LOG_FLAGS)
        result = runner.invoke(betting_app, ["export"])
        assert result.exit_code == 0, result.output
        assert "Exported" in result.output
        csv_path: Path = cli_env / "data" / "output" / "bets" / "bets_export.csv"
        assert csv_path.exists()

    def test_export_empty(self, cli_env: Path) -> None:
        """Export with no bets prints a message."""
        result = runner.invoke(betting_app, ["export"])
        assert result.exit_code == 0, result.output
        assert "No bets" in result.output
