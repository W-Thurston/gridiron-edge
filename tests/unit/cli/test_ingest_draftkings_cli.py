# tests/unit/cli/test_ingest_draftkings_cli.py

"""Tests for the explicit legacy DraftKings ingestion command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.ingest.odds import DraftKingsUnavailableError

runner = CliRunner()


def test_help_identifies_legacy_best_effort_adapter() -> None:
    result = runner.invoke(ingest_app, ["dk-odds", "--help"])

    assert result.exit_code == 0
    assert "legacy" in result.stdout.lower()
    assert "best-effort" in result.stdout.lower()
    assert "nflverse" in result.stdout.lower()


def test_adapter_unavailability_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_fetch = MagicMock(
        side_effect=DraftKingsUnavailableError("Legacy DraftKings adapter returned no usable JSON.")
    )
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.fetch_dk_odds",
        mock_fetch,
    )

    result = runner.invoke(ingest_app, ["dk-odds"])

    assert result.exit_code == 1
    assert "legacy draftkings adapter" in result.output.lower()
    assert "no usable json" in result.output.lower()
    mock_fetch.assert_called_once_with()


def test_valid_empty_result_does_not_claim_files_were_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_fetch = MagicMock(return_value=None)
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.fetch_dk_odds",
        mock_fetch,
    )

    result = runner.invoke(ingest_app, ["dk-odds"])

    assert result.exit_code == 0
    assert "no current rows returned" in result.output.lower()
    assert "no files written" in result.output.lower()
    mock_fetch.assert_called_once_with()


def test_success_reports_generic_storage_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "data" / "odds" / "odds_log.parquet"
    snapshot = tmp_path / "data" / "odds" / "odds_current.parquet"
    mock_fetch = MagicMock(return_value=(ledger, snapshot))
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.fetch_dk_odds",
        mock_fetch,
    )

    result = runner.invoke(ingest_app, ["dk-odds"])

    assert result.exit_code == 0
    assert str(ledger) in result.output
    assert str(snapshot) in result.output
    assert "dk_odds_" not in result.output
    mock_fetch.assert_called_once_with()
