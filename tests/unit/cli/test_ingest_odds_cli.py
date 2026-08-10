"""Tests for the explicit The Odds API ingest command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.ingest.odds.the_odds_api import OddsApiUsage

runner = CliRunner()


def _result(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        quote_count=96,
        game_count=16,
        sportsbook_count=4,
        ledger_path=tmp_path
        / "data"
        / "odds"
        / "history"
        / "season=2026-2027"
        / "week=01"
        / "observations.parquet",
        snapshot_path=tmp_path / "data" / "odds" / "odds_current.parquet",
        usage=OddsApiUsage(
            requests_remaining=487,
            requests_used=13,
            request_cost=3,
        ),
    )


@patch("gridiron_edge.ingest.odds.the_odds_api.ingest_the_odds_api_current")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_command_resolves_scope_and_renders_summary(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_ingest: MagicMock,
    tmp_path: Path,
) -> None:
    schedule = pd.DataFrame({"game_id": ["game-1"]})
    mock_key.return_value = "resolved-key"
    mock_settings.return_value = SimpleNamespace(repo_root=tmp_path)
    mock_schedule.return_value = schedule
    mock_ingest.return_value = _result(tmp_path)

    result = runner.invoke(
        ingest_app,
        [
            "odds",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--odds-api-key",
            "flag-key",
            "--timeout",
            "7.5",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_key.assert_called_once_with("flag-key")
    mock_schedule.assert_called_once_with(tmp_path)
    mock_ingest.assert_called_once_with(
        api_key="resolved-key",
        schedule=schedule,
        season="2026-2027",
        week=1,
        repo=tmp_path,
        timeout=7.5,
    )
    assert "96 quotes, 16 games, 4 sportsbooks" in result.output
    assert "Requests remaining: 487" in result.output
    assert "Requests used: 13" in result.output
    assert "Request cost: 3" in result.output


@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_missing_key_stops_before_schedule_load(mock_key: MagicMock) -> None:
    import typer

    mock_key.side_effect = typer.BadParameter("missing key")
    result = runner.invoke(
        ingest_app,
        ["odds", "--season", "2026-2027", "--week", "1"],
    )
    assert result.exit_code != 0
    assert "missing key" in result.output


def test_week_range_is_validated() -> None:
    result = runner.invoke(
        ingest_app,
        ["odds", "--season", "2026-2027", "--week", "0"],
    )
    assert result.exit_code == 2


def test_timeout_is_validated() -> None:
    result = runner.invoke(
        ingest_app,
        [
            "odds",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--timeout",
            "0",
        ],
    )
    assert result.exit_code == 2


@patch("gridiron_edge.ingest.odds.the_odds_api.ingest_the_odds_api_current")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_unknown_quota_values_are_not_fabricated(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_ingest: MagicMock,
    tmp_path: Path,
) -> None:
    mock_key.return_value = "key"
    mock_settings.return_value = SimpleNamespace(repo_root=tmp_path)
    mock_schedule.return_value = pd.DataFrame({"game_id": ["game-1"]})
    value = _result(tmp_path)
    value.usage = OddsApiUsage()
    mock_ingest.return_value = value

    result = runner.invoke(
        ingest_app,
        ["odds", "--season", "2026-2027", "--week", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "Requests remaining" not in result.output
    assert "Requests used" not in result.output
    assert "Request cost" not in result.output


def test_help_documents_explicit_network_command() -> None:
    result = runner.invoke(ingest_app, ["odds", "--help"])
    assert result.exit_code == 0, result.output
    assert "current NFL featured-market quotes" in result.output
    assert "ODDS_API_KEY" in result.output
    assert "--season" in result.output
    assert "--week" in result.output
