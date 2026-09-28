# tests/unit/cli/test_player_stats_commands.py

"""Tests for the player-stats ingest and clean commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.cli.transform import transform_app

runner = CliRunner()


@patch("gridiron_edge.core.settings.current_nfl_season", return_value=2026)
@patch("gridiron_edge.ingest.nflverse.player_stats.fetch_player_stats")
def test_default_refreshes_current_season(fetch: MagicMock, _season: MagicMock) -> None:
    fetch.return_value = [Path("player_stats_2026.parquet")]

    result = runner.invoke(ingest_app, ["player-stats"])

    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(seasons=[2026], force=True)


@patch("gridiron_edge.ingest.nflverse.player_stats.fetch_player_stats")
def test_explicit_seasons_respect_force_flag(fetch: MagicMock) -> None:
    fetch.return_value = []

    result = runner.invoke(ingest_app, ["player-stats", "--season", "2024", "--season", "2025"])
    assert result.exit_code == 0, result.output
    fetch.assert_called_with(seasons=[2024, 2025], force=False)

    result = runner.invoke(ingest_app, ["player-stats", "--season", "2025", "--force"])
    assert result.exit_code == 0, result.output
    fetch.assert_called_with(seasons=[2025], force=True)


@patch("gridiron_edge.ingest.nflverse.player_stats.fetch_player_stats")
def test_all_years_fetches_full_history(fetch: MagicMock) -> None:
    fetch.return_value = []

    result = runner.invoke(ingest_app, ["player-stats", "--all-years"])

    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(all_years=True)


@patch("gridiron_edge.transform.clean.player_stats.clean_player_stats")
def test_clean_command_rebuilds_logs(clean: MagicMock, tmp_path: Path) -> None:
    clean.return_value = tmp_path / "player_game_logs.parquet"

    result = runner.invoke(transform_app, ["clean-player-stats"])

    assert result.exit_code == 0, result.output
    clean.assert_called_once_with()
