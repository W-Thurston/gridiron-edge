"""Tests for the weather ingest and gap-filling backfill CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app

runner = CliRunner()


@patch("gridiron_edge.ingest.weather.fetch_weather")
@patch("gridiron_edge.cli.ingest.get_owm_api_key")
def test_weather_fetches_most_recent_week(
    mock_key: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    mock_key.return_value = "resolved-key"

    result = runner.invoke(ingest_app, ["weather", "--season-year", "2026-2027"])

    assert result.exit_code == 0
    mock_fetch.assert_called_once_with(season_year="2026-2027", owm_api_key="resolved-key")


@patch("gridiron_edge.ingest.weather.backfill_weather")
@patch("gridiron_edge.cli.ingest.get_owm_api_key")
def test_weather_backfill_closes_a_season_gap(
    mock_key: MagicMock,
    mock_backfill: MagicMock,
) -> None:
    mock_key.return_value = "resolved-key"
    mock_backfill.return_value = (48, 0)

    result = runner.invoke(
        ingest_app,
        ["weather-backfill", "--season-year", "2026-2027"],
    )

    assert result.exit_code == 0
    assert "Fetched: 48" in result.output
    assert "Failed: 0" in result.output
    mock_backfill.assert_called_once_with(
        season_year="2026-2027",
        owm_api_key="resolved-key",
        dry_run=False,
        max_calls=None,
    )


@patch("gridiron_edge.ingest.weather.backfill_weather")
@patch("gridiron_edge.cli.ingest.get_owm_api_key")
def test_weather_backfill_defaults_to_every_season(
    mock_key: MagicMock,
    mock_backfill: MagicMock,
) -> None:
    mock_key.return_value = "resolved-key"
    mock_backfill.return_value = (0, 0)

    result = runner.invoke(ingest_app, ["weather-backfill"])

    assert result.exit_code == 0
    mock_backfill.assert_called_once_with(
        season_year=None,
        owm_api_key="resolved-key",
        dry_run=False,
        max_calls=None,
    )


@patch("gridiron_edge.ingest.weather.backfill_weather")
@patch("gridiron_edge.cli.ingest.get_owm_api_key")
def test_weather_backfill_forwards_dry_run_and_max_calls(
    mock_key: MagicMock,
    mock_backfill: MagicMock,
) -> None:
    mock_key.return_value = "resolved-key"
    mock_backfill.return_value = (0, 0)

    result = runner.invoke(
        ingest_app,
        [
            "weather-backfill",
            "--season-year",
            "2026-2027",
            "--dry-run",
            "--max-calls",
            "100",
        ],
    )

    assert result.exit_code == 0
    mock_backfill.assert_called_once_with(
        season_year="2026-2027",
        owm_api_key="resolved-key",
        dry_run=True,
        max_calls=100,
    )
