"""Tests for the scheduler-neutral collection-plan CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pandas import DataFrame
from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app

runner = CliRunner()


@patch("gridiron_edge.market.collection_plan_store.write_collection_plan")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
def test_plan_odds_writes_reviewable_plan_without_provider_access(
    mock_settings,
    mock_schedule,
    mock_write,
    tmp_path: Path,
) -> None:
    mock_settings.return_value = SimpleNamespace(repo_root=tmp_path)
    mock_schedule.return_value = DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g",
                "game_date": "2026-09-10",
                "game_time": "20:20:00",
            }
        ]
    )
    mock_write.return_value = tmp_path / "week=01.json"
    result = runner.invoke(
        ingest_app,
        [
            "plan-odds",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--plan-start",
            "2026-09-08T12:00:00Z",
            "--created-at",
            "2026-08-11T14:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Planned polls:" in result.output
    assert "Projected credits:" in result.output
    mock_write.assert_called_once()
