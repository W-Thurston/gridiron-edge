"""Tests for explicit collection-plan execution command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.market.collection_execution import CollectionDueResult, CollectionDueStatus

runner = CliRunner()


@patch("gridiron_edge.market.collection_execution.execute_due_collection")
@patch("gridiron_edge.market.collection_plan_store.read_collection_plan")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_execute_odds_plan_is_explicit_single_shot(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_plan: MagicMock,
    mock_execute: MagicMock,
) -> None:
    mock_key.return_value = "key"
    mock_settings.return_value = SimpleNamespace(repo_root="/tmp/repo")
    mock_execute.return_value = CollectionDueResult(CollectionDueStatus.NOT_DUE, None)
    result = runner.invoke(
        ingest_app,
        [
            "execute-odds-plan",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--evaluated-at",
            "2026-09-08T11:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not_due" in result.output
    mock_execute.assert_called_once()
