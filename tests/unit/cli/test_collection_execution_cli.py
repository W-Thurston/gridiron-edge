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


@patch("gridiron_edge.market.collection_execution.execute_due_collection")
@patch("gridiron_edge.market.collection_plan_store.load_current_collection_plan")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_execute_selected_odds_plan_resolves_selection_and_executes_once(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_load_current: MagicMock,
    mock_execute: MagicMock,
) -> None:
    mock_key.return_value = "key"
    mock_settings.return_value = SimpleNamespace(repo_root="/tmp/repo")
    mock_execute.return_value = CollectionDueResult(CollectionDueStatus.NOT_DUE, None)
    result = runner.invoke(
        ingest_app,
        [
            "execute-selected-odds-plan",
            "--evaluated-at",
            "2026-09-08T11:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not_due" in result.output
    mock_load_current.assert_called_once_with(repo="/tmp/repo")
    mock_execute.assert_called_once()


@patch("gridiron_edge.market.collection_rollover.auto_rollover")
@patch("gridiron_edge.market.collection_execution.execute_due_collection")
@patch("gridiron_edge.market.collection_plan_store.load_current_collection_plan")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_auto_rollover_runs_before_the_poll_and_a_failure_does_not_stop_it(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_load_current: MagicMock,
    mock_execute: MagicMock,
    mock_rollover: MagicMock,
) -> None:
    mock_key.return_value = "key"
    mock_settings.return_value = SimpleNamespace(repo_root="/tmp/repo")
    mock_execute.return_value = CollectionDueResult(CollectionDueStatus.NOT_DUE, None)
    mock_rollover.side_effect = RuntimeError("source down")

    result = runner.invoke(
        ingest_app,
        [
            "execute-selected-odds-plan",
            "--evaluated-at",
            "2026-10-06T12:00:00Z",
            "--auto-rollover",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "auto-rollover failed: RuntimeError: source down" in result.output
    mock_rollover.assert_called_once()
    mock_execute.assert_called_once()


@patch("gridiron_edge.market.collection_rollover.auto_rollover")
@patch("gridiron_edge.market.collection_execution.execute_due_collection")
@patch("gridiron_edge.market.collection_plan_store.load_current_collection_plan")
@patch("gridiron_edge.datasets.loaders.load_schedule_upcoming_rich")
@patch("gridiron_edge.core.settings.get_settings")
@patch("gridiron_edge.cli.ingest.get_odds_api_key")
def test_without_the_flag_no_rollover_is_attempted(
    mock_key: MagicMock,
    mock_settings: MagicMock,
    mock_schedule: MagicMock,
    mock_load_current: MagicMock,
    mock_execute: MagicMock,
    mock_rollover: MagicMock,
) -> None:
    mock_key.return_value = "key"
    mock_settings.return_value = SimpleNamespace(repo_root="/tmp/repo")
    mock_execute.return_value = CollectionDueResult(CollectionDueStatus.NOT_DUE, None)

    result = runner.invoke(
        ingest_app,
        ["execute-selected-odds-plan", "--evaluated-at", "2026-10-06T12:00:00Z"],
    )

    assert result.exit_code == 0, result.output
    mock_rollover.assert_not_called()
