"""Tests for cli/main.py pipeline contracts and staleness checks."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.main import ALL_STAGES, _check_stage_staleness, run_data_pipeline
from gridiron_edge.datasets.registry import dataset_path


class TestPipelineContract:
    def test_default_stage_set_is_canonical_and_has_no_odds_dependency(self) -> None:
        assert ALL_STAGES == [
            "fetch-games",
            "clean-games",
            "fetch-upcoming",
            "clean-upcoming",
            "fetch-weather",
            "build-epa",
            "build-elo",
            "build-features",
        ]
        assert all("odds" not in stage for stage in ALL_STAGES)
        assert all("draftkings" not in stage.lower() for stage in ALL_STAGES)

    @patch("gridiron_edge.cli.main._run_pipeline_stages")
    @patch("gridiron_edge.core.settings.current_nfl_season", return_value=2026)
    def test_no_flags_runs_every_registered_stage(
        self,
        _mock_current_season,
        mock_run,
    ) -> None:
        app = typer.Typer()
        app.command()(run_data_pipeline)

        result = CliRunner().invoke(app, [])

        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["active"] == set(ALL_STAGES)

    def test_help_names_current_command_and_explicit_odds_boundary(self) -> None:
        app = typer.Typer()
        app.command("run-data-pipeline")(run_data_pipeline)

        result = CliRunner().invoke(app, ["run-data-pipeline", "--help"])

        assert result.exit_code == 0, result.output
        assert "all registered stages run" in result.output
        assert "not part of this command" in result.output
        assert "ingest dk-odds" in result.output


class TestStageStalenessCheck:
    def _settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from gridiron_edge.core import settings as settings_mod

        class FakeSettings:
            repo_root = tmp_path

        monkeypatch.setattr(settings_mod, "get_settings", FakeSettings)

    def test_warns_when_registered_output_is_older_than_input(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_path = dataset_path(tmp_path, "games_raw_nflverse")
        output_path = dataset_path(tmp_path, "games")
        input_path.parent.mkdir(parents=True)
        output_path.parent.mkdir(parents=True)
        output_path.write_text("old output")
        time.sleep(0.05)
        input_path.write_text("new input")
        self._settings(tmp_path, monkeypatch)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-games"})

        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1
        assert "has stale output" in messages[0]
        assert str(output_path) in messages[0]
        assert str(input_path) in messages[0]
        assert "will rebuild the output" in messages[0]
        assert "upstream data older" not in messages[0]

    def test_uses_registry_for_upcoming_schedule_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_path = dataset_path(tmp_path, "schedule_upcoming_raw_nflverse")
        output_path = dataset_path(tmp_path, "schedule_upcoming_rich")
        input_path.parent.mkdir(parents=True)
        output_path.parent.mkdir(parents=True)
        output_path.write_text("old output")
        time.sleep(0.05)
        input_path.write_text("new input")
        self._settings(tmp_path, monkeypatch)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-upcoming"})

        message = caplog.records[0].getMessage()
        assert str(output_path) in message
        assert str(input_path) in message

    def test_no_warning_when_registered_output_is_current(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        input_path = dataset_path(tmp_path, "games_raw_nflverse")
        output_path = dataset_path(tmp_path, "games")
        input_path.parent.mkdir(parents=True)
        output_path.parent.mkdir(parents=True)
        input_path.write_text("input")
        time.sleep(0.05)
        output_path.write_text("output")
        self._settings(tmp_path, monkeypatch)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-games"})

        assert caplog.records == []

    def test_no_warning_when_registered_files_do_not_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._settings(tmp_path, monkeypatch)
        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")

        _check_stage_staleness(active={"clean-games"})

        assert caplog.records == []
