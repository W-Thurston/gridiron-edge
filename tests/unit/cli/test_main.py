"""Tests for cli/main.py pipeline contracts and staleness checks."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.main import (
    ALL_STAGES,
    _check_stage_staleness,
    _run_pipeline_stages,
    run_data_pipeline,
)
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
        assert "ingest dk-odds" not in result.output

    def test_help_exposes_one_elo_rebuild_contract(self) -> None:
        app = typer.Typer()
        app.command("run-data-pipeline")(run_data_pipeline)

        result = CliRunner().invoke(
            app,
            [
                "run-data-pipeline",
                "--help",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "fit-elo-all-years" not in result.output
        assert "incremental" not in result.output.lower()


class TestFetchGamesStage:
    """Tests for raw-games fetch ownership in the shared pipeline."""

    @patch("gridiron_edge.cli.main._check_stage_staleness")
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games_refresh",
    )
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games",
    )
    def test_all_years_uses_explicit_full_replacement(
        self,
        fetch_games: MagicMock,
        refresh_games: MagicMock,
        _check_staleness: MagicMock,
        tmp_path: Path,
    ) -> None:
        fetch_games.return_value = tmp_path / "games_raw_nflverse.parquet"

        _run_pipeline_stages(
            active={"fetch-games"},
            all_years=True,
            resolved_season=2026,
            upcoming_target=2026,
            season=2026,
            season_year="2026-2027",
            owm_api_key=None,
        )

        fetch_games.assert_called_once_with()
        refresh_games.assert_not_called()

    @patch("gridiron_edge.cli.main._check_stage_staleness")
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games_refresh",
    )
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games",
    )
    def test_explicit_season_uses_history_preserving_refresh(
        self,
        fetch_games: MagicMock,
        refresh_games: MagicMock,
        _check_staleness: MagicMock,
        tmp_path: Path,
    ) -> None:
        refresh_games.return_value = tmp_path / "games_raw_nflverse.parquet"

        _run_pipeline_stages(
            active={"fetch-games"},
            all_years=False,
            resolved_season=2026,
            upcoming_target=2026,
            season=2026,
            season_year="2026-2027",
            owm_api_key=None,
        )

        refresh_games.assert_called_once_with(season=2026)
        fetch_games.assert_not_called()

    @patch("gridiron_edge.cli.main._check_stage_staleness")
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games_refresh",
    )
    @patch(
        "gridiron_edge.ingest.nflverse.fetch_nflverse_games",
    )
    def test_default_weekly_refresh_infers_current_season(
        self,
        fetch_games: MagicMock,
        refresh_games: MagicMock,
        _check_staleness: MagicMock,
        tmp_path: Path,
    ) -> None:
        refresh_games.return_value = tmp_path / "games_raw_nflverse.parquet"

        _run_pipeline_stages(
            active={"fetch-games"},
            all_years=False,
            resolved_season=2026,
            upcoming_target=2026,
            season=None,
            season_year=None,
            owm_api_key=None,
        )

        refresh_games.assert_called_once_with(season=None)
        fetch_games.assert_not_called()


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
