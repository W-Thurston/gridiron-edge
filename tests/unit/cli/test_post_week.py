"""Tests for completed-week live forecast closeout orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.post_week import (
    _build_stages,
    _stage_close_live_forecasts,
    _stage_refresh_next_week_state,
    _stage_refresh_results,
    post_week_cmd,
)


class TestStageList:
    def test_stage_names(self) -> None:
        assert [stage.name for stage in _build_stages()] == [
            "refresh-results",
            "refresh-next-week-state",
            "close-live-forecasts",
        ]

    def test_stages_are_independently_executable(self) -> None:
        assert all(not stage.depends_on for stage in _build_stages())

    def test_no_stage_is_soft_fail(self) -> None:
        assert all(not stage.soft_fail for stage in _build_stages())


@patch("gridiron_edge.cli.post_week._run_pipeline")
def test_refresh_results_runs_only_result_stages(run_pipeline: MagicMock) -> None:
    result = _stage_refresh_results({"season_int": 2025, "season": "2025-2026"})

    assert result.success
    run_pipeline.assert_called_once_with(
        {"season_int": 2025, "season": "2025-2026"},
        active={"fetch-games", "clean-games"},
    )


@patch("gridiron_edge.cli.post_week._run_pipeline")
def test_refresh_next_week_state_is_independent(run_pipeline: MagicMock) -> None:
    ctx = {"season_int": 2025, "season": "2025-2026"}

    result = _stage_refresh_next_week_state(ctx)

    assert result.success
    run_pipeline.assert_called_once_with(
        ctx,
        active={
            "fetch-upcoming",
            "clean-upcoming",
            "build-epa",
            "build-elo",
            "build-features",
        },
    )


def _closeout(*, complete: bool):
    missing = () if complete else ("g2",)
    return SimpleNamespace(
        complete=complete,
        scheduled_game_count=2,
        completed_outcome_count=1 if not complete else 2,
        selected_win_count=2,
        matched_win_event_count=2,
        selected_total_count=2,
        matched_total_event_count=2,
        missing_win_component_game_ids=(),
        missing_total_component_game_ids=(),
        missing_win_event_game_ids=(),
        missing_total_event_game_ids=(),
        missing_outcome_game_ids=missing,
        win=SimpleNamespace(
            evaluated_count=1,
            brier=0.2,
            log_loss=0.6,
            accuracy=0.75,
        ),
        total=SimpleNamespace(
            evaluated_count=1,
            mae=3.0,
            rmse=4.0,
            bias=-1.0,
        ),
    )


@patch("gridiron_edge.evaluation.live_forecast_closeout.load_live_forecast_closeout")
@patch("gridiron_edge.cli.post_week.get_settings", create=True)
def test_closeout_reports_exact_live_metrics(
    settings: MagicMock,
    load_closeout: MagicMock,
) -> None:
    settings.return_value.repo_root = "/repo"
    load_closeout.return_value = _closeout(complete=True)

    result = _stage_close_live_forecasts({"season": "2025-2026", "week": 1})

    assert result.success
    assert result.rows == 2
    assert "Win Brier 0.2000" in result.detail
    assert "Total MAE 3.00" in result.detail
    assert result.warnings == []


@patch("gridiron_edge.evaluation.live_forecast_closeout.load_live_forecast_closeout")
@patch("gridiron_edge.cli.post_week.get_settings", create=True)
def test_incomplete_closeout_is_visible_and_fails(
    settings: MagicMock,
    load_closeout: MagicMock,
) -> None:
    settings.return_value.repo_root = "/repo"
    load_closeout.return_value = _closeout(complete=False)

    result = _stage_close_live_forecasts({"season": "2025-2026", "week": 1})

    assert not result.success
    assert "incomplete closeout" in result.detail
    assert result.warnings == ["missing outcomes: g2"]


class TestCommandInvocation:
    @patch("gridiron_edge.cli.post_week._stage_close_live_forecasts")
    @patch("gridiron_edge.cli.post_week._stage_refresh_next_week_state")
    @patch("gridiron_edge.cli.post_week._stage_refresh_results")
    def test_runs_all_stages(
        self,
        refresh_results: MagicMock,
        refresh_state: MagicMock,
        closeout: MagicMock,
    ) -> None:
        refresh_results.return_value = StageResult(success=True, detail="ok")
        refresh_state.return_value = StageResult(success=True, detail="ok")
        closeout.return_value = StageResult(success=True, detail="ok")
        app = typer.Typer()
        app.command()(post_week_cmd)

        result = CliRunner().invoke(
            app,
            ["--week", "1", "--season", "2025-2026"],
        )

        assert result.exit_code == 0, result.output

    def test_only_closeout_is_executable(self) -> None:
        app = typer.Typer()
        app.command()(post_week_cmd)
        with patch(
            "gridiron_edge.cli.post_week._stage_close_live_forecasts",
            return_value=StageResult(success=True, detail="ok"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "--week",
                    "1",
                    "--season",
                    "2025-2026",
                    "--only",
                    "close-live-forecasts",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_help_has_no_model_or_backfill_options(self) -> None:
        app = typer.Typer()
        app.command()(post_week_cmd)
        result = CliRunner().invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "--model-name" not in result.output
        assert "--model-type" not in result.output
        assert "backfill-predictions" not in result.output
        assert "close-live-forecasts" in result.output

    def test_invalid_season_fails(self) -> None:
        app = typer.Typer()
        app.command()(post_week_cmd)
        result = CliRunner().invoke(
            app,
            ["--week", "1", "--season", "bad-season"],
        )

        assert result.exit_code != 0
        assert "Could not parse season" in result.output
