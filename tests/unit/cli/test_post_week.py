"""Tests for the post-week composite command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.post_week import _build_stages


class TestStageList:
    """Verify the stage list is well-formed."""

    def test_stages_have_expected_names(self) -> None:
        names = [s.name for s in _build_stages()]
        assert names == [
            "refresh-data",
            "backfill-predictions",
            "evaluate-summary",
        ]

    def test_backfill_depends_on_refresh(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "refresh-data" in stages["backfill-predictions"].depends_on

    def test_evaluate_depends_on_backfill(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "backfill-predictions" in stages["evaluate-summary"].depends_on

    def test_no_stages_are_soft_fail(self) -> None:
        """post-week stages should all be hard-fail.

        Data refresh and archive writes are local operations that
        should not fail silently.
        """
        for stage in _build_stages():
            assert not stage.soft_fail, f"Stage {stage.name!r} should not be soft-fail"


class TestBackfillPredictionsStage:
    """Cover the backfill-predictions stage's expected paths."""

    @patch("gridiron_edge.cli.post_week.backfill_model")
    def test_reports_no_generated_predictions(self, mock_backfill: MagicMock) -> None:
        from gridiron_edge.cli.post_week import (
            _stage_backfill_predictions,
        )

        mock_backfill.return_value = 0
        ctx = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2025-2026",
        }

        result = _stage_backfill_predictions(ctx)
        assert result.success
        assert result.detail == ("no predictions generated for the requested season")
        assert result.rows is None
        assert "overwrite" not in mock_backfill.call_args.kwargs

    @patch("gridiron_edge.cli.post_week.backfill_model")
    def test_reports_archive_count_on_success(self, mock_backfill: MagicMock) -> None:
        from gridiron_edge.cli.post_week import (
            _stage_backfill_predictions,
        )

        mock_backfill.return_value = 16
        ctx = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2025-2026",
        }

        result = _stage_backfill_predictions(ctx)
        assert result.success
        assert "16" in result.detail
        assert result.rows == 16


class TestEvaluateSummaryStage:
    """Cover the evaluate-summary stage's expected paths."""

    @patch("gridiron_edge.cli.post_week.build_evaluation_df")
    def test_returns_no_data_when_archive_empty(self, mock_build: MagicMock) -> None:
        import pandas as pd

        from gridiron_edge.cli.post_week import _stage_evaluate_summary

        mock_build.return_value = pd.DataFrame()
        ctx = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2025-2026",
            "week": 1,
        }

        result = _stage_evaluate_summary(ctx)
        assert result.success
        assert "no evaluated games" in result.detail

    @patch("gridiron_edge.cli.post_week.summarise")
    @patch("gridiron_edge.cli.post_week.build_evaluation_df")
    def test_warns_on_brier_drift(self, mock_build: MagicMock, mock_summarise: MagicMock) -> None:
        import pandas as pd

        from gridiron_edge.cli.post_week import _stage_evaluate_summary

        mock_build.return_value = pd.DataFrame(
            {"game_id": ["x"], "away_win_prob": [0.5], "away_team_won": [1]}
        )
        # Season mean is 0.22; week 1 is 0.30 (significantly worse)
        mock_summarise.return_value = pd.DataFrame(
            {
                "week": [1, 2, 3],
                "brier": [0.30, 0.20, 0.16],
                "accuracy": [0.50, 0.65, 0.70],
            }
        )
        ctx = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2025-2026",
            "week": 1,
        }

        result = _stage_evaluate_summary(ctx)
        assert result.success
        assert len(result.warnings) == 1
        assert "worse" in result.warnings[0]

    @patch("gridiron_edge.cli.post_week.summarise")
    @patch("gridiron_edge.cli.post_week.build_evaluation_df")
    def test_no_warning_when_week_in_line_with_season(
        self, mock_build: MagicMock, mock_summarise: MagicMock
    ) -> None:
        import pandas as pd

        from gridiron_edge.cli.post_week import _stage_evaluate_summary

        mock_build.return_value = pd.DataFrame(
            {"game_id": ["x"], "away_win_prob": [0.5], "away_team_won": [1]}
        )
        # Season mean and week 1 are close (within 0.02 tolerance)
        mock_summarise.return_value = pd.DataFrame(
            {
                "week": [1, 2, 3],
                "brier": [0.21, 0.20, 0.19],
                "accuracy": [0.62, 0.63, 0.64],
            }
        )
        ctx = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2025-2026",
            "week": 1,
        }

        result = _stage_evaluate_summary(ctx)
        assert result.success
        assert result.warnings == []


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.post_week._stage_refresh_data")
    @patch("gridiron_edge.cli.post_week._stage_backfill_predictions")
    @patch("gridiron_edge.cli.post_week._stage_evaluate_summary")
    def test_runs_all_stages_when_all_succeed(
        self,
        mock_eval: MagicMock,
        mock_backfill: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.post_week import post_week_cmd

        mock_refresh.return_value = StageResult(success=True, detail="ok")
        mock_backfill.return_value = StageResult(success=True, detail="ok")
        mock_eval.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(post_week_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--week", "1", "--season", "2025-2026"])
        assert result.exit_code == 0, result.output

    def test_invalid_season_raises(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.post_week import post_week_cmd

        app = typer.Typer()
        app.command()(post_week_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--week", "1", "--season", "not-a-season"])
        assert result.exit_code != 0
        assert "Could not parse season" in result.output
