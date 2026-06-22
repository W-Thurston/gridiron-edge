# tests/unit/cli/test_weekly_predict.py

"""Tests for the weekly-predict composite command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.weekly_predict import (
    _ALL_STAGES,
    _build_stages,
)


class TestStageList:
    """Verify the stage list is well-formed."""

    def test_stages_have_expected_names(self) -> None:
        names = [s.name for s in _build_stages()]
        assert names == [
            "ensure-data-fresh",
            "fetch-odds",
            "predict-week",
            "render-outputs",
            "generate-edges",
        ]

    def test_all_stages_alias_matches(self) -> None:
        assert [s.name for s in _build_stages()] == _ALL_STAGES

    def test_predict_week_depends_on_data_fresh(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "ensure-data-fresh" in stages["predict-week"].depends_on

    def test_render_depends_on_predict(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "predict-week" in stages["render-outputs"].depends_on

    def test_generate_edges_depends_on_predict_and_odds(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        deps = stages["generate-edges"].depends_on
        assert "predict-week" in deps
        assert "fetch-odds" in deps

    def test_fetch_odds_is_soft_fail(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert stages["fetch-odds"].soft_fail is True

    def test_generate_edges_is_soft_fail(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert stages["generate-edges"].soft_fail is True


class TestPredictWeekStage:
    """Cover the predict-week stage's expected paths."""

    @patch("gridiron_edge.viz.predictions.build_predictions_df")
    @patch("gridiron_edge.cli.weekly_predict.append_to_prediction_log")
    def test_returns_failure_on_empty_predictions(
        self,
        mock_append: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        import pandas as pd

        from gridiron_edge.cli.weekly_predict import _stage_predict_week

        mock_build.return_value = pd.DataFrame()
        ctx = {"week": 1, "season": "2026-2027"}

        result = _stage_predict_week(ctx)
        assert not result.success
        assert "no predictions" in result.detail
        mock_append.assert_not_called()

    @patch("gridiron_edge.viz.predictions.build_predictions_df")
    @patch("gridiron_edge.cli.weekly_predict.append_to_prediction_log")
    def test_archives_and_caches_df_on_success(
        self,
        mock_append: MagicMock,
        mock_build: MagicMock,
    ) -> None:
        import pandas as pd

        from gridiron_edge.cli.weekly_predict import _stage_predict_week

        df = pd.DataFrame({"game_id": ["x"], "away_win_prob": [0.5]})
        mock_build.return_value = df
        mock_append.return_value = Path("/tmp/archive.parquet")

        ctx: dict = {"week": 1, "season": "2026-2027"}

        result = _stage_predict_week(ctx)
        assert result.success
        assert result.rows == 1
        assert "1 predictions archived" in result.detail
        # DataFrame is stashed for downstream stages to consume
        assert ctx["predictions_df"] is df


class TestGenerateEdgesStage:
    """Cover the soft-failure paths in generate-edges."""

    @patch("gridiron_edge.cli.weekly_predict.load_prediction_log")
    def test_failure_when_no_predictions(self, mock_load: MagicMock) -> None:
        import pandas as pd

        from gridiron_edge.cli.weekly_predict import _stage_generate_edges

        mock_load.return_value = pd.DataFrame()
        ctx = {
            "week": 1,
            "season": "2026-2027",
            "model_type": "random_forest",
        }

        result = _stage_generate_edges(ctx)
        assert not result.success
        assert "no predictions" in result.detail

    @patch("gridiron_edge.cli.weekly_predict.load_prediction_log")
    @patch("gridiron_edge.cli.weekly_predict.load_current_odds")
    def test_failure_when_no_odds(self, mock_odds: MagicMock, mock_load: MagicMock) -> None:
        import pandas as pd

        from gridiron_edge.cli.weekly_predict import _stage_generate_edges

        mock_load.return_value = pd.DataFrame({"game_id": ["x"], "away_win_prob": [0.5]})
        mock_odds.return_value = None
        ctx = {
            "week": 1,
            "season": "2026-2027",
            "model_type": "random_forest",
        }

        result = _stage_generate_edges(ctx)
        assert not result.success
        assert "no current DK odds" in result.detail

    @patch("gridiron_edge.market.recommendations.rank_edges")
    @patch("gridiron_edge.market.recommendations.build_edge_report")
    @patch("gridiron_edge.models.game_prediction.post_process.get_total_std")
    @patch("gridiron_edge.models.game_prediction.post_process.get_margin_std")
    @patch("gridiron_edge.cli.weekly_predict.load_current_odds")
    @patch("gridiron_edge.cli.weekly_predict.load_prediction_log")
    def test_stashes_top_edge_preview_in_context(
        self,
        mock_predictions: MagicMock,
        mock_odds: MagicMock,
        mock_margin_std: MagicMock,
        mock_total_std: MagicMock,
        mock_build_report: MagicMock,
        mock_rank_edges: MagicMock,
    ) -> None:
        """weekly-predict should cache the top edge preview for display."""

        import pandas as pd

        from gridiron_edge.cli.weekly_predict import _stage_generate_edges

        mock_predictions.return_value = pd.DataFrame(
            {
                "game_id": ["g1"],
                "away_win_prob": [0.55],
            }
        )

        mock_odds.return_value = pd.DataFrame({"x": [1]})

        mock_margin_std.return_value = 13.0
        mock_total_std.return_value = 13.0

        mock_build_report.return_value = pd.DataFrame({"ev": [0.04]})

        ranked = pd.DataFrame(
            {
                "away_team": ["KC"],
                "home_team": ["LAC"],
                "market_type": ["spread"],
                "side": ["home"],
                "ev": [0.042],
                "kelly_stake": [18.2],
            }
        )

        mock_rank_edges.return_value = ranked

        ctx: dict[str, object] = {
            "week": 1,
            "season": "2026-2027",
            "model_type": "random_forest",
        }

        result = _stage_generate_edges(ctx)

        assert result.success
        assert "top_edges_preview" in ctx

        preview = ctx["top_edges_preview"]
        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 1
        assert preview.iloc[0]["away_team"] == "KC"
        assert preview.iloc[0]["home_team"] == "LAC"


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.weekly_predict._stage_ensure_data_fresh")
    @patch("gridiron_edge.cli.weekly_predict._stage_fetch_odds")
    @patch("gridiron_edge.cli.weekly_predict._stage_predict_week")
    @patch("gridiron_edge.cli.weekly_predict._stage_render_outputs")
    @patch("gridiron_edge.cli.weekly_predict._stage_generate_edges")
    def test_runs_all_stages_when_all_succeed(
        self,
        mock_edges: MagicMock,
        mock_render: MagicMock,
        mock_predict: MagicMock,
        mock_odds: MagicMock,
        mock_data: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        mock_data.return_value = StageResult(success=True, detail="ok")
        mock_odds.return_value = StageResult(success=True, detail="ok")
        mock_predict.return_value = StageResult(success=True, detail="ok")
        mock_render.return_value = StageResult(success=True, detail="ok")
        mock_edges.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(weekly_predict_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--week", "1", "--season", "2026-2027"])
        assert result.exit_code == 0, result.output

    def test_invalid_season_raises(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        app = typer.Typer()
        app.command()(weekly_predict_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--week", "1", "--season", "not-a-season"])
        assert result.exit_code != 0
        assert "Could not parse season" in result.output

    def test_skip_only_mutually_exclusive(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        app = typer.Typer()
        app.command()(weekly_predict_cmd)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--week",
                "1",
                "--season",
                "2026-2027",
                "--skip",
                "fetch-odds",
                "--only",
                "predict-week",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
