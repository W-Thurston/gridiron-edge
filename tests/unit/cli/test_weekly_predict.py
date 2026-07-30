# tests/unit/cli/test_weekly_predict.py

"""Tests for the weekly-predict composite command."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridiron_edge.cli._composites import CompositeStage, StageResult
from gridiron_edge.cli.weekly_predict import (
    _ALL_STAGES,
    _build_stages,
    _canonicalize_live_elo_predictions,
    _stage_predict_week,
)


def _live_elo_predictions() -> pd.DataFrame:
    """Create display-oriented live Elo prediction rows."""
    return pd.DataFrame(
        {
            "GAME_ID": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
            ],
            "GAME_DATE": [
                "2026-09-05",
                "2026-09-06",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
            ],
            "HOME_TEAM": [
                "Los Angeles Chargers",
                "Buffalo Bills",
            ],
            "AWAY_TEAM_ELO": [
                1520.0,
                1510.0,
            ],
            "HOME_TEAM_ELO": [
                1480.0,
                1530.0,
            ],
            "AWAY_WIN_PROB": [
                0.55,
                0.48,
            ],
            "HOME_WIN_PROB": [
                0.45,
                0.52,
            ],
        }
    )


class TestStageList:
    """Verify the stage list is well-formed."""

    def test_stages_have_expected_names(self) -> None:
        names: list[str] = [s.name for s in _build_stages()]
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
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert "ensure-data-fresh" in stages["predict-week"].depends_on

    def test_render_depends_on_predict(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert "predict-week" in stages["render-outputs"].depends_on

    def test_generate_edges_depends_on_predict_and_odds(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        deps: tuple[str, ...] = stages["generate-edges"].depends_on
        assert "predict-week" in deps
        assert "fetch-odds" in deps

    def test_fetch_odds_is_soft_fail(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert stages["fetch-odds"].soft_fail is True

    def test_generate_edges_is_soft_fail(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert stages["generate-edges"].soft_fail is True


class TestCanonicalizeLiveEloPredictions:
    """Verify display-oriented Elo rows map to canonical predictions."""

    def test_maps_live_elo_columns(self) -> None:
        source = _live_elo_predictions()

        canonical = _canonicalize_live_elo_predictions(
            source,
            season="2026-2027",
            week=1,
        )

        assert canonical["season"].tolist() == [
            "2026-2027",
            "2026-2027",
        ]
        assert canonical["week"].tolist() == [1, 1]
        assert canonical["game_id"].tolist() == [
            "2026_01_KC_LAC",
            "2026_01_BAL_BUF",
        ]
        assert canonical["away_elo"].tolist() == pytest.approx([1520.0, 1510.0])
        assert canonical["home_elo"].tolist() == pytest.approx([1480.0, 1530.0])
        assert canonical["away_win_prob"].tolist() == pytest.approx([0.55, 0.48])
        assert canonical["home_win_prob"].tolist() == pytest.approx([0.45, 0.52])

    def test_does_not_mutate_display_frame(self) -> None:
        source = _live_elo_predictions()
        original = source.copy(deep=True)

        _canonicalize_live_elo_predictions(
            source,
            season="2026-2027",
            week=1,
        )

        pd.testing.assert_frame_equal(
            source,
            original,
        )


class TestPredictWeekStage:
    """Cover the predict-week stage's expected paths."""

    @patch("gridiron_edge.cli.weekly_predict.write_forecast_events")
    @patch("gridiron_edge.viz.predictions.build_predictions_df")
    def test_returns_failure_on_empty_predictions(
        self,
        mock_build: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        mock_build.return_value = pd.DataFrame()
        ctx = {
            "week": 1,
            "season": "2026-2027",
        }

        result = _stage_predict_week(ctx)

        assert not result.success
        assert "no predictions" in result.detail
        mock_write.assert_not_called()

    @patch("gridiron_edge.cli.weekly_predict.datetime")
    @patch("gridiron_edge.cli.weekly_predict.new_forecast_run_id")
    @patch("gridiron_edge.cli.weekly_predict.write_forecast_events")
    @patch("gridiron_edge.viz.predictions.build_predictions_df")
    def test_writes_live_events_and_caches_display_frame(
        self,
        mock_build: MagicMock,
        mock_write: MagicMock,
        mock_run_id: MagicMock,
        mock_datetime: MagicMock,
    ) -> None:
        from gridiron_edge.cli.weekly_predict import _stage_predict_week

        predictions = _live_elo_predictions()
        generated_at = datetime(
            2026,
            9,
            1,
            12,
            tzinfo=UTC,
        )

        mock_build.return_value = predictions
        mock_write.return_value = Path("/tmp/forecast_events.parquet")
        mock_run_id.return_value = "live-run"
        mock_datetime.now.return_value = generated_at

        ctx: dict = {
            "week": 1,
            "season": "2026-2027",
        }

        result = _stage_predict_week(ctx)

        assert result.success
        assert result.rows == 2
        assert result.detail == "2 live forecast events written"
        assert ctx["predictions_df"] is predictions

        written_events = mock_write.call_args.args[0]

        assert written_events["run_id"].tolist() == [
            "live-run",
            "live-run",
        ]
        assert written_events["role"].tolist() == [
            "live",
            "live",
        ]
        assert written_events["generated_at"].tolist() == [
            pd.Timestamp(generated_at),
            pd.Timestamp(generated_at),
        ]
        assert written_events["event_id"].is_unique
        assert (written_events["model_name"] == "win_prob").all()
        assert (written_events["model_type"] == "elo").all()

        assert mock_write.call_args.kwargs["repo"] is not None

    @patch("gridiron_edge.cli.weekly_predict.new_forecast_run_id")
    @patch("gridiron_edge.cli.weekly_predict.write_forecast_events")
    @patch("gridiron_edge.viz.predictions.build_predictions_df")
    def test_separate_invocations_use_separate_run_ids(
        self,
        mock_build: MagicMock,
        mock_write: MagicMock,
        mock_run_id: MagicMock,
    ) -> None:
        from gridiron_edge.cli.weekly_predict import _stage_predict_week

        mock_build.return_value = _live_elo_predictions()
        mock_write.return_value = Path("/tmp/forecast_events.parquet")
        mock_run_id.side_effect = [
            "live-run-1",
            "live-run-2",
        ]

        ctx = {
            "week": 1,
            "season": "2026-2027",
        }

        _stage_predict_week(ctx.copy())
        _stage_predict_week(ctx.copy())

        first_events = mock_write.call_args_list[0].args[0]
        second_events = mock_write.call_args_list[1].args[0]

        assert set(first_events["run_id"]) == {
            "live-run-1",
        }
        assert set(second_events["run_id"]) == {
            "live-run-2",
        }


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
        result = runner.invoke(
            app,
            [
                "--week",
                "1",
                "--season",
                "2026-2027",
                "--model-type",
                "random_forest",
            ],
        )
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


class TestModelTypeResolution:
    """Cover --model-type auto sentinel handling (W13 Tier 3 Step 2)."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def test_explicit_model_type_passes_through(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        with (
            patch(
                "gridiron_edge.cli.weekly_predict._stage_ensure_data_fresh",
                return_value=StageResult(success=True, detail="ok"),
            ),
            patch(
                "gridiron_edge.cli.weekly_predict._stage_fetch_odds",
                return_value=StageResult(success=True, detail="ok"),
            ),
            patch(
                "gridiron_edge.cli.weekly_predict._stage_predict_week",
                return_value=StageResult(success=True, detail="ok"),
            ),
            patch(
                "gridiron_edge.cli.weekly_predict._stage_render_outputs",
                return_value=StageResult(success=True, detail="ok"),
            ),
            patch(
                "gridiron_edge.cli.weekly_predict._stage_generate_edges",
                return_value=StageResult(success=True, detail="ok"),
            ),
        ):
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
                    "--model-type",
                    "xgboost",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "model=xgboost" in result.output

    def test_auto_resolves_from_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:00:00",
                    "source_run_id": "RUN_X",
                    "metrics": {"brier": 0.213},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )

        for stage_fn in (
            "_stage_ensure_data_fresh",
            "_stage_fetch_odds",
            "_stage_predict_week",
            "_stage_render_outputs",
            "_stage_generate_edges",
        ):
            monkeypatch.setattr(
                f"gridiron_edge.cli.weekly_predict.{stage_fn}",
                lambda ctx: StageResult(success=True, detail="stubbed"),
            )

        app = typer.Typer()
        app.command()(weekly_predict_cmd)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--week", "1", "--season", "2026-2027"],
        )

        assert result.exit_code == 0, result.output
        assert "model=random_forest" in result.output

    def test_auto_fails_when_manifest_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        # tmp_path has no manifest.
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )

        for stage_fn in (
            "_stage_ensure_data_fresh",
            "_stage_fetch_odds",
            "_stage_predict_week",
            "_stage_render_outputs",
            "_stage_generate_edges",
        ):
            monkeypatch.setattr(
                f"gridiron_edge.cli.weekly_predict.{stage_fn}",
                lambda ctx: StageResult(success=True, detail="stubbed"),
            )

        app = typer.Typer()
        app.command()(weekly_predict_cmd)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--week", "1", "--season", "2026-2027"],
        )

        assert result.exit_code != 0
        assert "requires a champion manifest" in result.output


def test_canonicalization_preserves_missing_elo_values() -> None:
    predictions = pd.DataFrame(
        {
            "GAME_ID": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
            ],
            "GAME_DATE": [
                "2026-09-05",
                "2026-09-06",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
            ],
            "HOME_TEAM": [
                "Los Angeles Chargers",
                "Buffalo Bills",
            ],
            "AWAY_TEAM_ELO": [
                1520.0,
                1510.0,
            ],
            "HOME_TEAM_ELO": [
                1480.0,
                pd.NA,
            ],
            "AWAY_WIN_PROB": pd.Series(
                [
                    0.55,
                    pd.NA,
                ],
                dtype="Float64",
            ),
            "HOME_WIN_PROB": pd.Series(
                [
                    0.45,
                    pd.NA,
                ],
                dtype="Float64",
            ),
            "PREDICTION_STATUS": [
                "ready",
                "missing_home_elo",
            ],
        }
    )

    canonical = _canonicalize_live_elo_predictions(
        predictions,
        season="2026-2027",
        week=1,
    )

    assert len(canonical) == len(predictions)

    missing = canonical.loc[canonical["game_id"] == "2026_01_BAL_BUF"].iloc[0]

    assert missing["season"] == "2026-2027"
    assert missing["week"] == 1
    assert missing["away_elo"] == 1510.0
    assert pd.isna(missing["home_elo"])
    assert pd.isna(missing["away_win_prob"])
    assert pd.isna(missing["home_win_prob"])
