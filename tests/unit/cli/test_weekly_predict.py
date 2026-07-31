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
            "predict-week",
            "compose-weekly-product",
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

    def test_product_composition_depends_on_predict(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert stages["compose-weekly-product"].depends_on == ("predict-week",)

    def test_generate_edges_depends_on_selected_product(self) -> None:
        stages: dict[str, CompositeStage] = {s.name: s for s in _build_stages()}
        assert stages["generate-edges"].depends_on == ("compose-weekly-product",)

    def test_external_odds_fetch_is_not_a_stage(self) -> None:
        assert "fetch-odds" not in _ALL_STAGES

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
    """Cover the unified weekly edge service boundary."""

    def _diagnostics(
        self,
        *,
        state,
        blockers=(),
        calculated: int = 0,
        positive: int = 0,
        filtered: int = 0,
    ):
        from gridiron_edge.market.edge_diagnostics import EdgeDiagnostics

        return EdgeDiagnostics(
            season="2026-2027",
            week=1,
            prediction_game_count=1,
            market_game_count=1,
            matched_game_count=1,
            complete_moneyline_count=1,
            complete_spread_count=1,
            complete_total_count=0,
            eligible_market_count=2,
            calculated_edge_count=calculated,
            positive_edge_count=positive,
            filtered_edge_count=filtered,
            state=state,
            blockers=blockers,
        )

    def test_calls_service_and_writes_returned_rows(
        self,
        tmp_path: Path,
    ) -> None:
        from gridiron_edge.cli.weekly_predict import _stage_generate_edges
        from gridiron_edge.market.edge_diagnostics import EdgeResultState
        from gridiron_edge.market.recommendations import EdgeResult

        rows = pd.DataFrame(
            {
                "away_team": ["KC"],
                "home_team": ["LAC"],
                "market_type": ["spread"],
                "side": ["home"],
                "ev": [0.042],
                "kelly_stake": [None],
            }
        )
        edge_result = EdgeResult(
            rows=rows,
            diagnostics=self._diagnostics(
                state=EdgeResultState.POSITIVE_EDGES,
                calculated=1,
                positive=1,
                filtered=1,
            ),
        )
        with (
            patch(
                "gridiron_edge.cli.weekly_predict.get_settings",
                return_value=MagicMock(repo_root=tmp_path),
            ),
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=edge_result,
            ) as service,
        ):
            ctx = {
                "season": "2026-2027",
                "week": 1,
                "bankroll": None,
            }
            result = _stage_generate_edges(ctx)

        assert result.success
        service.assert_called_once_with(
            season="2026-2027",
            week=1,
            bankroll=None,
            kelly_multiplier=0.25,
            min_ev=0.0,
            repo=tmp_path,
        )
        assert result.rows == 1
        assert len(result.artifacts) == 1
        written = pd.read_csv(result.artifacts[0])
        assert written["ev"].tolist() == pytest.approx([0.042])
        preview = ctx["top_edges_preview"]
        assert isinstance(preview, pd.DataFrame)
        assert preview.iloc[0]["away_team"] == "KC"

    def test_blocked_result_soft_fails_without_writing(
        self,
        tmp_path: Path,
    ) -> None:
        from gridiron_edge.cli.weekly_predict import _stage_generate_edges
        from gridiron_edge.market.edge_diagnostics import (
            EdgeDiagnosticBlocker,
            EdgeResultState,
        )
        from gridiron_edge.market.recommendations import EdgeResult

        edge_result = EdgeResult(
            rows=pd.DataFrame(),
            diagnostics=self._diagnostics(
                state=EdgeResultState.BLOCKED,
                blockers=(EdgeDiagnosticBlocker.NO_MARKET_DATA,),
            ),
        )
        with (
            patch(
                "gridiron_edge.cli.weekly_predict.get_settings",
                return_value=MagicMock(repo_root=tmp_path),
            ),
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=edge_result,
            ),
        ):
            result = _stage_generate_edges({"season": "2026-2027", "week": 1, "bankroll": None})

        assert not result.success
        assert result.detail == "edge calculation blocked: no_market_data"
        assert not (tmp_path / "data" / "output" / "edges").exists()

    @pytest.mark.parametrize(
        ("state", "calculated", "positive", "message"),
        [
            (
                "no_calculable_edges",
                0,
                0,
                "no calculable edges",
            ),
            (
                "no_positive_edges",
                1,
                0,
                "no positive-EV edges",
            ),
            (
                "positive_edges",
                2,
                1,
                "below min_ev=0.0%",
            ),
        ],
    )
    def test_analytical_empty_result_is_success(
        self,
        tmp_path: Path,
        state: str,
        calculated: int,
        positive: int,
        message: str,
    ) -> None:
        from gridiron_edge.cli.weekly_predict import _stage_generate_edges
        from gridiron_edge.market.edge_diagnostics import EdgeResultState
        from gridiron_edge.market.recommendations import EdgeResult

        edge_result = EdgeResult(
            rows=pd.DataFrame(),
            diagnostics=self._diagnostics(
                state=EdgeResultState(state),
                calculated=calculated,
                positive=positive,
            ),
        )
        with (
            patch(
                "gridiron_edge.cli.weekly_predict.get_settings",
                return_value=MagicMock(repo_root=tmp_path),
            ),
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=edge_result,
            ),
        ):
            result = _stage_generate_edges({"season": "2026-2027", "week": 1, "bankroll": None})

        assert result.success
        assert message in result.detail
        assert result.artifacts == []


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.weekly_predict._stage_ensure_data_fresh")
    @patch("gridiron_edge.cli.weekly_predict._stage_predict_week")
    @patch("gridiron_edge.cli.weekly_predict._stage_compose_weekly_product")
    @patch("gridiron_edge.cli.weekly_predict._stage_render_outputs")
    @patch("gridiron_edge.cli.weekly_predict._stage_generate_edges")
    def test_runs_all_stages_when_all_succeed(
        self,
        mock_edges: MagicMock,
        mock_render: MagicMock,
        mock_product: MagicMock,
        mock_predict: MagicMock,
        mock_data: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        mock_data.return_value = StageResult(success=True, detail="ok")
        mock_predict.return_value = StageResult(success=True, detail="ok")
        mock_product.return_value = StageResult(success=True, detail="ok")
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
                "render-outputs",
                "--only",
                "predict-week",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


class TestWeeklyPredictModelContract:
    """The live weekly workflow owns an Elo forecast identity."""

    def test_help_has_no_model_type(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.weekly_predict import weekly_predict_cmd

        app = typer.Typer()
        app.command()(weekly_predict_cmd)
        result = CliRunner().invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "--model-type" not in result.output
        assert "Optional bankroll" in result.output
        assert "1000.0" not in result.output


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
