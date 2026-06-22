"""Tests for the full-retrain composite command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.full_retrain import (
    _GAME_MODEL_PAIRS,
    _PROP_ALGORITHMS,
    _PROP_STAT_FAMILIES,
    ModelPair,
    _build_stages,
    _resolve_game_pairs,
    _resolve_prop_pairs,
)


class TestStageList:
    """Verify the stage list is well-formed."""

    def test_stages_have_expected_names(self) -> None:
        names = [s.name for s in _build_stages()]
        assert names == [
            "refresh-all-data",
            "backfill-game-models",
            "backfill-prop-models",
            "refresh-calibrations",
            "baseline-report",
        ]

    def test_backfill_game_depends_on_refresh(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "refresh-all-data" in stages["backfill-game-models"].depends_on

    def test_calibrations_depend_on_game_backfill(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "backfill-game-models" in stages["refresh-calibrations"].depends_on

    def test_no_stages_are_soft_fail(self) -> None:
        for stage in _build_stages():
            assert not stage.soft_fail, (
                f"Stage {stage.name!r} should not be soft-fail in full-retrain"
            )


class TestPairResolution:
    """Cover --game-models and --prop-models resolution."""

    def test_empty_game_models_returns_all_pairs(self) -> None:
        pairs = _resolve_game_pairs([])
        assert len(pairs) == len(_GAME_MODEL_PAIRS)

    def test_specific_game_model_returns_only_that_pair(self) -> None:
        pairs = _resolve_game_pairs(["win_prob_random_forest"])
        assert len(pairs) == 1
        assert pairs[0].model_name == "win_prob"
        assert pairs[0].model_type == "random_forest"

    def test_unknown_game_model_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown game"):
            _resolve_game_pairs(["nonexistent_model"])

    def test_empty_prop_models_returns_all_15(self) -> None:
        pairs = _resolve_prop_pairs([])
        assert len(pairs) == len(_PROP_STAT_FAMILIES) * len(_PROP_ALGORITHMS)

    def test_specific_prop_model_returns_only_that_pair(self) -> None:
        pairs = _resolve_prop_pairs(["qb_pass_yards_elasticnet"])
        assert pairs == [("qb_pass_yards", "elasticnet")]

    def test_unknown_prop_model_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown prop"):
            _resolve_prop_pairs(["nonexistent_pair"])


class TestModelPair:
    def test_composite_key(self) -> None:
        pair = ModelPair(model_name="win_prob", model_type="random_forest")
        assert pair.composite_key == "win_prob_random_forest"


class TestBackfillGameModelsStage:
    """Cover the backfill-game-models stage's main paths."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_game_models,
        )

        ctx = {"game_pairs": []}
        result = _stage_backfill_game_models(ctx)
        assert result.success
        assert "no game pairs requested" in result.detail

    @patch("gridiron_edge.evaluation.backfill.backfill_model")
    def test_iterates_over_pairs(self, mock_backfill: MagicMock) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_game_models,
        )

        mock_backfill.return_value = 100
        ctx = {
            "game_pairs": [
                ModelPair(model_name="win_prob", model_type="elo"),
                ModelPair(model_name="win_prob", model_type="random_forest"),
            ]
        }

        result = _stage_backfill_game_models(ctx)
        assert result.success
        assert result.rows == 200  # 100 per pair x 2 pairs
        assert mock_backfill.call_count == 2


class TestBackfillPropModelsStage:
    """Cover the backfill-prop-models stage's no-op path."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_prop_models,
        )

        ctx = {"prop_pairs": []}
        result = _stage_backfill_prop_models(ctx)
        assert result.success
        assert "no prop pairs requested" in result.detail


class TestBaselineReportStage:
    """Cover the baseline-report stage."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_baseline_report,
        )

        ctx = {"game_pairs": []}
        result = _stage_baseline_report(ctx)
        assert result.success
        assert "no pairs to report" in result.detail


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.full_retrain._stage_refresh_all_data")
    @patch("gridiron_edge.cli.full_retrain._stage_backfill_game_models")
    @patch("gridiron_edge.cli.full_retrain._stage_backfill_prop_models")
    @patch("gridiron_edge.cli.full_retrain._stage_refresh_calibrations")
    @patch("gridiron_edge.cli.full_retrain._stage_baseline_report")
    def test_runs_all_stages_when_all_succeed(
        self,
        mock_report: MagicMock,
        mock_calib: MagicMock,
        mock_props: MagicMock,
        mock_games: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.full_retrain import full_retrain_cmd

        for m in [
            mock_refresh,
            mock_games,
            mock_props,
            mock_calib,
            mock_report,
        ]:
            m.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(full_retrain_cmd)

        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

    def test_skip_prop_backfill_shorthand(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.full_retrain import full_retrain_cmd

        with (
            patch("gridiron_edge.cli.full_retrain._stage_refresh_all_data") as mock_refresh,
            patch("gridiron_edge.cli.full_retrain._stage_backfill_game_models") as mock_games,
            patch("gridiron_edge.cli.full_retrain._stage_backfill_prop_models") as mock_props,
            patch("gridiron_edge.cli.full_retrain._stage_refresh_calibrations") as mock_calib,
            patch("gridiron_edge.cli.full_retrain._stage_baseline_report") as mock_report,
        ):
            for m in [
                mock_refresh,
                mock_games,
                mock_props,
                mock_calib,
                mock_report,
            ]:
                m.return_value = StageResult(success=True, detail="ok")

            app = typer.Typer()
            app.command()(full_retrain_cmd)

            runner = CliRunner()
            result = runner.invoke(app, ["--skip-prop-backfill"])
            assert result.exit_code == 0, result.output
            # Backfill-prop-models should not have been called.
            mock_props.assert_not_called()
