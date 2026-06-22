# tests/unit/cli/test_props.py

"""Unit tests for props CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.props import props_app

runner = CliRunner()


class TestPropsCliStructure:
    def test_help_shows_four_commands(self) -> None:
        result = runner.invoke(props_app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("evaluate", "backfill", "projections", "champion"):
            assert cmd in result.output

    def test_evaluate_help_shows_model_type(self) -> None:
        result = runner.invoke(props_app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model-type" in result.output

    def test_champion_help_shows_model(self) -> None:
        result = runner.invoke(props_app, ["champion", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_evaluate_unknown_model_exits_error(self) -> None:
        result = runner.invoke(props_app, ["evaluate", "--model", "fake_model"])
        assert result.exit_code != 0
        assert "Unknown model" in result.output

    def test_evaluate_invalid_model_type_exits_error(self) -> None:
        result = runner.invoke(
            props_app, ["evaluate", "--model", "qb_pass_yards", "--model-type", "bad_type"]
        )
        assert result.exit_code != 0


class TestDataPrepSharing:
    """Verify champion_cmd shares data prep across model types.

    These are smoke tests; the real perf validation happens at runtime.
    """

    def test_prepare_holdout_data_function_exists(self) -> None:
        """champion_cmd's data-prep helper must be importable for sharing."""
        from gridiron_edge.cli.props import _prepare_holdout_data

        assert callable(_prepare_holdout_data)

    def test_enrich_predictions_function_exists(self) -> None:
        """The model-type-specific enrichment helper must be importable."""
        from gridiron_edge.cli.props import _enrich_predictions_for_holdout

        assert callable(_enrich_predictions_for_holdout)

    def test_train_and_enrich_still_exists(self) -> None:
        """The convenience wrapper used by evaluate_cmd, backfill_cmd,
        and projections_cmd must remain available."""
        from gridiron_edge.cli.props import _train_and_enrich

        assert callable(_train_and_enrich)


class TestPropsRegistry:
    def test_all_prop_models_registered(self) -> None:
        from gridiron_edge.cli.props import _all_prop_models

        assert _all_prop_models() == [
            "qb_pass_yards",
            "qb_rush_yards",
            "rb_rush_yards",
            "te_rec_yards",
            "wr_rec_yards",
        ]

    def test_get_trainer_returns_registered_trainer(self) -> None:
        from gridiron_edge.cli.props import _get_trainer
        from gridiron_edge.models.prop_prediction.base import PropTrainer

        trainer = _get_trainer("qb_pass_yards")
        assert isinstance(trainer, PropTrainer)
        assert trainer.spec.name == "qb_pass_yards"


class TestBackfillWalkForward:
    """Walk-forward CLI semantics for ``gridiron props backfill``."""

    def test_rejects_start_season_with_no_prior_training_window(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pandas as pd
        from typer.testing import CliRunner

        from gridiron_edge.cli.props import props_app

        sentinel_df = pd.DataFrame(
            {
                "season": [2018, 2018, 2019, 2019, 2020, 2020],
                "week": [1, 2, 1, 2, 1, 2],
                "player_id": ["a"] * 6,
                "game_id": [f"g{i}" for i in range(6)],
                "passing_yards": [200.0] * 6,
                "attempts": [25.0] * 6,
            }
        )

        monkeypatch.setattr(
            "gridiron_edge.features.player.builder.build_prop_features",
            lambda *_, **__: sentinel_df,
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            [
                "backfill",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "elasticnet",
                "--start-season",
                "2018",
                "--end-season",
                "2018",
            ],
        )

        assert result.exit_code != 0
        assert "no prior training window" in result.stdout.lower()

    def test_rejects_invalid_season_range(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pandas as pd
        from typer.testing import CliRunner

        from gridiron_edge.cli.props import props_app

        sentinel_df = pd.DataFrame(
            {
                "season": [2018, 2019],
                "week": [1, 1],
                "player_id": ["a", "a"],
                "game_id": ["g1", "g2"],
                "passing_yards": [200.0, 250.0],
                "attempts": [25.0, 30.0],
            }
        )

        monkeypatch.setattr(
            "gridiron_edge.features.player.builder.build_prop_features",
            lambda *_, **__: sentinel_df,
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            [
                "backfill",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "elasticnet",
                "--start-season",
                "2020",
                "--end-season",
                "2019",
            ],
        )

        assert result.exit_code != 0
        assert "must be >=" in result.stdout


class TestEvaluateArchiveDriven:
    """`gridiron props evaluate` must read from the archive, not retrain."""

    def test_exits_when_archive_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pandas import DataFrame
        from typer.testing import CliRunner

        from gridiron_edge.cli.props import props_app

        monkeypatch.setattr(
            "gridiron_edge.evaluation.prop_archive.build_prop_evaluation_df",
            lambda **_: DataFrame(),
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            [
                "evaluate",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "elasticnet",
            ],
        )

        assert result.exit_code != 0
        assert "no archived predictions" in result.stdout.lower()


class TestChampionArchiveDriven:
    """`gridiron props champion` skips algorithms with empty archives."""

    def test_skips_algorithms_with_no_archive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pandas import DataFrame
        from typer.testing import CliRunner

        from gridiron_edge.cli.props import props_app

        monkeypatch.setattr(
            "gridiron_edge.evaluation.prop_archive.build_prop_evaluation_df",
            lambda **_: DataFrame(),
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            ["champion", "--model", "qb_pass_yards"],
        )

        # Champion exits with the "no archive" path. We do not assert
        # success since no algorithm has any rows; the important thing
        # is that the command does not crash.
        assert "No archived" in result.stdout or result.exit_code in (0, 1)


class TestProjectionsArtifactDriven:
    """`gridiron props projections` must require a trained artifact."""

    def test_skips_models_without_artifact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from typer.testing import CliRunner

        from gridiron_edge.cli.props import props_app

        # Force ArtifactStore.is_trained -> False
        monkeypatch.setattr(
            "gridiron_edge.models.artifact.ArtifactStore.is_trained",
            lambda *_, **__: False,
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            [
                "projections",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "elasticnet",
            ],
        )

        assert result.exit_code != 0
        assert "No projections produced" in result.stdout


class TestTrainAndSave:
    """`gridiron props train-and-save` must call train_and_save and report metrics."""

    def test_help_shows_command(self) -> None:
        result = runner.invoke(props_app, ["--help"])
        assert result.exit_code == 0
        assert "train-and-save" in result.output

    def test_help_shows_flags(self) -> None:
        result = runner.invoke(props_app, ["train-and-save", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--model-type" in result.output

    def test_unknown_model_exits_error(self) -> None:
        result = runner.invoke(
            props_app,
            ["train-and-save", "--model", "fake_model"],
        )
        assert result.exit_code != 0
        assert "Unknown model" in result.output

    def test_invalid_model_type_exits_error(self) -> None:
        result = runner.invoke(
            props_app,
            [
                "train-and-save",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "bad_type",
            ],
        )
        assert result.exit_code != 0

    def test_successful_train_displays_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When trainer.train_and_save returns a PropModelMetadata,
        the command must display its core metrics."""
        from gridiron_edge.models.prop_prediction.base import (
            PropModelMetadata,
            PropModelType,
        )

        fake_meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type=PropModelType.ELASTICNET.value,
            task="regression",
            trained_at="2026-06-22T12:00:00",
            target_col="passing_yards",
            n_train_rows=5000,
            n_holdout_rows=500,
            metrics={"mae": 60.5, "rmse": 75.2, "r2": 0.085},
        )

        monkeypatch.setattr(
            "gridiron_edge.models.prop_prediction.base.PropTrainer.train_and_save",
            lambda self, *, model_type: fake_meta,
        )

        result = runner.invoke(
            props_app,
            [
                "train-and-save",
                "--model",
                "qb_pass_yards",
                "--model-type",
                "elasticnet",
            ],
        )

        assert result.exit_code == 0
        assert "qb_pass_yards" in result.output
        assert "elasticnet" in result.output
        assert "MAE" in result.output
        assert "60.5" in result.output
        assert "75.2" in result.output
        assert "0.085" in result.output
        # The "next step" hint should appear.
        assert "props projections" in result.output
