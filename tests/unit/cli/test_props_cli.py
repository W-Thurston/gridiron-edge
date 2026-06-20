# tests/unit/cli/test_props_cli.py

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
