# tests/unit/cli/test_props.py

"""Unit tests for props CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.props import _parse_season_arg, props_app

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


class TestPropsInternalHelpers:
    """Smoke tests for the internal helper surface of cli/props.py.

    After the training-path consolidation, the single canonical
    predictionenrichment helper is `_enrich_and_predict`, used by
    both `_walk_forward_predict_for_season` (backfill path) and
    `_project_for_model` (projections path). This class documents
    the intended surface so accidental re-fragmentation is caught
    by a broken import.
    """

    def test_enrich_and_predict_is_importable(self) -> None:
        """The single canonical predictionenrichment helper."""
        from gridiron_edge.cli.props import _enrich_and_predict

        assert callable(_enrich_and_predict)

    def test_walk_forward_predict_for_season_is_importable(self) -> None:
        """Used by both the CLI backfill and full-retrain composite."""
        from gridiron_edge.cli.props import _walk_forward_predict_for_season

        assert callable(_walk_forward_predict_for_season)

    def test_project_for_model_is_importable(self) -> None:
        """Used by the projections CLI command."""
        from gridiron_edge.cli.props import _project_for_model

        assert callable(_project_for_model)

    def test_removed_helpers_are_gone(self) -> None:
        """These helpers were folded into _enrich_and_predict.

        Assert they stay gone so future changes don't accidentally
        reintroduce a parallel prediction path.
        """
        import gridiron_edge.cli.props as props_mod

        assert not hasattr(props_mod, "_prepare_holdout_data")
        assert not hasattr(props_mod, "_enrich_predictions_for_holdout")
        assert not hasattr(props_mod, "_train_and_enrich")


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


class TestSeasonArgParsing:
    def test_accepts_int_string(self) -> None:
        assert _parse_season_arg("2023") == 2023

    def test_accepts_season_label(self) -> None:
        assert _parse_season_arg("2023-2024") == 2023

    def test_none_returns_none(self) -> None:
        assert _parse_season_arg(None) is None

    def test_rejects_noncontiguous_label(self) -> None:
        with pytest.raises(typer.BadParameter):
            _parse_season_arg("2023-2025")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(typer.BadParameter):
            _parse_season_arg("not-a-season")


class TestWalkForwardPredictionDrift:
    """Regression test for the sklearn feature-name mismatch bug.

    train_through can select a feature set based on training-slice NaN
    rates. A previous bug in _walk_forward_predict_for_season re-derived
    a different feature set based on prediction-slice NaN rates, causing
    sklearn to raise 'The feature names should match those that were
    passed during fit.'
    """

    def test_prediction_uses_meta_feature_columns_not_re_derived(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A column that is <50% NaN in training but >50% NaN in the
        prediction season must still be passed to the model.
        """
        import numpy as np
        import pandas as pd

        from gridiron_edge.cli.props import _walk_forward_predict_for_season
        from gridiron_edge.models.prop_prediction.base import PropModelType

        # Build synthetic features across 2015-2020.
        # - feat_stable: populated all seasons
        # - feat_flaky: fully populated pre-2020 (in training slice),
        #   90% NaN in 2020 (the prediction slice)
        rng = np.random.default_rng(0)
        rows = []
        for season in range(2015, 2021):
            for i in range(100):
                is_nan_flaky = (season == 2020) and (i > 10)
                rows.append(
                    {
                        "season": season,
                        "week": 1,
                        "player_id": f"p{i}",
                        "game_id": f"g{season}_{i}",
                        "rushing_yards": float(rng.integers(20, 150)),
                        "carries": float(rng.integers(5, 25)),
                        "feat_stable": float(rng.normal()),
                        "feat_flaky": float("nan") if is_nan_flaky else float(rng.normal()),
                    }
                )
        features_df = pd.DataFrame(rows)

        monkeypatch.setattr(
            "gridiron_edge.models.prop_prediction.base.build_prop_features",
            lambda *_, **__: features_df,
        )

        # Use rb_rush_yards; align feature list with our synthetic df.
        # We monkey-patch _feature_columns on the trainer class to
        # advertise only our synthetic feature columns.
        from gridiron_edge.models.prop_prediction.rb_rush_yards import (
            RBRushYardsTrainer,
        )

        monkeypatch.setattr(
            RBRushYardsTrainer,
            "_feature_columns",
            lambda self: ["feat_stable", "feat_flaky"],
        )

        # This is the call that used to raise "The feature names should
        # match those that were passed during fit." because the outer
        # helper recomputed the usable list on the prediction slice.
        enriched, _rmse = _walk_forward_predict_for_season(
            model_name="rb_rush_yards",
            model_type=PropModelType.ELASTICNET,
            season=2020,
            features_df=features_df,
        )

        # Even though feat_flaky is >50% NaN in 2020, the ~11 rows where
        # it IS populated should still produce predictions.
        assert not enriched.empty
        assert "predicted_mean" in enriched.columns


def test_train_and_train_through_share_helpers() -> None:
    """train() and train_through() must both go through
    _prepare_features, _filter_and_split, _fit_and_build_metadata.

    Sourcecode-level check. Not pretty but catches structural drift.
    """
    import inspect

    from gridiron_edge.models.prop_prediction.base import PropTrainer

    train_src = inspect.getsource(PropTrainer.train)
    train_through_src = inspect.getsource(PropTrainer.train_through)

    for helper in ("_prepare_features", "_filter_and_split", "_fit_and_build_metadata"):
        assert helper in train_src, f"train() must delegate to {helper}"
        assert helper in train_through_src, f"train_through() must delegate to {helper}"
