# tests/unit/cli/test_props_cli.py

"""Unit tests for props CLI commands."""

from __future__ import annotations

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
