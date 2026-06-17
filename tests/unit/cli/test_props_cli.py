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
