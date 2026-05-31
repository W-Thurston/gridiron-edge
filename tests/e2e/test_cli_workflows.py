# tests/e2e/test_cli_workflows.py
"""E2E: CLI subcommand smoke tests covering all top-level commands."""

from __future__ import annotations

from typer.testing import CliRunner

from gridiron_edge.cli import app

runner = CliRunner()


class TestCliHelp:
    """Every top-level command and subcommand should display help without error."""

    def test_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_ingest_help(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0

    def test_ratings_help(self) -> None:
        result = runner.invoke(app, ["ratings", "--help"])
        assert result.exit_code == 0

    def test_ratings_elo_help(self) -> None:
        result = runner.invoke(app, ["ratings", "elo", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_features_help(self) -> None:
        result = runner.invoke(app, ["features", "--help"])
        assert result.exit_code == 0


class TestCliSubcommandDiscovery:
    """Verify expected subcommands are registered."""

    def test_ingest_has_nflverse_games(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert "nflverse-games" in result.stdout

    def test_ratings_elo_has_predict(self) -> None:
        result = runner.invoke(app, ["ratings", "elo", "--help"])
        assert "predict" in result.stdout

    def test_root_has_expected_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("ingest", "ratings", "evaluate"):
            assert cmd in result.stdout, f"Missing command: {cmd}"
