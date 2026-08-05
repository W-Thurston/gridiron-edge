# tests/unit/cli/test_api.py
"""Unit tests for `gridiron api serve`.

Covers:
- `gridiron api --help` renders without import errors.
- `gridiron api serve --help` renders and lists expected options.
- `gridiron api serve` invokes uvicorn.run with the right arguments.
- The `api` sub-app is wired into the top-level `gridiron` CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.api import api_app
from gridiron_edge.cli.main import app as main_app

runner = CliRunner()


class TestApiHelp:
    def test_api_help_renders(self) -> None:
        result = runner.invoke(api_app, ["--help"])
        assert result.exit_code == 0
        assert "Run the Gridiron Edge API" in result.stdout

    def test_serve_help_lists_options(self) -> None:
        result = runner.invoke(api_app, ["serve", "--help"])
        assert result.exit_code == 0
        for option in ("--host", "--port", "--reload", "--log-level"):
            assert option in result.stdout


class TestServeInvokesUvicorn:
    def test_default_invocation(self) -> None:
        with patch("gridiron_edge.cli.api.uvicorn.run") as mock_run:
            result = runner.invoke(main_app, ["api", "serve"])
        assert result.exit_code == 0, result.stdout
        mock_run.assert_called_once_with(
            "gridiron_edge.api.app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
        )

    def test_custom_host_and_port(self) -> None:
        with patch("gridiron_edge.cli.api.uvicorn.run") as mock_run:
            result = runner.invoke(
                main_app,
                ["api", "serve", "--host", "0.0.0.0", "--port", "9001"],
            )
        assert result.exit_code == 0, result.stdout
        mock_run.assert_called_once_with(
            "gridiron_edge.api.app:app",
            host="0.0.0.0",
            port=9001,
            reload=False,
            log_level="info",
        )

    def test_reload_flag(self) -> None:
        with patch("gridiron_edge.cli.api.uvicorn.run") as mock_run:
            result = runner.invoke(main_app, ["api", "serve", "--reload"])
        assert result.exit_code == 0, result.stdout
        _, kwargs = mock_run.call_args
        assert kwargs["reload"] is True

    def test_log_level(self) -> None:
        with patch("gridiron_edge.cli.api.uvicorn.run") as mock_run:
            result = runner.invoke(
                main_app,
                ["api", "serve", "--log-level", "debug"],
            )
        assert result.exit_code == 0, result.stdout
        _, kwargs = mock_run.call_args
        assert kwargs["log_level"] == "debug"


class TestApiWiredIntoMainCli:
    def test_api_command_appears_in_top_level_help(self) -> None:
        """`gridiron --help` must list `api` as a subcommand."""
        result = runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "api" in result.stdout

    def test_api_serve_reachable_via_top_level(self) -> None:
        """`gridiron api serve --help` must work end-to-end."""
        result = runner.invoke(main_app, ["api", "serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout


class TestExportSchema:
    """Cover the OpenAPI schema export command."""

    def test_writes_schema_to_output(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.api import export_schema

        output = tmp_path / "test-schema.json"

        app = typer.Typer()
        app.command()(export_schema)

        runner = CliRunner()
        result = runner.invoke(app, ["--output", str(output)])

        assert result.exit_code == 0, result.output
        assert output.exists()

        # File is valid JSON and looks like an OpenAPI doc.
        schema = json.loads(output.read_text())
        assert "paths" in schema
        assert "openapi" in schema

    def test_default_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default output writes to api-schema.json in the CWD."""
        from gridiron_edge.cli.api import export_schema

        monkeypatch.chdir(tmp_path)

        app = typer.Typer()
        app.command()(export_schema)

        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "api-schema.json").exists()

    def test_deterministic_output(self, tmp_path: Path) -> None:
        """Two exports produce identical files (sort_keys=True)."""
        from gridiron_edge.cli.api import export_schema

        output_1 = tmp_path / "schema-1.json"
        output_2 = tmp_path / "schema-2.json"

        app = typer.Typer()
        app.command()(export_schema)

        runner = CliRunner()
        result_1 = runner.invoke(app, ["--output", str(output_1)])
        result_2 = runner.invoke(app, ["--output", str(output_2)])

        assert result_1.exit_code == 0
        assert result_2.exit_code == 0
        assert output_1.read_text() == output_2.read_text()

    def test_checked_in_schema_matches_application(self) -> None:
        """The checked-in OpenAPI artifact matches the live application contract."""
        from gridiron_edge.api.app import create_app

        project_root = Path(__file__).resolve().parents[3]
        checked_in = json.loads((project_root / "api-schema.json").read_text(encoding="utf-8"))

        assert checked_in == create_app().openapi()

    def test_short_flag(self, tmp_path: Path) -> None:
        """The -o short flag also works."""
        from gridiron_edge.cli.api import export_schema

        output = tmp_path / "schema.json"

        app = typer.Typer()
        app.command()(export_schema)

        runner = CliRunner()
        result = runner.invoke(app, ["-o", str(output)])

        assert result.exit_code == 0, result.output
        assert output.exists()
