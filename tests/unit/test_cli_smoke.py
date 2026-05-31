from typer.testing import CliRunner

from gridiron_edge.cli import app

runner = CliRunner()


def test_gridiron_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout


def test_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "nflverse-games" in result.stdout


def test_ratings_elo_help() -> None:
    result = runner.invoke(app, ["ratings", "elo", "--help"])
    assert result.exit_code == 0
    assert "predict" in result.stdout
