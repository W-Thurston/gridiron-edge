# tests/unit/cli/test_ratings.py

"""Tests for ratings CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from gridiron_edge.cli.ratings import ratings_app

runner = CliRunner()


@patch("gridiron_edge.ratings.elo.predict.predict_elo_only")
def test_elo_predict_delegates_to_domain_output(
    mock_predict: MagicMock,
    tmp_path: Path,
) -> None:
    mock_predict.return_value = tmp_path / "week_01_predictions.csv"

    result = runner.invoke(
        ratings_app,
        [
            "elo",
            "predict",
            "--year",
            "2026-2027",
            "--week",
            "1",
        ],
    )

    assert result.exit_code == 0
    mock_predict.assert_called_once_with(
        year="2026-2027",
        week=1,
    )


def test_elo_predict_requires_year() -> None:
    result = runner.invoke(
        ratings_app,
        [
            "elo",
            "predict",
            "--week",
            "1",
        ],
    )

    assert result.exit_code == 2


def test_elo_predict_requires_week() -> None:
    result = runner.invoke(
        ratings_app,
        [
            "elo",
            "predict",
            "--year",
            "2026-2027",
        ],
    )

    assert result.exit_code == 2
