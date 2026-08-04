# tests/unit/cli/test_output.py

"""Tests for pure weekly-product rendering commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.output import output_predictions


def _product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["product-1"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
        }
    )


def _invoke(*args: str):
    app = typer.Typer()
    app.command()(output_predictions)
    return CliRunner().invoke(app, list(args))


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.viz.predictions.build_weekly_product_display_frame")
@patch("gridiron_edge.datasets.loaders.load_current_weekly_product")
@patch("gridiron_edge.core.settings.get_settings")
def test_renders_both_formats_from_selected_product(
    mock_settings: MagicMock,
    mock_load: MagicMock,
    mock_adapt: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    repo = Path("/repo")
    product = _product()
    display = pd.DataFrame({"GAME_ID": ["g1"]})
    mock_settings.return_value.repo_root = repo
    mock_load.return_value = product
    mock_adapt.return_value = display
    mock_image.return_value = repo / "predictions.png"
    mock_html.return_value = repo / "predictions.html"

    result = _invoke("--season", "2026-2027", "--week", "1")

    assert result.exit_code == 0, result.output
    mock_load.assert_called_once_with(repo, season="2026-2027", week=1)
    mock_adapt.assert_called_once_with(product)
    mock_image.assert_called_once_with(display, year="2026-2027", week=1, repo=repo)
    mock_html.assert_called_once_with(display, year="2026-2027", week=1, repo=repo)


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.viz.predictions.build_weekly_product_display_frame")
@patch("gridiron_edge.datasets.loaders.load_current_weekly_product")
@patch("gridiron_edge.core.settings.get_settings")
def test_renders_only_requested_format(
    mock_settings: MagicMock,
    mock_load: MagicMock,
    mock_adapt: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    repo = Path("/repo")
    mock_settings.return_value.repo_root = repo
    mock_load.return_value = _product()
    mock_adapt.return_value = pd.DataFrame({"GAME_ID": ["g1"]})

    result = _invoke("--season", "2026-2027", "--week", "1", "--format", "png")

    assert result.exit_code == 0, result.output
    mock_image.assert_called_once()
    mock_html.assert_not_called()


@patch("gridiron_edge.datasets.loaders.load_current_weekly_product")
def test_invalid_format_fails_before_loading(mock_load: MagicMock) -> None:
    result = _invoke("--season", "2026-2027", "--week", "1", "--format", "pdf")

    assert result.exit_code != 0
    assert "Unsupported format(s): pdf" in result.output
    mock_load.assert_not_called()


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.datasets.loaders.load_current_weekly_product")
def test_missing_selected_product_exits_nonzero(
    mock_load: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    mock_load.side_effect = FileNotFoundError("No current weekly product selected")

    result = _invoke("--season", "2026-2027", "--week", "1")

    assert result.exit_code != 0
    mock_image.assert_not_called()
    mock_html.assert_not_called()
