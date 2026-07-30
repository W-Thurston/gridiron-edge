# tests/unit/cli/test_output.py

"""Tests for output-rendering CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from gridiron_edge.cli.output import output_predictions


def _predictions() -> pd.DataFrame:
    """Create one display-oriented prediction row."""
    return pd.DataFrame(
        {
            "GAME_ID": ["2026_01_KC_LAC"],
            "GAME_DATE": ["2026-09-05"],
            "AWAY_TEAM": ["Kansas City Chiefs"],
            "HOME_TEAM": ["Los Angeles Chargers"],
            "AWAY_TEAM_ELO": [1520.0],
            "HOME_TEAM_ELO": [1480.0],
            "AWAY_WIN_PROB": [0.55],
            "HOME_WIN_PROB": [0.45],
        }
    )


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.viz.predictions.build_predictions_df")
def test_renders_both_formats_by_default(
    mock_build: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    predictions = _predictions()
    mock_build.return_value = predictions
    mock_image.return_value = Path("/tmp/predictions.png")
    mock_html.return_value = Path("/tmp/predictions.html")

    output_predictions(
        year="2026-2027",
        week=1,
        format=[],
    )

    mock_build.assert_called_once_with(
        year="2026-2027",
        week=1,
    )
    mock_image.assert_called_once_with(
        predictions,
        year="2026-2027",
        week=1,
    )
    mock_html.assert_called_once_with(
        predictions,
        year="2026-2027",
        week=1,
    )


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.viz.predictions.build_predictions_df")
def test_renders_only_requested_format(
    mock_build: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    predictions = _predictions()
    mock_build.return_value = predictions
    mock_image.return_value = Path("/tmp/predictions.png")

    output_predictions(
        year="2026-2027",
        week=1,
        format=["png"],
    )

    mock_image.assert_called_once_with(
        predictions,
        year="2026-2027",
        week=1,
    )
    mock_html.assert_not_called()


@patch("gridiron_edge.viz.predictions.render_predictions_image")
@patch("gridiron_edge.viz.predictions.render_predictions_html")
@patch("gridiron_edge.viz.predictions.build_predictions_df")
def test_empty_predictions_produce_no_outputs(
    mock_build: MagicMock,
    mock_html: MagicMock,
    mock_image: MagicMock,
) -> None:
    mock_build.return_value = pd.DataFrame()

    output_predictions(
        year="2026-2027",
        week=1,
        format=[],
    )

    mock_image.assert_not_called()
    mock_html.assert_not_called()
