# tests/unit/cli/test_output.py

"""Tests for pure weekly-product rendering commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
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
@patch(
    "gridiron_edge.evaluation."
    "forecast_evidence_disposition_store."
    "list_forecast_evidence_dispositions",
    return_value=(),
)
def test_renders_only_requested_format(
    mock_dispositions: MagicMock,
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
    mock_dispositions.assert_called_once_with(
        season="2026-2027",
        week=1,
        product_id="product-1",
        repo=repo,
    )


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


def test_known_defective_product_is_rejected_before_rendering() -> None:
    from gridiron_edge.evaluation.forecast_evidence_disposition import (
        ForecastEvidenceNotOperationalError,
    )

    repo = Path("/repo")
    product = _product()
    disposition = object()

    with (
        patch(
            "gridiron_edge.core.settings.get_settings",
        ) as mock_settings,
        patch(
            "gridiron_edge.datasets.loaders.load_current_weekly_product",
            return_value=product,
        ),
        patch(
            "gridiron_edge.evaluation."
            "forecast_evidence_disposition_store."
            "list_forecast_evidence_dispositions",
            return_value=(disposition,),
        ) as mock_dispositions,
        patch(
            "gridiron_edge.evaluation."
            "forecast_evidence_disposition."
            "require_operational_weekly_product",
            side_effect=ForecastEvidenceNotOperationalError(
                "Known-defective forecast evidence: product-1"
            ),
        ) as mock_require,
        patch(
            "gridiron_edge.viz.predictions.build_weekly_product_display_frame",
        ) as mock_adapt,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_image",
        ) as mock_image,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_html",
        ) as mock_html,
    ):
        mock_settings.return_value.repo_root = repo

        result = _invoke(
            "--season",
            "2026-2027",
            "--week",
            "1",
        )

    assert result.exit_code == 2
    assert "Known-defective forecast evidence" in result.output

    mock_dispositions.assert_called_once_with(
        season="2026-2027",
        week=1,
        product_id="product-1",
        repo=repo,
    )
    mock_require.assert_called_once_with(
        product,
        (disposition,),
    )
    mock_adapt.assert_not_called()
    mock_image.assert_not_called()
    mock_html.assert_not_called()


@pytest.mark.parametrize(
    "product_ids",
    [
        ["product-1", "product-2"],
        [" "],
    ],
)
def test_rendering_requires_one_product_identity(
    product_ids: list[str],
) -> None:
    repo = Path("/repo")
    product = pd.DataFrame(
        {
            "product_id": product_ids,
            "away_team": ["Away"] * len(product_ids),
            "home_team": ["Home"] * len(product_ids),
        }
    )

    with (
        patch(
            "gridiron_edge.core.settings.get_settings",
        ) as mock_settings,
        patch(
            "gridiron_edge.datasets.loaders.load_current_weekly_product",
            return_value=product,
        ),
        patch(
            "gridiron_edge.viz.predictions.build_weekly_product_display_frame",
        ) as mock_adapt,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_image",
        ) as mock_image,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_html",
        ) as mock_html,
    ):
        mock_settings.return_value.repo_root = repo

        result = _invoke(
            "--season",
            "2026-2027",
            "--week",
            "1",
        )

    assert result.exit_code == 2
    assert "one nonempty product_id" in result.output
    mock_adapt.assert_not_called()
    mock_image.assert_not_called()
    mock_html.assert_not_called()


def test_ambiguous_dispositions_remain_explicit() -> None:
    repo = Path("/repo")
    product = _product()

    with (
        patch(
            "gridiron_edge.core.settings.get_settings",
        ) as mock_settings,
        patch(
            "gridiron_edge.datasets.loaders.load_current_weekly_product",
            return_value=product,
        ),
        patch(
            "gridiron_edge.evaluation."
            "forecast_evidence_disposition_store."
            "list_forecast_evidence_dispositions",
            return_value=(object(), object()),
        ),
        patch(
            "gridiron_edge.evaluation."
            "forecast_evidence_disposition."
            "require_operational_weekly_product",
            side_effect=ValueError("Multiple forecast-evidence dispositions apply"),
        ),
        patch(
            "gridiron_edge.viz.predictions.build_weekly_product_display_frame",
        ) as mock_adapt,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_image",
        ) as mock_image,
        patch(
            "gridiron_edge.viz.predictions.render_predictions_html",
        ) as mock_html,
    ):
        mock_settings.return_value.repo_root = repo

        result = _invoke(
            "--season",
            "2026-2027",
            "--week",
            "1",
        )

    assert result.exit_code == 2
    assert "Multiple forecast-evidence dispositions apply" in result.output
    mock_adapt.assert_not_called()
    mock_image.assert_not_called()
    mock_html.assert_not_called()
