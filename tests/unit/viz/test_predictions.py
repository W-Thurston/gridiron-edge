# tests/unit/viz/test_predictions.py

"""Tests for pure weekly-product visualization adaptation."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.viz.predictions import (
    _optional_float,
    build_weekly_product_display_frame,
    render_predictions_html,
)


def _product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["product-1", "product-1"],
            "product_run_id": ["run-1", "run-1"],
            "product_generated_at": ["2026-08-04T16:39:38+00:00"] * 2,
            "season": ["2026-2027"] * 2,
            "week": [1, 1],
            "game_id": ["g1", "g2"],
            "game_day_of_week": ["Sunday", "Monday"],
            "game_date": ["2026-09-06", "2026-09-07"],
            "game_time": ["13:00:00", "20:15:00"],
            "away_team": ["Kansas City Chiefs", "Baltimore Ravens"],
            "home_team": ["Los Angeles Chargers", "Buffalo Bills"],
            "neutral_site": [0, 1],
            "away_moneyline": [120.0, pd.NA],
            "home_moneyline": [-140.0, pd.NA],
            "win_status": ["available", "unavailable"],
            "away_win_prob": pd.Series([0.55, pd.NA], dtype="Float64"),
            "home_win_prob": pd.Series([0.45, pd.NA], dtype="Float64"),
            "win_model_name": ["win_prob", pd.NA],
            "win_model_type": ["logistic", pd.NA],
            "win_event_id": ["event-1", pd.NA],
            "win_run_id": ["run-1", pd.NA],
        }
    )


def test_maps_persisted_product_to_display_contract() -> None:
    result = build_weekly_product_display_frame(_product())

    ready = result.loc[result["GAME_ID"] == "g1"].iloc[0]
    assert ready["AWAY_TEAM"] == "Kansas City Chiefs"
    assert ready["HOME_TEAM"] == "Los Angeles Chargers"
    assert ready["AWAY_WIN_PROB"] == pytest.approx(0.55)
    assert ready["HOME_WIN_PROB"] == pytest.approx(0.45)
    assert ready["AWAY_TEAM_WIN_PROB"] == "55.0 %"
    assert ready["HOME_TEAM_WIN_PROB"] == "45.0 %"
    assert ready["AWAY_MONEYLINE"] == pytest.approx(120.0)
    assert ready["HOME_MONEYLINE"] == pytest.approx(-140.0)
    assert ready["win_model_type"] == "logistic"
    assert ready["product_id"] == "product-1"


def test_preserves_unavailable_probabilities() -> None:
    result = build_weekly_product_display_frame(_product())
    missing = result.loc[result["GAME_ID"] == "g2"].iloc[0]

    assert pd.isna(missing["AWAY_WIN_PROB"])
    assert pd.isna(missing["HOME_WIN_PROB"])
    assert pd.isna(missing["AWAY_TEAM_WIN_PROB"])
    assert pd.isna(missing["HOME_TEAM_WIN_PROB"])


def test_does_not_mutate_product() -> None:
    product = _product()
    original = product.copy(deep=True)

    build_weekly_product_display_frame(product)

    pd.testing.assert_frame_equal(product, original)


def test_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="away_win_prob"):
        build_weekly_product_display_frame(_product().drop(columns=["away_win_prob"]))


def test_maps_absent_moneylines_to_unavailable_display_values() -> None:
    product = _product().drop(columns=["away_moneyline", "home_moneyline"])

    result = build_weekly_product_display_frame(product)

    assert result["AWAY_MONEYLINE"].isna().all()
    assert result["HOME_MONEYLINE"].isna().all()


def test_optional_float_handles_absent_null_and_supplied_values() -> None:
    row = pd.Series({"NULL_VALUE": pd.NA, "PRICE": 120.0})

    assert _optional_float(row, "ABSENT") is None
    assert _optional_float(row, "NULL_VALUE") is None
    assert _optional_float(row, "PRICE") == pytest.approx(120.0)


def test_html_renders_without_moneyline_columns(tmp_path) -> None:
    display = build_weekly_product_display_frame(
        _product().iloc[[0]].drop(columns=["away_moneyline", "home_moneyline"])
    )
    original = display.copy(deep=True)

    output = render_predictions_html(
        display,
        year="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert output.is_file()
    html = output.read_text(encoding="utf-8")
    assert "Chiefs" in html
    assert "Chargers" in html
    assert "+100" not in html
    assert "-110" not in html
    pd.testing.assert_frame_equal(display, original)


def test_html_preserves_supplied_moneylines_for_underdog_highlight(tmp_path) -> None:
    display = build_weekly_product_display_frame(_product().iloc[[0]])

    output = render_predictions_html(
        display,
        year="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert output.is_file()
    assert "outline: 2px solid gold" in output.read_text(encoding="utf-8")
