# tests/unit/market/test_weekly_edge_service.py

"""Tests for the persisted weekly edge domain service."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import EdgeResult
from gridiron_edge.market.weekly_edge_service import build_weekly_edge_result

SEASON = "2026-2027"
WEEK = 8
GAME_ID = "2026_08_KC_LAC"


def _product() -> DataFrame:
    return DataFrame(
        {
            "product_schema_version": [1],
            "product_id": ["product-1"],
            "product_run_id": ["product-run-1"],
            "product_generated_at": [datetime(2026, 10, 20, tzinfo=UTC)],
            "season": [SEASON],
            "week": [WEEK],
            "game_id": [GAME_ID],
            "game_date": ["2026-10-22"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "win_status": ["available"],
            "away_win_prob": [0.40],
            "home_win_prob": [0.60],
            "win_model_name": ["win_prob"],
            "win_model_type": ["elo"],
            "win_event_id": ["win-1"],
            "win_run_id": ["win-run-1"],
            "spread_status": ["available"],
            "model_spread": [-3.0],
            "spread_uncertainty": [13.5],
            "spread_source_event_id": ["win-1"],
            "spread_model_name": ["win_prob"],
            "spread_model_type": ["elo"],
            "spread_calibration_key": ["win_prob_elo"],
            "spread_calibration_updated_at": ["2026-07-30T12:00:00+00:00"],
            "total_status": ["available"],
            "model_total": [44.0],
            "total_uncertainty": [12.8],
            "total_model_name": ["total"],
            "total_model_type": ["xgboost"],
            "total_event_id": ["total-1"],
            "total_run_id": ["total-run-1"],
            "total_uncertainty_trained_at": ["2026-07-01T14:20:00"],
        }
    )


def _markets() -> DataFrame:
    base: dict[str, object] = {
        "fetched_at": datetime(2026, 10, 20, 12, tzinfo=UTC),
        "provider": "nflverse",
        "provider_event_id": None,
        "sportsbook": None,
        "sportsbook_updated_at": pd.NaT,
        "commence_time": pd.NaT,
        "is_live": False,
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_ID,
        "game_date": "2026-10-22",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
    }
    return DataFrame(
        [
            {**base, "market": "moneyline", "side": "home", "odds": -120.0, "line": None},
            {**base, "market": "moneyline", "side": "away", "odds": 105.0, "line": None},
            {**base, "market": "spread", "side": "home", "odds": -110.0, "line": -2.5},
            {**base, "market": "spread", "side": "away", "odds": -110.0, "line": 2.5},
            {**base, "market": "total", "side": "over", "odds": -110.0, "line": 45.5},
            {**base, "market": "total", "side": "under", "odds": -110.0, "line": 45.5},
        ]
    )


def _empty_result(blocker: EdgeDiagnosticBlocker) -> EdgeResult:
    diagnostics = EdgeDiagnostics(
        season=SEASON,
        week=WEEK,
        prediction_game_count=0,
        market_game_count=0,
        matched_game_count=0,
        complete_moneyline_count=0,
        complete_spread_count=0,
        complete_total_count=0,
        eligible_market_count=0,
        calculated_edge_count=0,
        positive_edge_count=0,
        filtered_edge_count=0,
        state=EdgeResultState.BLOCKED,
        blockers=(blocker,),
    )
    return EdgeResult(rows=DataFrame(), diagnostics=diagnostics)


def test_service_uses_selected_product_market_snapshot_and_uncertainties(
    tmp_path: Path,
) -> None:
    expected = _empty_result(EdgeDiagnosticBlocker.NO_MARKET_DATA)
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            return_value=_product(),
        ) as mock_product,
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=_markets(),
        ) as mock_markets,
        patch(
            "gridiron_edge.market.weekly_edge_service.build_edge_result",
            return_value=expected,
        ) as mock_build,
    ):
        result = build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    assert result is expected
    mock_product.assert_called_once_with(tmp_path, season=SEASON, week=WEEK)
    mock_markets.assert_called_once_with(repo=tmp_path)
    args, kwargs = mock_build.call_args
    adapted = args[0]
    markets = args[1]
    assert adapted.loc[0, "game_id"] == GAME_ID
    assert adapted.loc[0, "home_win_prob"] == 0.60
    assert adapted.loc[0, "model_spread"] == -3.0
    assert adapted.loc[0, "model_total"] == 44.0
    assert adapted.loc[0, "model_name"] == "win_prob"
    assert adapted.loc[0, "model_type"] == "elo"
    assert adapted.loc[0, "total_model_type"] == "xgboost"
    assert markets.equals(_markets())
    assert kwargs["margin_std"] == 13.5
    assert kwargs["total_std"] == 12.8


def test_bankroll_and_filter_options_flow_to_edge_result(tmp_path: Path) -> None:
    expected = _empty_result(EdgeDiagnosticBlocker.NO_MARKET_DATA)
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            return_value=_product(),
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=_markets(),
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.build_edge_result",
            return_value=expected,
        ) as mock_build,
    ):
        build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            bankroll=2500.0,
            kelly_multiplier=0.10,
            min_ev=0.03,
            repo=tmp_path,
        )

    kwargs = mock_build.call_args.kwargs
    assert kwargs["bankroll"] == 2500.0
    assert kwargs["kelly_multiplier"] == 0.10
    assert kwargs["min_ev"] == 0.03


def test_missing_current_selection_becomes_no_predictions(tmp_path: Path) -> None:
    missing = FileNotFoundError(
        f"No current weekly product selected for season={SEASON!r}, week={WEEK}."
    )
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            side_effect=missing,
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=_markets(),
        ),
    ):
        result = build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    assert result.rows.empty
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.NO_PREDICTIONS,)


def test_missing_selected_artifact_is_not_hidden(tmp_path: Path) -> None:
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            side_effect=FileNotFoundError("Weekly product artifact is missing"),
        ),
        pytest.raises(FileNotFoundError, match="artifact is missing"),
    ):
        build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )


def test_missing_market_snapshot_becomes_no_market_data(tmp_path: Path) -> None:
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            return_value=_product(),
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=None,
        ),
    ):
        result = build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    assert result.rows.empty
    assert result.diagnostics.blockers == (EdgeDiagnosticBlocker.NO_MARKET_DATA,)


def test_unavailable_component_statuses_disable_only_their_market(
    tmp_path: Path,
) -> None:
    product = _product()
    product.loc[0, "spread_status"] = "calibration_unavailable"
    product.loc[0, ["model_spread", "spread_uncertainty"]] = pd.NA
    product.loc[0, "total_status"] = "uncertainty_unavailable"
    product.loc[0, "total_uncertainty"] = pd.NA

    expected = _empty_result(EdgeDiagnosticBlocker.INCOMPLETE_MARKETS)
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            return_value=product,
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=_markets(),
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.build_edge_result",
            return_value=expected,
        ) as mock_build,
    ):
        build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    adapted = mock_build.call_args.args[0]
    kwargs = mock_build.call_args.kwargs
    assert pd.isna(adapted.loc[0, "model_spread"])
    assert pd.isna(adapted.loc[0, "model_total"])
    assert kwargs["margin_std"] is None
    assert kwargs["total_std"] is None
    assert adapted.loc[0, "home_win_prob"] == 0.60


@pytest.mark.parametrize(
    ("column", "values", "label"),
    [
        ("spread_uncertainty", [13.5, 14.0], "spread uncertainty"),
        ("total_uncertainty", [12.8, 13.1], "total uncertainty"),
    ],
)
def test_mixed_available_uncertainties_are_rejected(
    tmp_path: Path,
    column: str,
    values: list[float],
    label: str,
) -> None:
    product = pd.concat([_product(), _product()], ignore_index=True)
    product.loc[1, "game_id"] = "2026_08_BUF_MIA"
    product[column] = values
    with (
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_weekly_product",
            return_value=product,
        ),
        patch(
            "gridiron_edge.market.weekly_edge_service.load_current_odds",
            return_value=_markets(),
        ),
        pytest.raises(ValueError, match=label),
    ):
        build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )
