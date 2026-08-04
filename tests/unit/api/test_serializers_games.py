"""Tests for schedule-complete /games serializers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridiron_edge.api.serializers.games import (
    _derive_day_of_week,
    serialize_game_detail,
    serialize_game_summary,
    serialize_games_list,
)


def _row() -> dict[str, object]:
    return {
        "game_id": "2026_01_KC_LAC",
        "season": "2026-2027",
        "week": 1,
        "game_day_of_week": "Saturday",
        "game_date": "2026-09-05",
        "game_time": "18:00:00",
        "stadium": "SoFi Stadium",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "win_status": "available",
        "win_selection_status": "selected",
        "away_win_prob": 0.45,
        "home_win_prob": 0.55,
        "win_model_name": "win_prob",
        "win_model_type": "elo",
        "win_event_id": "win-event",
        "win_run_id": "win-run",
        "win_generated_at": "2026-08-01T00:00:00+00:00",
        "win_role": "live",
        "spread_status": "available",
        "model_spread": -2.5,
        "spread_uncertainty": 13.0,
        "spread_source_event_id": "win-event",
        "spread_model_name": "win_prob",
        "spread_model_type": "elo",
        "spread_calibration_key": "win_prob_elo",
        "spread_calibration_updated_at": "2026-08-01T00:00:00+00:00",
        "total_status": "available",
        "total_selection_status": "selected",
        "model_total": 47.5,
        "total_uncertainty": 12.0,
        "total_model_name": "total",
        "total_model_type": "random_forest",
        "total_event_id": "total-event",
        "total_run_id": "total-run",
        "total_generated_at": "2026-08-01T00:00:00+00:00",
        "total_role": "live",
        "total_uncertainty_trained_at": "2026-07-31T00:00:00+00:00",
        "projected_score_status": "available",
        "projected_home_score": 25.0,
        "projected_away_score": 22.5,
    }


def test_serializes_separate_component_provenance() -> None:
    summary = serialize_game_summary(_row())

    assert summary.away_team == "Kansas City Chiefs"
    assert summary.win.event_id == "win-event"
    assert summary.win.model_type == "elo"
    assert summary.total.event_id == "total-event"
    assert summary.total.model_type == "random_forest"
    assert summary.spread.source_event_id == "win-event"


def test_unavailable_prediction_values_do_not_remove_status_blocks() -> None:
    row = _row()
    row.update(
        {
            "win_status": "forecast_missing",
            "win_selection_status": "no_eligible_candidate",
            "home_win_prob": np.nan,
            "away_win_prob": pd.NA,
            "win_event_id": pd.NA,
            "total_status": "forecast_missing",
            "model_total": pd.NA,
            "total_event_id": pd.NA,
            "projected_score_status": "spread_and_total_unavailable",
            "projected_home_score": pd.NA,
            "projected_away_score": pd.NA,
        }
    )

    summary = serialize_game_summary(row)

    assert summary.win.status == "forecast_missing"
    assert summary.win.home_win_prob is None
    assert summary.win.event_id is None
    assert summary.total.status == "forecast_missing"
    assert summary.total.model_total is None
    assert summary.projected_score.home is None


def test_list_serializes_every_scheduled_row() -> None:
    missing = _row() | {
        "game_id": "2026_01_BUF_MIA",
        "win_status": "forecast_missing",
        "home_win_prob": pd.NA,
    }
    response = serialize_games_list(
        pd.DataFrame([_row(), missing]),
        season="2026-2027",
        week=1,
    )

    assert response.total == 2
    assert [item.game_id for item in response.items] == [
        "2026_01_KC_LAC",
        "2026_01_BUF_MIA",
    ]
    assert response.items[1].win.status == "forecast_missing"


def test_detail_uses_persisted_schedule_metadata() -> None:
    detail = serialize_game_detail(_row())

    assert detail.day_of_week == "Saturday"
    assert detail.kick == "18:00:00"
    assert detail.venue == "SoFi Stadium"
    assert detail.response_meta is not None
    assert "kick" not in detail.response_meta.field_status
    assert "venue" not in detail.response_meta.field_status


def test_derive_day_of_week_handles_missing_and_invalid_values() -> None:
    assert _derive_day_of_week("2026-09-05") == "Saturday"
    assert _derive_day_of_week(None) is None
    assert _derive_day_of_week(np.nan) is None
    assert _derive_day_of_week("invalid") is None
