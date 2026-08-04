"""Integration test for persisted selected live forecast closeout."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pandas import DataFrame
import pytest
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    WeeklyProductIdentity,
)
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    write_forecast_events,
)
from gridiron_edge.evaluation.live_forecast_closeout import (
    load_live_forecast_closeout,
)
from gridiron_edge.models.game_prediction.weekly_game_product import (
    build_weekly_game_product,
)


def _event(
    *,
    event_id: str,
    model_name: str,
    model_type: str,
    home_win_prob: float | None,
    model_total: float | None,
) -> DataFrame:
    values: dict[str, object] = {
        "event_id": event_id,
        "run_id": "run-1",
        "role": ForecastRole.LIVE.value,
        "generated_at": datetime(2025, 9, 1, 12, tzinfo=UTC),
        "season": "2025-2026",
        "week": 1,
        "game_id": "2025_01_A_B",
        "model_name": model_name,
        "model_type": model_type,
        "game_date": "2025-09-07",
        "away_team": "Team B",
        "home_team": "Team A",
        "away_elo": None,
        "home_elo": None,
        "away_win_prob": (None if home_win_prob is None else 1.0 - home_win_prob),
        "home_win_prob": home_win_prob,
        "model_spread": None,
        "model_total": model_total,
        "projected_home_score": None,
        "projected_away_score": None,
        "margin_std": None,
        "win_prob_lo": None,
        "win_prob_hi": None,
        "confidence_tier": None,
    }
    return DataFrame([{column: values[column] for column in FORECAST_EVENT_COLUMNS}])


def _product() -> DataFrame:
    generated_at = datetime(2025, 9, 1, 12, tzinfo=UTC)
    base = DataFrame(
        {
            "season": ["2025-2026"],
            "week": [1],
            "game_id": ["2025_01_A_B"],
            "away_team": ["Team B"],
            "home_team": ["Team A"],
            "neutral_site": [False],
            "win_status": ["available"],
            "win_selection_status": ["selected"],
            "away_win_prob": [0.30],
            "home_win_prob": [0.70],
            "win_model_name": ["win_prob"],
            "win_model_type": ["logistic"],
            "win_event_id": ["win-1"],
            "win_run_id": ["run-1"],
            "win_generated_at": [generated_at],
            "win_role": ["live"],
            "spread_status": ["available"],
            "model_spread": [-3.0],
            "spread_uncertainty": [13.0],
            "spread_source_event_id": ["win-1"],
            "spread_model_name": ["win_prob"],
            "spread_model_type": ["logistic"],
            "spread_calibration_key": ["win_prob_logistic"],
            "spread_calibration_updated_at": ["2025-09-01T12:00:00+00:00"],
            "total_status": ["available"],
            "total_selection_status": ["selected"],
            "model_total": [45.0],
            "total_uncertainty": [12.0],
            "total_model_name": ["total"],
            "total_model_type": ["random_forest"],
            "total_event_id": ["total-1"],
            "total_run_id": ["run-1"],
            "total_generated_at": [generated_at],
            "total_role": ["live"],
            "total_uncertainty_trained_at": ["2025-08-31T12:00:00+00:00"],
        }
    )
    return build_weekly_game_product(base)


def test_persisted_selected_product_closes_against_completed_outcome(
    tmp_path: Path,
) -> None:
    MiniRepoBuilder(tmp_path).with_games()
    event_frames = [
        _event(
            event_id="win-1",
            model_name="win_prob",
            model_type="logistic",
            home_win_prob=0.70,
            model_total=None,
        ),
        _event(
            event_id="total-1",
            model_name="total",
            model_type="random_forest",
            home_win_prob=None,
            model_total=45.0,
        ),
    ]
    events = DataFrame.from_records(
        [frame.iloc[0].to_dict() for frame in event_frames],
        columns=FORECAST_EVENT_COLUMNS,
    )
    write_forecast_events(events, repo=tmp_path)

    generated_at = datetime(2025, 9, 1, 12, tzinfo=UTC)
    identity = WeeklyProductIdentity(
        product_id="closeout-product",
        run_id="run-1",
        season="2025-2026",
        week=1,
        generated_at=generated_at,
    )
    write_weekly_product(tmp_path, _product(), identity=identity)
    select_current_weekly_product(
        tmp_path,
        identity.product_id,
        season=identity.season,
        week=identity.week,
        selected_at=generated_at,
    )

    closeout = load_live_forecast_closeout(
        repo=tmp_path,
        season="2025-2026",
        week=1,
    )

    assert closeout.complete
    assert closeout.product_id == "closeout-product"
    assert closeout.product_run_id == "run-1"
    assert closeout.scheduled_game_count == 1
    assert closeout.completed_outcome_count == 1
    assert closeout.matched_win_event_count == 1
    assert closeout.matched_total_event_count == 1
    assert closeout.win.evaluated_count == 1
    assert closeout.win.brier == pytest.approx(0.09)
    assert closeout.win.accuracy == pytest.approx(1.0)
    assert closeout.total.evaluated_count == 1
    assert closeout.total.mae == pytest.approx(2.0)
    assert closeout.total.rmse == pytest.approx(2.0)
    assert closeout.total.bias == pytest.approx(-2.0)
    assert closeout.reconciliation["game_id"].tolist() == ["2025_01_A_B"]
    assert closeout.reconciliation["actual_total"].tolist() == [47.0]
