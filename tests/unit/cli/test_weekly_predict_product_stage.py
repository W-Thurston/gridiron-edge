# tests/unit/cli/test_weekly_predict_product_stage.py

"""Tests for weekly-predict persisted product composition."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pandas import DataFrame

from gridiron_edge.cli.weekly_predict import _stage_compose_weekly_product
from gridiron_edge.models.game_prediction.prediction_policy import (
    PredictionAvailability,
)

SEASON = "2026-2027"
WEEK = 1
RUN_ID = "run-1"
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _context() -> dict[str, object]:
    return {
        "season": SEASON,
        "week": WEEK,
        "forecast_run_id": RUN_ID,
        "forecast_generated_at": GENERATED_AT,
    }


def test_missing_run_identity_fails_before_loading_artifacts() -> None:
    result = _stage_compose_weekly_product(
        {
            "season": SEASON,
            "week": WEEK,
            "forecast_generated_at": GENERATED_AT,
        }
    )

    assert not result.success
    assert result.detail == "forecast run identity is unavailable"


def test_missing_generation_time_fails_before_loading_artifacts() -> None:
    result = _stage_compose_weekly_product(
        {
            "season": SEASON,
            "week": WEEK,
            "forecast_run_id": RUN_ID,
        }
    )

    assert not result.success
    assert result.detail == "forecast generation time is unavailable"


def test_composes_writes_and_selects_exact_forecast_run(tmp_path: Path) -> None:
    schedule = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": ["2026_01_KC_LAC"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
        }
    )
    events = DataFrame({"event_id": ["event-1"]})
    win_product = schedule.assign(win_status="available")
    spread_product = win_product.assign(spread_status="available")
    total_product = spread_product.assign(total_status="policy_unavailable")
    final_product = total_product.assign(projected_score_status="total_unavailable")
    artifact = tmp_path / "weekly.parquet"
    selected_run = SimpleNamespace(found=True, events=events)
    resolutions = (MagicMock(),)
    inspected_availability = PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=False,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=False,
    )

    with (
        patch(
            "gridiron_edge.cli.weekly_predict.get_settings",
            return_value=SimpleNamespace(repo_root=tmp_path),
        ),
        patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming_rich",
            return_value=schedule,
        ),
        patch(
            "gridiron_edge.models.game_prediction.availability.inspect_prediction_availability",
            return_value=inspected_availability,
        ) as inspect_availability,
        patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=events,
        ) as load_events,
        patch(
            "gridiron_edge.evaluation.forecast_selection.select_forecast_run",
            return_value=selected_run,
        ) as select_run,
        patch(
            "gridiron_edge.evaluation.forecast_selection.resolve_forecast_candidates",
            return_value=resolutions,
        ) as resolve_candidates,
        patch(
            "gridiron_edge.models.game_prediction.weekly_win_product.build_weekly_win_product",
            return_value=win_product,
        ) as build_win,
        patch(
            "gridiron_edge.models.game_prediction.weekly_spread_product.load_and_attach_derived_spreads",
            return_value=spread_product,
        ) as attach_spread,
        patch(
            "gridiron_edge.models.game_prediction.weekly_total_product.load_and_attach_selected_totals",
            return_value=total_product,
        ) as attach_total,
        patch(
            "gridiron_edge.models.game_prediction.weekly_game_product.build_weekly_game_product",
            return_value=final_product,
        ) as build_product,
        patch(
            "gridiron_edge.datasets.writers.write_weekly_product",
            return_value=artifact,
        ) as write_product,
        patch("gridiron_edge.datasets.writers.select_current_weekly_product") as select_product,
    ):
        context = _context()
        result = _stage_compose_weekly_product(context)

    assert result.success
    assert result.rows == 1
    assert result.artifacts == [artifact]
    inspect_availability.assert_called_once_with(
        schedule,
        season=SEASON,
        week=WEEK,
        repo=tmp_path,
    )
    load_events.assert_called_once_with(
        season=SEASON,
        week=WEEK,
        run_id=RUN_ID,
        repo=tmp_path,
    )
    select_run.assert_called_once_with(events, run_id=RUN_ID)
    identity = resolve_candidates.call_args.args[1][0]
    assert identity.game_id == "2026_01_KC_LAC"
    assert identity.model_name == "win_prob"
    assert identity.model_type == "elo"
    build_win.assert_called_once()
    attach_spread.assert_called_once_with(win_product, repo=tmp_path)
    assert attach_total.call_args.args[2] == ()
    policy = attach_total.call_args.kwargs["policy"]
    assert policy.availability is inspected_availability
    assert policy.availability.win_logistic_features_available
    assert policy.availability.win_xgboost_features_available
    assert policy.availability.total_random_forest_features_available
    assert policy.win.model_type == "elo"
    assert policy.total.model_type is None
    build_product.assert_called_once_with(total_product)

    written_identity = write_product.call_args.kwargs["identity"]
    assert written_identity.run_id == RUN_ID
    assert written_identity.generated_at == GENERATED_AT
    assert written_identity.season == SEASON
    assert written_identity.week == WEEK
    assert context["weekly_product_id"] == written_identity.product_id
    assert context["weekly_product_path"] == artifact
    select_product.assert_called_once()
    assert select_product.call_args.args[:2] == (
        tmp_path,
        written_identity.product_id,
    )


def test_missing_persisted_run_fails_without_writing(tmp_path: Path) -> None:
    schedule = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": ["2026_01_KC_LAC"],
        }
    )
    with (
        patch(
            "gridiron_edge.cli.weekly_predict.get_settings",
            return_value=SimpleNamespace(repo_root=tmp_path),
        ),
        patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming_rich",
            return_value=schedule,
        ),
        patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=DataFrame(),
        ),
        patch(
            "gridiron_edge.evaluation.forecast_selection.select_forecast_run",
            return_value=SimpleNamespace(found=False, events=DataFrame()),
        ),
        patch("gridiron_edge.datasets.writers.write_weekly_product") as write_product,
    ):
        result = _stage_compose_weekly_product(_context())

    assert not result.success
    assert result.detail == "forecast run is not persisted"
    write_product.assert_not_called()
