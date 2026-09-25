# tests/unit/cli/test_weekly_product_composition.py

"""Tests for the shared, schedule-agnostic weekly-product composition helper.

``test_weekly_predict_product_stage.py`` already proves the live pipeline's
exact call chain and output through this helper. These tests cover what is
new here: composition driven by an arbitrary caller-supplied schedule (not
just the fetch-derived upcoming one), and the empty-scope failure this
introduces for a caller (development regeneration) that can request a scope
the schedule does not cover.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from pandas import DataFrame
import pytest

from gridiron_edge.cli._weekly_product_composition import (
    compose_and_select_weekly_product,
)
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    resolve_prediction_policy,
)

SEASON = "2026-2027"
WEEK = 2
RUN_ID = "run-1"
GENERATED_AT = datetime(2026, 9, 20, 12, tzinfo=UTC)


def _policy():
    availability = PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=True,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=True,
    )
    return resolve_prediction_policy(
        availability,
        win_champion=ModelProvenance(
            model_name="win_prob",
            model_type="logistic",
            source=PredictionModelSource.CHAMPION,
        ),
        total_champion=ModelProvenance(
            model_name="total",
            model_type="random_forest",
            source=PredictionModelSource.CHAMPION,
        ),
    )


def test_empty_scoped_schedule_raises_before_any_composition() -> None:
    schedule = DataFrame({"season": [SEASON], "week": [WEEK + 1], "game_id": ["other-week"]})

    with pytest.raises(ValueError, match="Rich schedule has no rows"):
        compose_and_select_weekly_product(
            schedule=schedule,
            events=DataFrame({"event_id": ["event-1"]}),
            policy=_policy(),
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
        )


def test_retained_history_schedule_composes_and_selects(tmp_path: Path) -> None:
    schedule = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": ["2026_02_DET_BUF"],
            "away_team": ["Detroit Lions"],
            "home_team": ["Buffalo Bills"],
        }
    )
    events = DataFrame({"event_id": ["event-1"]})
    win_product = schedule.assign(win_status="available")
    spread_product = win_product.assign(spread_status="available")
    total_product = spread_product.assign(total_status="available")
    final_product = total_product.assign(projected_score_status="available")
    artifact = tmp_path / "weekly.parquet"
    resolutions = (MagicMock(),)

    with (
        patch(
            "gridiron_edge.cli._weekly_product_composition.resolve_forecast_candidates",
            return_value=resolutions,
        ) as resolve_candidates,
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_win_product",
            return_value=win_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_derived_spreads",
            return_value=spread_product,
        ) as attach_spread,
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_selected_totals",
            return_value=total_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_game_product",
            return_value=final_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.write_weekly_product",
            return_value=artifact,
        ) as write_product,
        patch(
            "gridiron_edge.cli._weekly_product_composition.select_current_weekly_product"
        ) as select_product,
    ):
        composed = compose_and_select_weekly_product(
            schedule=schedule,
            events=events,
            policy=_policy(),
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    assert composed.artifact == artifact
    assert composed.row_count == 1
    assert composed.product_id == f"weekly_{SEASON.replace('-', '_')}_wk{WEEK:02d}_{RUN_ID}"
    assert resolve_candidates.call_count == 2
    attach_spread.assert_called_once_with(win_product, repo=tmp_path)
    write_product.assert_called_once()
    select_product.assert_called_once_with(
        tmp_path,
        composed.product_id,
        season=SEASON,
        week=WEEK,
        selected_at=select_product.call_args.kwargs["selected_at"],
    )


def test_unavailable_family_skips_candidate_resolution(tmp_path: Path) -> None:
    """A policy family with no model_type must not attempt resolution."""
    availability = PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=False,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=False,
        total_xgboost_features_available=False,
    )
    policy = resolve_prediction_policy(availability, win_champion=None, total_champion=None)
    assert policy.win.model_type is None
    assert policy.total.model_type is None

    schedule = DataFrame({"season": [SEASON], "week": [WEEK], "game_id": ["2026_02_DET_BUF"]})
    win_product = schedule.assign(win_status="policy_unavailable")
    spread_product = win_product.assign(spread_status="unavailable")
    total_product = spread_product.assign(total_status="policy_unavailable")
    final_product = total_product.assign(projected_score_status="total_unavailable")

    with (
        patch(
            "gridiron_edge.cli._weekly_product_composition.resolve_forecast_candidates",
        ) as resolve_candidates,
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_win_product",
            return_value=win_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_derived_spreads",
            return_value=spread_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_selected_totals",
            return_value=total_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_game_product",
            return_value=final_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.write_weekly_product",
            return_value=tmp_path / "weekly.parquet",
        ),
        patch("gridiron_edge.cli._weekly_product_composition.select_current_weekly_product"),
    ):
        compose_and_select_weekly_product(
            schedule=schedule,
            events=DataFrame({"event_id": ["event-1"]}),
            policy=policy,
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    resolve_candidates.assert_not_called()
