# tests/unit/models/game_prediction/test_weekly_total_product.py

"""Tests for independent total attachment to weekly products."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole, SelectedForecast
from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateIdentity,
    ForecastCandidateResolution,
    ForecastCandidateStatus,
)
from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelDecision,
    PredictionModelSource,
    PredictionModelStatus,
    PredictionPolicy,
    PredictionPolicyRationale,
    resolve_prediction_policy,
)
from gridiron_edge.models.game_prediction.weekly_total_product import (
    TotalUncertainty,
    WeeklyTotalStatus,
    attach_selected_totals,
)


def _weekly_product() -> DataFrame:
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "week": [8, 8],
            "game_id": ["game-1", "game-2"],
            "away_team": ["Away One", "Away Two"],
            "home_team": ["Home One", "Home Two"],
            "neutral_site": [False, True],
            "win_model_type": ["elo", "elo"],
            "spread_model_type": ["elo", "elo"],
            "model_spread": [-3.0, 2.5],
        }
    )


def _decision(
    model_name: str,
    model_type: str | None,
) -> PredictionModelDecision:
    if model_type is None:
        return PredictionModelDecision(
            model_name=model_name,
            model_type=None,
            status=PredictionModelStatus.UNAVAILABLE,
            rationale=PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE,
            explanation="Unavailable.",
            provenance=None,
        )
    provenance = ModelProvenance(
        model_name=model_name,
        model_type=model_type,
        source=PredictionModelSource.CHAMPION,
    )
    return PredictionModelDecision(
        model_name=model_name,
        model_type=model_type,
        status=PredictionModelStatus.SELECTED,
        rationale=PredictionPolicyRationale.CHAMPION_ELIGIBLE,
        explanation="Selected.",
        provenance=provenance,
    )


def _policy(*, total_model_type: str | None = "xgboost") -> PredictionPolicy:
    """Create a resolved policy for weekly Total composition tests."""
    availability = PredictionAvailability(
        season="2026-2027",
        week=8,
        elo_available=True,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=(total_model_type == "random_forest"),
        total_xgboost_features_available=(total_model_type == "xgboost"),
    )

    return resolve_prediction_policy(
        availability,
        win_champion=None,
        total_champion=None,
        win_override="elo",
        total_override=total_model_type,
    )


def _event(
    *,
    event_id: str,
    game_id: str,
    away_team: str,
    home_team: str,
    model_type: str = "xgboost",
    model_total: float = 44.5,
) -> DataFrame:
    values: dict[str, object] = {
        "event_id": event_id,
        "run_id": "total-run",
        "role": ForecastRole.LIVE.value,
        "generated_at": datetime(2026, 10, 20, 12, tzinfo=UTC),
        "season": "2026-2027",
        "week": 8,
        "game_id": game_id,
        "model_name": "total",
        "model_type": model_type,
        "game_date": "2026-10-25",
        "away_team": away_team,
        "home_team": home_team,
        "away_elo": None,
        "home_elo": None,
        "away_win_prob": None,
        "home_win_prob": None,
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


def _resolution(
    *,
    game_id: str,
    event_id: str | None = None,
    status: ForecastCandidateStatus = ForecastCandidateStatus.SELECTED,
    model_type: str = "xgboost",
) -> ForecastCandidateResolution:
    identity = ForecastCandidateIdentity(
        game_id=game_id,
        model_name="total",
        model_type=model_type,
    )
    if status is ForecastCandidateStatus.SELECTED:
        if event_id is None:
            raise ValueError("Selected fixture requires event_id.")
        selected = SelectedForecast(
            event_id=event_id,
            game_id=game_id,
            model_name="total",
            model_type=model_type,
        )
        eligible = (event_id,)
    elif status is ForecastCandidateStatus.AMBIGUOUS:
        selected = None
        eligible = ("event-a", "event-b")
    else:
        selected = None
        eligible = ()
    return ForecastCandidateResolution(
        identity=identity,
        status=status,
        selected=selected,
        eligible_event_ids=eligible,
    )


def _uncertainty(model_type: str = "xgboost") -> TotalUncertainty:
    return TotalUncertainty(
        model_name="total",
        model_type=model_type,
        total_std=12.8,
        trained_at="2026-07-01T14:20:00",
    )


def test_total_model_is_independent_from_win_model() -> None:
    product = attach_selected_totals(
        _weekly_product(),
        _event(
            event_id="total-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="total-1")],
        {("total", "xgboost"): _uncertainty()},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["win_model_type"] == "elo"
    assert row["total_model_type"] == "xgboost"
    assert row["total_status"] == WeeklyTotalStatus.AVAILABLE.value


def test_total_uncertainty_uses_total_model_identity() -> None:
    product = attach_selected_totals(
        _weekly_product(),
        _event(
            event_id="total-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="total-1")],
        {("total", "xgboost"): _uncertainty()},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["total_uncertainty"] == pytest.approx(12.8)
    assert row["total_uncertainty_trained_at"] == "2026-07-01T14:20:00"


def test_missing_total_prediction_does_not_remove_game() -> None:
    product = attach_selected_totals(
        _weekly_product(),
        _event(
            event_id="total-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="total-1")],
        {("total", "xgboost"): _uncertainty()},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    missing = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert len(product) == 2
    assert missing["total_status"] == WeeklyTotalStatus.FORECAST_MISSING.value
    assert pd.isna(missing["model_total"])


def test_total_provenance_is_preserved() -> None:
    product = attach_selected_totals(
        _weekly_product(),
        _event(
            event_id="total-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="total-1")],
        {("total", "xgboost"): _uncertainty()},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["model_total"] == pytest.approx(44.5)
    assert row["total_model_name"] == "total"
    assert row["total_model_type"] == "xgboost"
    assert row["total_event_id"] == "total-1"
    assert row["total_run_id"] == "total-run"
    assert row["total_role"] == ForecastRole.LIVE.value


def test_independent_total_events_compose_by_game_id() -> None:
    events = pd.concat(
        [
            _event(
                event_id="total-2",
                game_id="game-2",
                away_team="Away Two",
                home_team="Home Two",
                model_total=48.0,
            ),
            _event(
                event_id="total-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
                model_total=41.0,
            ),
        ],
        ignore_index=True,
    )
    product = attach_selected_totals(
        _weekly_product(),
        events,
        [
            _resolution(game_id="game-1", event_id="total-1"),
            _resolution(game_id="game-2", event_id="total-2"),
        ],
        {("total", "xgboost"): _uncertainty()},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    assert product["game_id"].tolist() == ["game-1", "game-2"]
    assert product["model_total"].tolist() == [41.0, 48.0]


def test_missing_uncertainty_preserves_total_point_estimate() -> None:
    product = attach_selected_totals(
        _weekly_product(),
        _event(
            event_id="total-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="total-1")],
        {},
        policy=_policy(),
        season="2026-2027",
        week=8,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["total_status"] == WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value
    assert row["model_total"] == pytest.approx(44.5)
    assert pd.isna(row["total_uncertainty"])


def test_total_policy_unavailable_preserves_rows_and_existing_fields() -> None:
    source = _weekly_product()
    product = attach_selected_totals(
        source,
        DataFrame(columns=FORECAST_EVENT_COLUMNS),
        [],
        {},
        policy=_policy(total_model_type=None),
        season="2026-2027",
        week=8,
    )

    assert len(product) == len(source)
    assert product["game_id"].tolist() == source["game_id"].tolist()
    assert product["model_spread"].tolist() == source["model_spread"].tolist()
    assert set(product["total_status"]) == {WeeklyTotalStatus.POLICY_UNAVAILABLE.value}


def test_total_event_orientation_is_validated() -> None:
    with pytest.raises(ValueError, match="away_team does not match"):
        attach_selected_totals(
            _weekly_product(),
            _event(
                event_id="total-1",
                game_id="game-1",
                away_team="Home One",
                home_team="Away One",
            ),
            [_resolution(game_id="game-1", event_id="total-1")],
            {("total", "xgboost"): _uncertainty()},
            policy=_policy(),
            season="2026-2027",
            week=8,
        )


def test_total_event_must_match_total_policy_identity() -> None:
    with pytest.raises(ValueError, match="model_type does not match policy"):
        attach_selected_totals(
            _weekly_product(),
            _event(
                event_id="total-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
                model_type="random_forest",
            ),
            [
                _resolution(
                    game_id="game-1",
                    event_id="total-1",
                    model_type="random_forest",
                )
            ],
            {("total", "xgboost"): _uncertainty()},
            policy=_policy(),
            season="2026-2027",
            week=8,
        )


def test_total_uncertainty_identity_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_type does not match forecast"):
        attach_selected_totals(
            _weekly_product(),
            _event(
                event_id="total-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
            ),
            [_resolution(game_id="game-1", event_id="total-1")],
            {("total", "xgboost"): _uncertainty("random_forest")},
            policy=_policy(),
            season="2026-2027",
            week=8,
        )
