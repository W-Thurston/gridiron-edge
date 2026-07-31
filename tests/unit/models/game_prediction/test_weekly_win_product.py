# tests/unit/models/game_prediction/test_weekly_win_product.py

"""Tests for schedule-complete weekly win prediction products."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    SelectedForecast,
)
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
)
from gridiron_edge.models.game_prediction.weekly_win_product import (
    WeeklyWinStatus,
    build_weekly_win_product,
)


def _schedule() -> DataFrame:
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [1, 1, 2],
            "game_id": ["game-1", "game-2", "game-3"],
            "game_date": ["2026-09-06", "2026-09-06", "2026-09-10"],
            "game_time": ["13:00:00", "16:25:00", "20:15:00"],
            "away_team": ["Away One", "Away Two", "Away Three"],
            "home_team": ["Home One", "Home Two", "Home Three"],
            "neutral_site": [False, True, False],
        }
    )


def _policy(*, available: bool = True) -> PredictionPolicy:
    availability = PredictionAvailability(
        season="2026-2027",
        week=1,
        elo_available=True,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=False,
        total_xgboost_features_available=False,
    )
    if available:
        provenance = ModelProvenance(
            model_name="win_prob",
            model_type="elo",
            source=PredictionModelSource.POLICY,
        )
        win = PredictionModelDecision(
            model_name="win_prob",
            model_type="elo",
            status=PredictionModelStatus.SELECTED,
            rationale=PredictionPolicyRationale.ELO_ONLY_AVAILABLE,
            explanation="Elo is available.",
            provenance=provenance,
        )
    else:
        win = PredictionModelDecision(
            model_name="win_prob",
            model_type=None,
            status=PredictionModelStatus.UNAVAILABLE,
            rationale=PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE,
            explanation="Win prediction is unavailable.",
            provenance=None,
        )
    total = PredictionModelDecision(
        model_name="total",
        model_type=None,
        status=PredictionModelStatus.UNAVAILABLE,
        rationale=PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE,
        explanation="Total prediction is unavailable.",
        provenance=None,
    )
    return PredictionPolicy(availability=availability, win=win, total=total)


def _event(
    *,
    event_id: str,
    game_id: str,
    away_team: str,
    home_team: str,
    away_win_prob: float = 0.55,
    home_win_prob: float = 0.45,
    model_type: str = "elo",
) -> DataFrame:
    values: dict[str, object] = {
        "event_id": event_id,
        "run_id": "run-1",
        "role": ForecastRole.LIVE.value,
        "generated_at": datetime(2026, 9, 1, 12, tzinfo=UTC),
        "season": "2026-2027",
        "week": 1,
        "game_id": game_id,
        "model_name": "win_prob",
        "model_type": model_type,
        "game_date": "2026-09-06",
        "away_team": away_team,
        "home_team": home_team,
        "away_elo": 1520.0,
        "home_elo": 1480.0,
        "away_win_prob": away_win_prob,
        "home_win_prob": home_win_prob,
        "model_spread": None,
        "model_total": None,
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
) -> ForecastCandidateResolution:
    identity = ForecastCandidateIdentity(
        game_id=game_id,
        model_name="win_prob",
        model_type="elo",
    )
    if status is ForecastCandidateStatus.SELECTED:
        if event_id is None:
            raise ValueError("Selected fixture requires event_id.")
        selected = SelectedForecast(
            event_id=event_id,
            game_id=game_id,
            model_name="win_prob",
            model_type="elo",
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


def test_output_has_exactly_one_row_per_scheduled_game() -> None:
    events = pd.concat(
        [
            _event(
                event_id="event-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
            ),
            _event(
                event_id="event-2",
                game_id="game-2",
                away_team="Away Two",
                home_team="Home Two",
            ),
        ],
        ignore_index=True,
    )
    product = build_weekly_win_product(
        _schedule(),
        events,
        [
            _resolution(game_id="game-1", event_id="event-1"),
            _resolution(game_id="game-2", event_id="event-2"),
        ],
        policy=_policy(),
        season="2026-2027",
        week=1,
    )

    assert len(product) == 2
    assert product["game_id"].tolist() == ["game-1", "game-2"]
    assert set(product["win_status"]) == {WeeklyWinStatus.AVAILABLE.value}


def test_missing_win_forecast_does_not_remove_game() -> None:
    product = build_weekly_win_product(
        _schedule(),
        _event(
            event_id="event-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="event-1")],
        policy=_policy(),
        season="2026-2027",
        week=1,
    )

    missing = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert len(product) == 2
    assert missing["win_status"] == WeeklyWinStatus.FORECAST_MISSING.value
    assert pd.isna(missing["away_win_prob"])
    assert pd.isna(missing["win_event_id"])


def test_available_prediction_preserves_model_and_event_identity() -> None:
    product = build_weekly_win_product(
        _schedule(),
        _event(
            event_id="event-1",
            game_id="game-1",
            away_team="Away One",
            home_team="Home One",
        ),
        [_resolution(game_id="game-1", event_id="event-1")],
        policy=_policy(),
        season="2026-2027",
        week=1,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["win_model_name"] == "win_prob"
    assert row["win_model_type"] == "elo"
    assert row["win_event_id"] == "event-1"
    assert row["win_run_id"] == "run-1"
    assert row["win_role"] == ForecastRole.LIVE.value
    assert row["win_selection_status"] == ForecastCandidateStatus.SELECTED.value


def test_ambiguous_selection_is_explicit() -> None:
    product = build_weekly_win_product(
        _schedule(),
        DataFrame(columns=FORECAST_EVENT_COLUMNS),
        [
            _resolution(
                game_id="game-1",
                status=ForecastCandidateStatus.AMBIGUOUS,
            )
        ],
        policy=_policy(),
        season="2026-2027",
        week=1,
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["win_status"] == WeeklyWinStatus.FORECAST_AMBIGUOUS.value
    assert row["win_selection_status"] == ForecastCandidateStatus.AMBIGUOUS.value


def test_policy_unavailable_marks_every_schedule_row() -> None:
    product = build_weekly_win_product(
        _schedule(),
        DataFrame(columns=FORECAST_EVENT_COLUMNS),
        [],
        policy=_policy(available=False),
        season="2026-2027",
        week=1,
    )

    assert len(product) == 2
    assert set(product["win_status"]) == {WeeklyWinStatus.POLICY_UNAVAILABLE.value}
    assert product["away_win_prob"].isna().all()


def test_probability_complements_are_validated() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        build_weekly_win_product(
            _schedule(),
            _event(
                event_id="event-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
                away_win_prob=0.70,
                home_win_prob=0.40,
            ),
            [_resolution(game_id="game-1", event_id="event-1")],
            policy=_policy(),
            season="2026-2027",
            week=1,
        )


def test_team_orientation_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="away_team does not match"):
        build_weekly_win_product(
            _schedule(),
            _event(
                event_id="event-1",
                game_id="game-1",
                away_team="Home One",
                home_team="Away One",
            ),
            [_resolution(game_id="game-1", event_id="event-1")],
            policy=_policy(),
            season="2026-2027",
            week=1,
        )


def test_policy_and_event_model_identity_must_match() -> None:
    with pytest.raises(ValueError, match="model_type does not match policy"):
        build_weekly_win_product(
            _schedule(),
            _event(
                event_id="event-1",
                game_id="game-1",
                away_team="Away One",
                home_team="Home One",
                model_type="random_forest",
            ),
            [_resolution(game_id="game-1", event_id="event-1")],
            policy=_policy(),
            season="2026-2027",
            week=1,
        )


def test_duplicate_schedule_game_ids_are_rejected() -> None:
    schedule = pd.concat(
        [_schedule(), _schedule().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate game IDs"):
        build_weekly_win_product(
            schedule,
            DataFrame(columns=FORECAST_EVENT_COLUMNS),
            [],
            policy=_policy(),
            season="2026-2027",
            week=1,
        )


def test_neutral_site_identity_is_preserved() -> None:
    product = build_weekly_win_product(
        _schedule(),
        DataFrame(columns=FORECAST_EVENT_COLUMNS),
        [],
        policy=_policy(),
        season="2026-2027",
        week=1,
    )

    neutral = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert bool(neutral["neutral_site"])
