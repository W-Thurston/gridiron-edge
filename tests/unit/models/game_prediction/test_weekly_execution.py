# tests/unit/models/game_prediction/test_weekly_execution.py
"""Tests for policy-selected weekly game-model execution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    resolve_prediction_policy,
)
from gridiron_edge.models.game_prediction.weekly_execution import (
    execute_weekly_prediction_policy,
)

SEASON = "2026-2027"
WEEK = 1
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [WEEK, WEEK],
            "game_id": ["g1", "g2"],
            "game_day_of_week": ["Sun", "Sun"],
            "game_date": ["2026-09-13", "2026-09-13"],
            "game_time": ["13:00", "16:25"],
            "away_team": ["Away A", "Away B"],
            "home_team": ["Home A", "Home B"],
            "neutral_site": [0, 0],
        }
    )


def _availability() -> PredictionAvailability:
    return PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=True,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=True,
    )


def _policy():
    return resolve_prediction_policy(
        _availability(),
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


def test_executes_exact_selected_families_under_one_run(tmp_path: Path) -> None:
    win = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "AWAY_TEAM": ["Away A", "Away B"],
            "HOME_TEAM": ["Home A", "Home B"],
            "WEEK_NUM": [1, 1],
            "AWAY_TEAM_ELO": [1500.0, 1490.0],
            "HOME_TEAM_ELO": [1510.0, 1520.0],
            "AWAY_WIN_PROB": [0.45, 0.40],
            "HOME_WIN_PROB": [0.55, 0.60],
        }
    )
    total = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "AWAY_TEAM": ["Away A", "Away B"],
            "HOME_TEAM": ["Home A", "Home B"],
            "WEEK_NUM": [1, 1],
            "model_total": [44.5, 47.0],
        }
    )
    win_model = MagicMock()
    win_model.predict_upcoming.return_value = win
    total_model = MagicMock()
    total_model.predict_upcoming.return_value = total

    def registry_get(key: str):
        return {
            "win_prob_logistic": lambda: win_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
    ):
        execution = execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="run-1",
            generated_at=GENERATED_AT,
        )

    assert len(execution.events) == 4
    assert set(execution.events["model_name"]) == {"win_prob", "total"}
    assert set(execution.events["model_type"]) == {"logistic", "random_forest"}
    assert set(execution.events["run_id"]) == {"run-1"}
    assert execution.win_display is not None
    assert execution.win_display["GAME_ID"].tolist() == ["g1", "g2"]
    win_model.predict_upcoming.assert_called_once()
    total_model.predict_upcoming.assert_called_once()


def test_rejects_partial_selected_family_before_return(tmp_path: Path) -> None:
    win_model = MagicMock()
    win_model.predict_upcoming.return_value = pd.DataFrame(
        {
            "GAME_ID": ["g1"],
            "AWAY_WIN_PROB": [0.45],
            "HOME_WIN_PROB": [0.55],
        }
    )
    total_model = MagicMock()
    total_model.predict_upcoming.return_value = pd.DataFrame(
        {"GAME_ID": ["g1", "g2"], "model_total": [44.5, 47.0]}
    )

    def registry_get(key: str):
        return {
            "win_prob_logistic": lambda: win_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
        pytest.raises(ValueError, match="Win prediction coverage"),
    ):
        execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="run-1",
            generated_at=GENERATED_AT,
        )
