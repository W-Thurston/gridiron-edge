# tests/unit/api/test_serializers_games.py

"""Tests for /games serializers (W8 Tier 2 Step 5c)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridiron_edge.api.meta import BlockedStatus, FieldStatus
from gridiron_edge.api.schemas.games import GameDetail, GameList, GameSummary, PredictionBlock
from gridiron_edge.api.serializers.games import (
    _build_prediction_block,
    _derive_day_of_week,
    serialize_game_detail,
    serialize_game_summary,
    serialize_games_list,
)


def _valid_row() -> dict:
    """A canonical valid loader row for reuse across tests."""
    return {
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-05",
        "week": 1,
        "season": "2026-2027",
        "away_team": "KC",
        "home_team": "LAC",
        "home_win_prob": 0.55,
        "away_win_prob": 0.45,
        "model_spread": -2.5,
        "model_total": 47.5,
        "projected_home_score": 25.0,
        "projected_away_score": 22.5,
        "confidence_tier": "Moderate",
        "win_prob_lo": 0.42,
        "win_prob_hi": 0.68,
    }


class TestDeriveDayOfWeek:
    def test_iso_date_returns_weekday_name(self) -> None:
        assert _derive_day_of_week("2026-09-05") == "Saturday"

    def test_none_returns_none(self) -> None:
        assert _derive_day_of_week(None) is None

    def test_nan_returns_none(self) -> None:
        assert _derive_day_of_week(np.nan) is None

    def test_empty_string_returns_none(self) -> None:
        assert _derive_day_of_week("") is None

    def test_unparseable_returns_none(self) -> None:
        assert _derive_day_of_week("not a date") is None


class TestBuildPredictionBlock:
    def test_full_row_builds_block(self) -> None:
        block: PredictionBlock | None = _build_prediction_block(_valid_row())
        assert block is not None
        assert block.home_win_prob == 0.55
        assert block.confidence_tier == "Moderate"
        assert block.home_win_lo == 0.42

    def test_all_none_row_returns_none(self) -> None:
        row: dict[str, None] = {
            "home_win_prob": None,
            "away_win_prob": None,
            "win_prob_lo": None,
            "win_prob_hi": None,
            "confidence_tier": None,
            "model_spread": None,
            "model_total": None,
            "projected_home_score": None,
            "projected_away_score": None,
        }
        assert _build_prediction_block(row) is None

    def test_nan_fields_become_none(self) -> None:
        row: dict = _valid_row()
        row["model_total"] = np.nan
        row["projected_home_score"] = np.nan
        block: PredictionBlock | None = _build_prediction_block(row)
        assert block is not None
        assert block.model_total is None
        assert block.projected_home_score is None
        # Non-NaN fields still populate.
        assert block.home_win_prob == 0.55


class TestSerializeGameSummary:
    def test_full_row(self) -> None:
        summary: GameSummary = serialize_game_summary(_valid_row())
        assert isinstance(summary, GameSummary)
        assert summary.game_id == "2026_01_KC_LAC"
        assert summary.away_team == "KC"
        assert summary.home_team == "LAC"
        assert summary.prediction is not None
        assert summary.prediction.home_win_prob == 0.55

    def test_nan_scalar_fields_become_none(self) -> None:
        row: dict = _valid_row()
        row["week"] = np.nan
        row["game_date"] = np.nan
        summary: GameSummary = serialize_game_summary(row)
        assert summary.week is None
        assert summary.game_date is None


class TestSerializeGamesList:
    def test_empty_dataframe_returns_empty_list(self) -> None:
        response: GameList = serialize_games_list(
            pd.DataFrame(),
            season="2026-2027",
            week=1,
        )
        assert isinstance(response, GameList)
        assert response.items == []
        assert response.total == 0
        assert response.season == "2026-2027"
        assert response.week == 1

    def test_dataframe_of_rows_serializes_each(self) -> None:
        df = pd.DataFrame([_valid_row(), _valid_row() | {"game_id": "2026_01_BUF_MIA"}])
        response: GameList = serialize_games_list(df, season="2026-2027", week=1)
        assert len(response.items) == 2
        assert response.total == 2
        assert response.items[0].game_id == "2026_01_KC_LAC"
        assert response.items[1].game_id == "2026_01_BUF_MIA"

    def test_none_season_and_week_pass_through(self) -> None:
        response: GameList = serialize_games_list(pd.DataFrame(), season=None, week=None)
        assert response.season is None
        assert response.week is None


class TestSerializeGameDetail:
    def test_full_row(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert isinstance(detail, GameDetail)
        assert detail.game_id == "2026_01_KC_LAC"
        assert detail.game_date == "2026-09-05"
        assert detail.day_of_week == "Saturday"
        assert detail.away_team == "KC"
        assert detail.home_team == "LAC"

    def test_prediction_populated_from_row(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.prediction is not None
        assert detail.prediction.home_win_prob == 0.55
        assert detail.prediction.confidence_tier == "Moderate"

    def test_kick_venue_weather_ship_null(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.kick is None
        assert detail.venue is None
        assert detail.weather is None

    def test_field_status_marks_pending_fields(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.response_meta is not None
        status: dict[str, FieldStatus] = detail.response_meta.field_status
        assert status["kick"] == "pending"
        assert status["venue"] == "pending"
        assert status["weather"] == "pending"
        assert status["team_comparison"] == "pending"
        assert status["top_prop_edges"] == "pending"

    def test_field_status_marks_swing_factors_blocked(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.response_meta is not None
        status: FieldStatus = detail.response_meta.field_status["swing_factors"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "feature_attribution"

    def test_field_status_marks_injuries_blocked(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.response_meta is not None
        status: FieldStatus = detail.response_meta.field_status["injuries"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "injury_data_source"

    def test_scaffolded_fields_are_null(self) -> None:
        detail: GameDetail = serialize_game_detail(_valid_row())
        assert detail.team_comparison is None
        assert detail.swing_factors is None
        assert detail.injuries is None
        assert detail.top_prop_edges is None

    def test_day_of_week_derived_from_game_date(self) -> None:
        row: dict = _valid_row()
        row["game_date"] = "2026-09-14"  # Monday
        detail: GameDetail = serialize_game_detail(row)
        assert detail.day_of_week == "Monday"

    def test_missing_game_date_yields_none_day_of_week(self) -> None:
        row: dict = _valid_row()
        row["game_date"] = None
        detail: GameDetail = serialize_game_detail(row)
        assert detail.day_of_week is None


class TestGameDetailTeamComparison:
    def test_populates_team_comparison(self) -> None:
        team_comparison = {
            "KC": {"season": {"off_epa_per_play": 0.15, "sample_size": 4}},
            "LAC": {"season": {"off_epa_per_play": 0.10, "sample_size": 4}},
        }

        detail: GameDetail = serialize_game_detail(
            _valid_row(),
            team_comparison=team_comparison,
        )

        assert detail.team_comparison == team_comparison
        # Marker removed when populated
        assert "team_comparison" not in detail.response_meta.field_status

    def test_none_leaves_pending_marker(self) -> None:
        detail: GameDetail = serialize_game_detail(
            _valid_row(),
            team_comparison=None,
        )
        assert detail.team_comparison is None
        assert detail.response_meta.field_status["team_comparison"] == "pending"
