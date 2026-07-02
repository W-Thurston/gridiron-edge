"""Tests for /games response schemas (W8 Tier 2 Step 5b)."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.games import (
    GameDetail,
    GameList,
    GameSummary,
    PredictionBlock,
    WeatherBlock,
)


class TestPredictionBlock:
    def test_accepts_full_prediction(self) -> None:
        block = PredictionBlock(
            home_win_prob=0.55,
            away_win_prob=0.45,
            home_win_lo=0.42,
            home_win_hi=0.68,
            confidence_tier="Moderate",
            model_spread=-2.5,
            model_total=47.5,
            projected_home_score=25.0,
            projected_away_score=22.5,
        )
        assert block.home_win_prob == 0.55
        assert block.confidence_tier == "Moderate"

    def test_all_fields_default_to_none(self) -> None:
        block = PredictionBlock()
        assert block.home_win_prob is None
        assert block.confidence_tier is None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            PredictionBlock(home_win_prob=0.5, mystery_field=1)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        block = PredictionBlock(home_win_prob=0.5)
        with pytest.raises(ValidationError):
            block.home_win_prob = 0.6  # type: ignore[misc]


class TestWeatherBlock:
    def test_accepts_full_weather(self) -> None:
        weather = WeatherBlock(
            temp_f=72.0,
            wind_mph=8.0,
            conditions="Clear",
            precip_pct=0.05,
        )
        assert weather.temp_f == 72.0
        assert weather.conditions == "Clear"

    def test_all_fields_default_to_none(self) -> None:
        weather = WeatherBlock()
        assert weather.temp_f is None

    def test_frozen(self) -> None:
        weather = WeatherBlock(temp_f=72.0)
        with pytest.raises(ValidationError):
            weather.temp_f = 60.0  # type: ignore[misc]


class TestGameSummary:
    def test_minimum_shape(self) -> None:
        summary = GameSummary(
            game_id="2026_01_KC_LAC",
            away_team="KC",
            home_team="LAC",
        )
        assert summary.game_id == "2026_01_KC_LAC"
        assert summary.prediction is None

    def test_with_prediction(self) -> None:
        summary = GameSummary(
            game_id="2026_01_KC_LAC",
            game_date="2026-09-05",
            week=1,
            season="2026-2027",
            away_team="KC",
            home_team="LAC",
            prediction=PredictionBlock(
                home_win_prob=0.55,
                away_win_prob=0.45,
                confidence_tier="Moderate",
            ),
        )
        assert summary.prediction is not None
        assert summary.prediction.home_win_prob == 0.55

    def test_rejects_missing_required_teams(self) -> None:
        with pytest.raises(ValidationError):
            GameSummary(game_id="x")  # type: ignore[call-arg]


class TestGameList:
    def test_empty_list(self) -> None:
        response = GameList(items=[], _meta={"field_status": {}})  # type: ignore[call-arg]
        assert response.items == []

    def test_with_summaries(self) -> None:
        response = GameList(
            items=[
                GameSummary(
                    game_id="2026_01_KC_LAC",
                    away_team="KC",
                    home_team="LAC",
                ),
                GameSummary(
                    game_id="2026_01_BUF_MIA",
                    away_team="BUF",
                    home_team="MIA",
                ),
            ],
            season="2026-2027",
            week=1,
            _meta={"field_status": {}},  # type: ignore[call-arg]
        )
        assert len(response.items) == 2
        assert response.season == "2026-2027"


class TestGameDetail:
    def test_minimum_shape(self) -> None:
        detail = GameDetail(
            game_id="2026_01_KC_LAC",
            away_team="KC",
            home_team="LAC",
            _meta={"field_status": {}},  # type: ignore[call-arg]
        )
        assert detail.game_id == "2026_01_KC_LAC"
        assert detail.weather is None
        assert detail.prediction is None
        assert detail.team_comparison is None
        assert detail.swing_factors is None
        assert detail.injuries is None
        assert detail.top_prop_edges is None

    def test_with_all_populated_blocks(self) -> None:
        detail = GameDetail(
            game_id="2026_01_KC_LAC",
            game_date="2026-09-05",
            week=1,
            season="2026-2027",
            day_of_week="Thursday",
            kick="8:20 PM ET",
            venue="Arrowhead Stadium",
            away_team="KC",
            home_team="LAC",
            weather=WeatherBlock(temp_f=72.0, wind_mph=5.0),
            prediction=PredictionBlock(
                home_win_prob=0.55,
                confidence_tier="Moderate",
            ),
            _meta={"field_status": {}},  # type: ignore[call-arg]
        )
        assert detail.weather is not None
        assert detail.weather.temp_f == 72.0
        assert detail.prediction is not None
        assert detail.day_of_week == "Thursday"

    def test_scaffolded_fields_accept_dicts_and_lists(self) -> None:
        detail = GameDetail(
            game_id="2026_01_KC_LAC",
            away_team="KC",
            home_team="LAC",
            team_comparison={"placeholder": "shape TBD"},
            swing_factors=[{"factor": "example"}],
            injuries=[{"player": "example"}],
            top_prop_edges=[{"edge": "example"}],
            _meta={"field_status": {}},  # type: ignore[call-arg]
        )
        assert detail.team_comparison == {"placeholder": "shape TBD"}
        assert detail.swing_factors == [{"factor": "example"}]
