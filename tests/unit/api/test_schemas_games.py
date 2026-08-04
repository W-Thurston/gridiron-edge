"""Tests for schedule-complete /games response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.games import (
    GameDetail,
    GameSummary,
    ProjectedScoreBlock,
    SpreadPredictionBlock,
    TotalPredictionBlock,
    WinPredictionBlock,
)


def _blocks():
    return {
        "win": WinPredictionBlock(status="forecast_missing"),
        "spread": SpreadPredictionBlock(status="win_unavailable"),
        "total": TotalPredictionBlock(status="forecast_missing"),
        "projected_score": ProjectedScoreBlock(status="spread_and_total_unavailable"),
    }


def test_unavailable_component_blocks_remain_present() -> None:
    summary = GameSummary(
        game_id="2026_01_KC_LAC",
        away_team="Kansas City Chiefs",
        home_team="Los Angeles Chargers",
        **_blocks(),
    )

    assert summary.win.status == "forecast_missing"
    assert summary.win.home_win_prob is None
    assert summary.total.status == "forecast_missing"
    assert summary.projected_score.home is None


def test_win_and_total_provenance_are_independent() -> None:
    summary = GameSummary(
        game_id="2026_01_KC_LAC",
        away_team="Kansas City Chiefs",
        home_team="Los Angeles Chargers",
        win=WinPredictionBlock(
            status="available",
            event_id="win-event",
            run_id="win-run",
            model_type="elo",
        ),
        spread=SpreadPredictionBlock(status="available", source_event_id="win-event"),
        total=TotalPredictionBlock(
            status="available",
            event_id="total-event",
            run_id="total-run",
            model_type="random_forest",
        ),
        projected_score=ProjectedScoreBlock(status="available", home=25.0, away=22.5),
    )

    assert summary.win.event_id == "win-event"
    assert summary.total.event_id == "total-event"
    assert summary.win.run_id != summary.total.run_id


def test_spread_schema_documents_runtime_sign_convention() -> None:
    schema = SpreadPredictionBlock.model_json_schema()
    description = schema["properties"]["model_spread"]["description"]

    assert "negative means the Home team is favored" in description
    assert "Away score minus projected Home score" in description


def test_game_detail_requires_component_status_blocks() -> None:
    with pytest.raises(ValidationError):
        GameDetail(
            game_id="2026_01_KC_LAC",
            away_team="Kansas City Chiefs",
            home_team="Los Angeles Chargers",
        )


def test_component_blocks_are_frozen_and_forbid_extras() -> None:
    block = WinPredictionBlock(status="available")
    with pytest.raises(ValidationError):
        block.status = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        WinPredictionBlock(status="available", mystery=True)  # type: ignore[call-arg]
