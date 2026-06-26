# tests/unit/api/test_schemas_live.py

"""Unit tests for live game response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.live import (
    DrivePoint,
    LiveGame,
    LiveGameSummary,
    LiveOdds,
    LiveScore,
)


class TestLiveGameConstruction:
    def test_minimal(self) -> None:
        game = LiveGame(game_id="sf-bal")
        assert game.game_id == "sf-bal"
        assert game.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("status", *Blocker.LIVE_STATE)
        game = LiveGame(game_id="sf-bal", response_meta=meta)
        status = game.response_meta.field_status["status"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "live_state_ingest"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("status", *Blocker.LIVE_STATE)
        game = LiveGame(game_id="sf-bal", response_meta=meta)
        dumped = game.model_dump(by_alias=True)
        assert "_meta" in dumped
        assert "response_meta" not in dumped


class TestLiveGameStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            LiveGame(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        game = LiveGame(game_id="sf-bal")
        with pytest.raises(ValidationError):
            game.game_id = "other"


class TestElementShapes:
    def test_live_score_default(self) -> None:
        assert LiveScore() is not None

    def test_live_score_populated(self) -> None:
        score = LiveScore(home=14, away=17)
        assert score.home == 14
        assert score.away == 17

    def test_drive_point_default(self) -> None:
        assert DrivePoint() is not None

    def test_live_odds_default(self) -> None:
        assert LiveOdds() is not None

    def test_live_game_summary_default(self) -> None:
        assert LiveGameSummary() is not None

    def test_live_score_frozen(self) -> None:
        score = LiveScore()
        with pytest.raises(ValidationError):
            score.home = 7

    def test_live_score_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            LiveScore(unexpected="x")


class TestLiveGameComposition:
    def test_holds_score_drives_odds(self) -> None:
        game = LiveGame(
            game_id="sf-bal",
            status="Q3 · 8:42",
            score=LiveScore(home=14, away=17),
            drives=[
                DrivePoint(team="BAL", quarter="Q1", result="TD", wp_change=18.0),
            ],
            odds=[LiveOdds(market="spread", home_line="-4.5", away_line="+4.5")],
        )
        assert game.score.home == 14
        assert game.drives[0].team == "BAL"
        assert game.odds[0].market == "spread"
