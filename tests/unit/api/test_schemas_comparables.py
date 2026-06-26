# tests/unit/api/test_schemas_comparables.py

"""Unit tests for game-comparables response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.comparables import ComparableGame, GameComparables


class TestGameComparablesConstruction:
    def test_minimal(self) -> None:
        comps = GameComparables(game_id="sf-bal")
        assert comps.game_id == "sf-bal"
        assert comps.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked(
            "comparables",
            *Blocker.COMPARABLES,
        )
        comps = GameComparables(game_id="sf-bal", response_meta=meta)
        status = comps.response_meta.field_status["comparables"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "comparables_retrieval"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked(
            "comparables",
            *Blocker.COMPARABLES,
        )
        comps = GameComparables(game_id="sf-bal", response_meta=meta)
        dumped = comps.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameComparablesStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameComparables(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        comps = GameComparables(game_id="sf-bal")
        with pytest.raises(ValidationError):
            comps.game_id = "other"


class TestComparableGame:
    def test_default(self) -> None:
        assert ComparableGame() is not None

    def test_populated(self) -> None:
        comp = ComparableGame(
            date_label="2024 · Wk 11",
            favorite="BAL",
            underdog="PIT",
            line="-5.0",
            final_score="30-23",
            favorite_won=True,
            favorite_covered=True,
            note="Run game ground it out late",
        )
        assert comp.favorite == "BAL"
        assert comp.favorite_won is True

    def test_is_frozen(self) -> None:
        comp = ComparableGame()
        with pytest.raises(ValidationError):
            comp.favorite = "PIT"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComparableGame(unexpected="x")


class TestGameComparablesComposition:
    def test_holds_comparables_and_rates(self) -> None:
        comps = GameComparables(
            game_id="sf-bal",
            comparables=[
                ComparableGame(date_label="2024 · Wk 11", favorite_won=True),
                ComparableGame(date_label="2023 · Wk 14", favorite_won=True),
            ],
            sample_size=17,
            favorite_win_rate=0.71,
            favorite_cover_rate=0.59,
        )
        assert len(comps.comparables) == 2
        assert comps.sample_size == 17
        assert comps.favorite_win_rate == 0.71
