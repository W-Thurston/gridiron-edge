# tests/unit/api/test_schemas_comparables.py

"""Unit tests for game-comparables response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.comparables import ComparableFactor, ComparableGame, GameComparables


def _match(**overrides: object) -> ComparableGame:
    defaults: dict[str, object] = {
        "game_id": "2024_11_PIT_BAL",
        "rank": 1,
        "distance": 8.59,
        "season": "2024-2025",
        "week": 11,
        "game_date": "2024-11-17",
        "away_team": "Pittsburgh Steelers",
        "home_team": "Baltimore Ravens",
        "away_score": 23,
        "home_score": 30,
    }
    defaults.update(overrides)
    return ComparableGame(**defaults)


class TestGameComparablesConstruction:
    def test_minimal(self) -> None:
        comps = GameComparables(game_id="sf-bal")
        assert comps.game_id == "sf-bal"
        assert comps.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("comparables", *Blocker.COMPARABLES)
        comps = GameComparables.model_validate(
            {
                "game_id": "sf-bal",
                "response_meta": meta,
            }
        )
        assert comps.response_meta is not None
        status = comps.response_meta.field_status["comparables"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "comparables_retrieval"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("comparables", *Blocker.COMPARABLES)
        comps = GameComparables.model_validate(
            {
                "game_id": "sf-bal",
                "response_meta": meta,
            }
        )
        dumped = comps.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameComparablesStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameComparables.model_validate(
                {
                    "game_id": "sf-bal",
                    "unexpected": "x",
                }
            )

    def test_is_frozen(self) -> None:
        comps = GameComparables(game_id="sf-bal")
        with pytest.raises(ValidationError):
            setattr(comps, "game_id", "other")  # noqa: B010


class TestComparableGame:
    def test_requires_core_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComparableGame()

    def test_populated(self) -> None:
        comp = _match(favorite_team="Baltimore Ravens", spread_magnitude=3.0, favorite_won=True)
        assert comp.favorite_team == "Baltimore Ravens"
        assert comp.favorite_won is True
        assert comp.top_contributing_features == []

    def test_favorite_fields_default_to_none(self) -> None:
        comp = _match()
        assert comp.favorite_team is None
        assert comp.spread_magnitude is None
        assert comp.favorite_won is None
        assert comp.favorite_covered is None

    def test_is_frozen(self) -> None:
        comp = _match()
        with pytest.raises(ValidationError):
            setattr(comp, "favorite_team", "Someone")  # noqa: B010

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComparableGame.model_validate(
                {
                    "unexpected": "x",
                }
            )


class TestComparableFactor:
    def test_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            ComparableFactor()

    def test_populated(self) -> None:
        factor = ComparableFactor(
            feature_name="ELO_DIFF",
            query_value=0.1,
            candidate_value=0.2,
            squared_difference=0.01,
        )
        assert factor.feature_name == "ELO_DIFF"
        assert factor.squared_difference == 0.01

    def test_is_frozen(self) -> None:
        factor = ComparableFactor(
            feature_name="ELO_DIFF", query_value=0.1, candidate_value=0.2, squared_difference=0.01
        )
        with pytest.raises(ValidationError):
            setattr(factor, "feature_name", "OTHER")  # noqa: B010


class TestGameComparablesComposition:
    def test_holds_comparables_and_rates(self) -> None:
        comps = GameComparables(
            game_id="sf-bal",
            comparables=[
                _match(favorite_won=True),
                _match(game_id="2003_11_WAS_CAR", favorite_won=True),
            ],
            sample_size=2,
            favorite_win_rate=1.0,
            favorite_cover_rate=0.5,
        )
        assert comps.comparables is not None
        assert len(comps.comparables) == 2
        assert comps.sample_size == 2
        assert comps.favorite_win_rate == 1.0

    def test_comparable_carries_top_contributing_features(self) -> None:
        comp = _match(
            top_contributing_features=[
                ComparableFactor(
                    feature_name="DEF_PENALTY_RATE_DIFF",
                    query_value=-2.32,
                    candidate_value=0.15,
                    squared_difference=6.13,
                )
            ]
        )
        assert comp.top_contributing_features[0].feature_name == "DEF_PENALTY_RATE_DIFF"
