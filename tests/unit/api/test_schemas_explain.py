# tests/unit/api/test_schemas_explain.py

"""Unit tests for game-explainability response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.explain import (
    CredibleBand,
    ExplainDistribution,
    ExplainFactor,
    GameExplain,
)


class TestGameExplainConstruction:
    def test_minimal(self) -> None:
        explain = GameExplain(game_id="sf-bal")
        assert explain.game_id == "sf-bal"
        assert explain.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("factors", *Blocker.SCENARIO_ENGINE)
        explain = GameExplain(game_id="sf-bal", response_meta=meta)
        status = explain.response_meta.field_status["factors"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "scenario_engine"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("factors", *Blocker.SCENARIO_ENGINE)
        explain = GameExplain(game_id="sf-bal", response_meta=meta)
        dumped = explain.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameExplainStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameExplain(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        explain = GameExplain(game_id="sf-bal")
        with pytest.raises(ValidationError):
            explain.game_id = "other"


class TestElementShapes:
    def test_credible_band_default(self) -> None:
        assert CredibleBand() is not None

    def test_credible_band_populated(self) -> None:
        band = CredibleBand(point=0.71, lo=0.62, hi=0.78)
        assert band.point == 0.71
        assert band.lo == 0.62
        assert band.hi == 0.78

    def test_explain_factor_default(self) -> None:
        assert ExplainFactor() is not None

    def test_explain_distribution_default(self) -> None:
        assert ExplainDistribution() is not None

    def test_credible_band_frozen(self) -> None:
        band = CredibleBand()
        with pytest.raises(ValidationError):
            band.point = 0.5

    def test_credible_band_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            CredibleBand(unexpected="x")


class TestGameExplainComposition:
    def test_holds_factors_band_distribution(self) -> None:
        explain = GameExplain(
            game_id="sf-bal",
            headline_win_prob=0.71,
            band=CredibleBand(point=0.71, lo=0.62, hi=0.78),
            factors=[
                ExplainFactor(key="rush", label="Rushing matchup", delta=7.0),
            ],
            distribution=ExplainDistribution(samples=2000, mean_margin=5.8, sd=10.5),
        )
        assert explain.band.point == 0.71
        assert explain.factors[0].key == "rush"
        assert explain.distribution.samples == 2000
