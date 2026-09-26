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
        explain = GameExplain.model_validate(
            {
                "game_id": "sf-bal",
                "response_meta": meta,
            }
        )
        assert explain.response_meta is not None
        status = explain.response_meta.field_status["factors"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "scenario_engine"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("factors", *Blocker.SCENARIO_ENGINE)
        explain = GameExplain.model_validate(
            {
                "game_id": "sf-bal",
                "response_meta": meta,
            }
        )
        dumped = explain.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameExplainStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameExplain.model_validate(
                {
                    "game_id": "sf-bal",
                    "unexpected": "x",
                }
            )

    def test_is_frozen(self) -> None:
        explain = GameExplain(game_id="sf-bal")
        with pytest.raises(ValidationError):
            setattr(explain, "game_id", "other")  # noqa: B010


class TestElementShapes:
    def test_credible_band_default(self) -> None:
        assert CredibleBand() is not None

    def test_credible_band_populated(self) -> None:
        band = CredibleBand(point=0.71, lo=0.62, hi=0.78)
        assert band.point == 0.71
        assert band.lo == 0.62
        assert band.hi == 0.78

    def test_explain_factor_default(self) -> None:
        factor = ExplainFactor()
        assert factor.log_odds_contribution is None
        assert factor.coefficient is None
        assert factor.transformed_value is None

    def test_explain_factor_populated(self) -> None:
        factor = ExplainFactor(
            key="ELO_DIFF",
            label="ELO_DIFF",
            log_odds_contribution=0.512,
            coefficient=0.256,
            transformed_value=2.0,
            is_baseline=False,
            is_adjustable=False,
        )
        assert factor.log_odds_contribution == 0.512
        assert factor.coefficient == 0.256
        assert factor.transformed_value == 2.0

    def test_explain_factor_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            ExplainFactor.model_validate({"unexpected": "x"})

    def test_explain_distribution_default(self) -> None:
        assert ExplainDistribution() is not None

    def test_credible_band_frozen(self) -> None:
        band = CredibleBand()
        with pytest.raises(ValidationError):
            setattr(band, "point", 0.5)  # noqa: B010

    def test_credible_band_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            CredibleBand.model_validate(
                {
                    "unexpected": "x",
                }
            )


class TestGameExplainComposition:
    def test_holds_factors_band_distribution(self) -> None:
        explain = GameExplain(
            game_id="sf-bal",
            headline_win_prob=0.71,
            band=CredibleBand(point=0.71, lo=0.62, hi=0.78),
            factors=[
                ExplainFactor(
                    key="rush",
                    label="Rushing matchup",
                    log_odds_contribution=0.34,
                    coefficient=0.12,
                    transformed_value=2.83,
                ),
            ],
            distribution=ExplainDistribution(samples=2000, mean_margin=5.8, sd=10.5),
        )
        assert explain.band is not None
        assert explain.band.point == 0.71
        assert explain.factors is not None
        assert explain.factors[0].key == "rush"
        assert explain.distribution is not None
        assert explain.distribution.samples == 2000
