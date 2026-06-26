# tests/unit/api/test_schemas_swing_factors.py

"""Unit tests for swing-factors response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.swing_factors import GameSwingFactors, SwingFactor


class TestGameSwingFactorsConstruction:
    def test_minimal(self) -> None:
        factors = GameSwingFactors(game_id="sf-bal")
        assert factors.game_id == "sf-bal"
        assert factors.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("factors", *Blocker.FEATURE_ATTRIBUTION)
        factors = GameSwingFactors(game_id="sf-bal", response_meta=meta)
        status = factors.response_meta.field_status["factors"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "feature_attribution"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("factors", *Blocker.FEATURE_ATTRIBUTION)
        factors = GameSwingFactors(game_id="sf-bal", response_meta=meta)
        dumped = factors.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameSwingFactorsStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameSwingFactors(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        factors = GameSwingFactors(game_id="sf-bal")
        with pytest.raises(ValidationError):
            factors.game_id = "other"


class TestSwingFactor:
    def test_default(self) -> None:
        assert SwingFactor() is not None

    def test_populated(self) -> None:
        factor = SwingFactor(
            tag="Run game",
            text="Jackson + Henry vs SF run D.",
            leans_to="BAL",
        )
        assert factor.tag == "Run game"
        assert factor.leans_to == "BAL"

    def test_is_frozen(self) -> None:
        factor = SwingFactor()
        with pytest.raises(ValidationError):
            factor.tag = "Pass rush"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            SwingFactor(unexpected="x")


class TestGameSwingFactorsComposition:
    def test_holds_factors(self) -> None:
        factors = GameSwingFactors(
            game_id="sf-bal",
            factors=[
                SwingFactor(tag="Run game", text="...", leans_to="BAL"),
                SwingFactor(tag="Pass rush", text="...", leans_to=None),
            ],
        )
        assert len(factors.factors) == 2
        assert factors.factors[0].leans_to == "BAL"
