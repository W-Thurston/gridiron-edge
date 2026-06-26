# tests/unit/api/test_schemas_prop_reasoning.py

"""Unit tests for per-prop model-reasoning response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.prop_reasoning import PropReasoning, ReasoningEntry


class TestPropReasoningConstruction:
    def test_minimal(self) -> None:
        reasoning = PropReasoning(prop_id="lamar-rush")
        assert reasoning.prop_id == "lamar-rush"
        assert reasoning.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked(
            "entries",
            *Blocker.FEATURE_ATTRIBUTION,
        )
        reasoning = PropReasoning(prop_id="lamar-rush", response_meta=meta)
        status = reasoning.response_meta.field_status["entries"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "feature_attribution"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked(
            "entries",
            *Blocker.FEATURE_ATTRIBUTION,
        )
        reasoning = PropReasoning(prop_id="lamar-rush", response_meta=meta)
        dumped = reasoning.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestPropReasoningStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PropReasoning(prop_id="lamar-rush", unexpected="x")

    def test_is_frozen(self) -> None:
        reasoning = PropReasoning(prop_id="lamar-rush")
        with pytest.raises(ValidationError):
            reasoning.prop_id = "other"


class TestReasoningEntry:
    def test_default(self) -> None:
        assert ReasoningEntry() is not None

    def test_populated(self) -> None:
        entry = ReasoningEntry(
            tag="Volume",
            text="8.1 carries/g, +1.4 vs season avg in cold-weather games.",
            weight="high",
        )
        assert entry.tag == "Volume"
        assert entry.weight == "high"

    def test_is_frozen(self) -> None:
        entry = ReasoningEntry()
        with pytest.raises(ValidationError):
            entry.tag = "Matchup"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningEntry(unexpected="x")


class TestPropReasoningComposition:
    def test_holds_lean_and_entries(self) -> None:
        reasoning = PropReasoning(
            prop_id="lamar-rush",
            lean="OVER",
            entries=[
                ReasoningEntry(tag="Volume", text="...", weight="high"),
                ReasoningEntry(tag="Matchup", text="...", weight="high"),
                ReasoningEntry(tag="Caveat", text="...", weight="low"),
            ],
        )
        assert reasoning.lean == "OVER"
        assert len(reasoning.entries) == 3
        assert reasoning.entries[0].weight == "high"
