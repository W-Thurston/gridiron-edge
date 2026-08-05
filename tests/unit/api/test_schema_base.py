# tests/unit/api/test_schemas_base.py
"""Unit tests for api/schemas/_base.py.

Covers:
- BaseResponse construction with and without _meta.
- _meta alias handling (wire shape vs. Python attribute).
- frozen / extra=forbid semantics.
- BaseListResponse[T] generic instantiation.
- List-level blocking via field_status["items"].
- JSON round-trip with mixed populated and blocked responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse

# ---------------------------------------------------------------------------
# Test fixtures — minimal subclass models that exercise the base shapes.
# ---------------------------------------------------------------------------


class _SampleObject(BaseResponse):
    """Minimal BaseResponse subclass for tests."""

    name: str
    value: int | None = None


class _SampleItem(BaseModel):
    """Minimal element type for BaseListResponse[T] tests."""

    model_config = {"frozen": True, "extra": "forbid"}

    label: str
    score: float


# ---------------------------------------------------------------------------
# BaseResponse
# ---------------------------------------------------------------------------


class TestBaseResponseConstruction:
    def test_without_meta(self) -> None:
        obj = _SampleObject(name="alpha", value=1)
        assert obj.name == "alpha"
        assert obj.value == 1
        assert obj.response_meta is None

    def test_with_meta_via_attribute_name(self) -> None:
        meta = ResponseMeta().with_pending("value")
        obj = _SampleObject(name="alpha", response_meta=meta)
        assert obj.response_meta is meta

    def test_with_meta_via_wire_alias(self) -> None:
        """Construction via the `_meta` alias should work (populate_by_name)."""
        meta = ResponseMeta().with_pending("value")
        # mypy/pyrefly may flag this; the construction is intentional for
        # round-tripping from JSON.
        obj = _SampleObject.model_validate(
            {"name": "alpha", "_meta": meta.model_dump()},
        )
        assert obj.response_meta is not None
        assert obj.response_meta.field_status == {"value": "pending"}


class TestBaseResponseSerialization:
    def test_omits_meta_key_when_none(self) -> None:
        obj = _SampleObject(name="alpha", value=1)
        dumped = obj.model_dump(by_alias=True, exclude_none=True)
        assert "_meta" not in dumped
        assert dumped == {"name": "alpha", "value": 1}

    def test_uses_wire_alias_meta(self) -> None:
        meta = ResponseMeta().with_blocked("value", *Blocker.INJURY_DATA)
        obj = _SampleObject(name="alpha", response_meta=meta)
        dumped = obj.model_dump(by_alias=True)
        assert "_meta" in dumped
        assert "response_meta" not in dumped
        assert dumped["_meta"]["field_status"]["value"] == {
            "status": "blocked",
            "blocker": "injury_data_source",
            "roadmap": "injury data source",
        }

    def test_json_round_trip(self) -> None:
        meta = ResponseMeta().with_pending("value").with_blocked("name", *Blocker.GAMEDAY_METADATA)
        obj = _SampleObject(name="alpha", value=None, response_meta=meta)
        json_str = obj.model_dump_json(by_alias=True)
        rebuilt = _SampleObject.model_validate_json(json_str)
        assert rebuilt.name == "alpha"
        assert rebuilt.value is None
        assert rebuilt.response_meta is not None
        assert rebuilt.response_meta.field_status["value"] == "pending"
        blocked = rebuilt.response_meta.field_status["name"]
        assert isinstance(blocked, BlockedStatus)
        assert blocked.blocker == "gameday_metadata"


class TestBaseResponseStrict:
    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SampleObject(
                name="alpha",
                value=1,
                unexpected="x",  # type: ignore[call-arg]
            )

    def test_misspelled_meta_rejected(self) -> None:
        """Surfaces typos like `reponse_meta=...` at construction time."""
        with pytest.raises(ValidationError):
            _SampleObject(
                name="alpha",
                reponse_meta=ResponseMeta(),  # type: ignore[call-arg]
            )

    def test_frozen_blocks_attribute_assignment(self) -> None:
        obj = _SampleObject(name="alpha")
        with pytest.raises(ValidationError):
            obj.name = "beta"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BaseListResponse[T]
# ---------------------------------------------------------------------------


class TestBaseListResponseConstruction:
    def test_empty_default(self) -> None:
        resp: BaseListResponse[_SampleItem] = BaseListResponse[_SampleItem]()
        assert resp.items == []
        assert resp.total is None
        assert resp.response_meta is None

    def test_populated(self) -> None:
        items = [_SampleItem(label="x", score=1.0), _SampleItem(label="y", score=2.0)]
        resp = BaseListResponse[_SampleItem](items=items, total=2)
        assert resp.items == items
        assert resp.total == 2

    def test_total_optional(self) -> None:
        resp = BaseListResponse[_SampleItem](
            items=[_SampleItem(label="x", score=1.0)],
        )
        assert resp.total is None


class TestBaseListResponseBlocking:
    def test_marks_items_blocked_via_field_status(self) -> None:
        meta = ResponseMeta().with_blocked("items", *Blocker.MULTI_BOOK)
        resp = BaseListResponse[_SampleItem](
            items=[],
            total=0,
            response_meta=meta,
        )
        assert resp.items == []
        assert resp.total == 0
        assert resp.response_meta is not None
        status = resp.response_meta.field_status["items"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "multi_book_ingest"
        assert status.roadmap == "multi-book markets"

    def test_serializes_blocked_list_with_wire_shape(self) -> None:
        meta = ResponseMeta().with_blocked("items", *Blocker.LIVE_STATE)
        resp = BaseListResponse[_SampleItem](
            items=[],
            total=0,
            response_meta=meta,
        )
        dumped = resp.model_dump(by_alias=True)
        assert dumped["items"] == []
        assert dumped["total"] == 0
        assert dumped["_meta"]["field_status"]["items"] == {
            "status": "blocked",
            "blocker": "live_state_ingest",
            "roadmap": "live-game data",
        }


class TestBaseListResponseGenerics:
    def test_distinct_parameterizations_isolated(self) -> None:
        """BaseListResponse[A] and BaseListResponse[B] are distinct types."""

        class _A(BaseModel):
            model_config = {"frozen": True, "extra": "forbid"}
            a: int

        class _B(BaseModel):
            model_config = {"frozen": True, "extra": "forbid"}
            b: str

        ra = BaseListResponse[_A](items=[_A(a=1)])
        rb = BaseListResponse[_B](items=[_B(b="x")])

        assert ra.items[0].a == 1
        assert rb.items[0].b == "x"

    def test_rejects_wrong_item_type(self) -> None:
        with pytest.raises(ValidationError):
            BaseListResponse[_SampleItem](
                items=[{"label": "x"}],  # missing required `score`
            )


class TestBaseListResponseStrict:
    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseListResponse[_SampleItem](
                items=[],
                page=1,  # type: ignore[call-arg]
            )

    def test_frozen_blocks_items_reassignment(self) -> None:
        resp = BaseListResponse[_SampleItem](items=[])
        with pytest.raises(ValidationError):
            resp.items = [_SampleItem(label="x", score=1.0)]  # type: ignore[misc]
