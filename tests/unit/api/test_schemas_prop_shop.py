# tests/unit/api/test_schemas_prop_shop.py

"""Unit tests for per-prop multi-book shopping response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.prop_shop import PropBookLine, PropShop


class TestPropShopConstruction:
    def test_minimal(self) -> None:
        shop = PropShop(prop_id="lamar-rush")
        assert shop.prop_id == "lamar-rush"
        assert shop.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("books", *Blocker.MULTI_BOOK)
        shop = PropShop(prop_id="lamar-rush", response_meta=meta)
        status = shop.response_meta.field_status["books"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "multi_book_ingest"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("books", *Blocker.MULTI_BOOK)
        shop = PropShop(prop_id="lamar-rush", response_meta=meta)
        dumped = shop.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestPropShopStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PropShop(prop_id="lamar-rush", unexpected="x")

    def test_is_frozen(self) -> None:
        shop = PropShop(prop_id="lamar-rush")
        with pytest.raises(ValidationError):
            shop.prop_id = "other"


class TestPropBookLine:
    def test_default(self) -> None:
        assert PropBookLine() is not None

    def test_populated(self) -> None:
        line = PropBookLine(
            book="Pinnacle",
            line=49.5,
            price=-115,
            is_best_over=True,
            is_best_under=False,
        )
        assert line.book == "Pinnacle"
        assert line.is_best_over is True

    def test_is_frozen(self) -> None:
        line = PropBookLine()
        with pytest.raises(ValidationError):
            line.book = "dk"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PropBookLine(unexpected="x")


class TestPropShopComposition:
    def test_holds_books_and_bests(self) -> None:
        shop = PropShop(
            prop_id="lamar-rush",
            books=[
                PropBookLine(book="Pinnacle", line=49.5, price=-115, is_best_over=True),
                PropBookLine(book="DraftKings", line=51.5, price=-110),
            ],
            best_over=PropBookLine(book="Pinnacle", line=49.5, price=-115),
        )
        assert len(shop.books) == 2
        assert shop.best_over.book == "Pinnacle"
