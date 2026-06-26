# tests/unit/api/test_schemas_lines.py

"""Unit tests for line-shopping response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.lines import (
    ArbitrageOpportunity,
    BookLine,
    LineDetail,
    LineRow,
    SteamMove,
)


class TestLineDetailConstruction:
    def test_minimal(self) -> None:
        detail = LineDetail(game_id="sf-bal")
        assert detail.game_id == "sf-bal"
        assert detail.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("books", *Blocker.MULTI_BOOK)
        detail = LineDetail(game_id="sf-bal", response_meta=meta)
        status = detail.response_meta.field_status["books"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "multi_book_ingest"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("books", *Blocker.MULTI_BOOK)
        detail = LineDetail(game_id="sf-bal", response_meta=meta)
        dumped = detail.model_dump(by_alias=True)
        assert "_meta" in dumped
        assert "response_meta" not in dumped


class TestLineDetailStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            LineDetail(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        detail = LineDetail(game_id="sf-bal")
        with pytest.raises(ValidationError):
            detail.game_id = "other"


class TestElementShapes:
    def test_book_line_default(self) -> None:
        assert BookLine() is not None

    def test_book_line_populated(self) -> None:
        line = BookLine(book="dk", line=-4.5, price=-110, is_best=True)
        assert line.book == "dk"
        assert line.is_best is True

    def test_line_row_default(self) -> None:
        assert LineRow() is not None

    def test_steam_move_default(self) -> None:
        assert SteamMove() is not None

    def test_arbitrage_opportunity_default(self) -> None:
        assert ArbitrageOpportunity() is not None

    def test_book_line_frozen(self) -> None:
        line = BookLine()
        with pytest.raises(ValidationError):
            line.book = "dk"

    def test_book_line_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            BookLine(unexpected="x")


class TestLineDetailComposition:
    def test_holds_books_steam_arbitrage(self) -> None:
        detail = LineDetail(
            game_id="sf-bal",
            books=[BookLine(book="dk", line=-4.5, price=-110, is_best=False)],
            steam_moves=[
                SteamMove(
                    timestamp="2:18 PM",
                    book="Pinnacle",
                    description="BAL -4.5 → -5",
                    rationale="Sharp money home",
                ),
            ],
            arbitrage=[
                ArbitrageOpportunity(
                    book_a="fd",
                    side_a="BAL +5",
                    book_b="dk",
                    side_b="SF -4.5",
                    edge_pct=0.4,
                ),
            ],
        )
        assert detail.books[0].book == "dk"
        assert detail.steam_moves[0].book == "Pinnacle"
        assert detail.arbitrage[0].edge_pct == 0.4
