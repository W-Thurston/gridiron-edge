# tests/unit/api/test_schemas_news.py

"""Unit tests for news wire response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.news import NewsItem


class TestNewsItem:
    def test_default_construction(self) -> None:
        assert NewsItem() is not None

    def test_populated(self) -> None:
        item = NewsItem(
            timestamp="14:22",
            team="KC",
            category="lineup",
            title="Pacheco activated from IR",
            body="Per beat report, Pacheco returns this week.",
            betting_impact="No edge identified.",
            priority="med",
        )
        assert item.team == "KC"
        assert item.priority == "med"

    def test_is_frozen(self) -> None:
        item = NewsItem()
        with pytest.raises(ValidationError):
            item.team = "BUF"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            NewsItem(unexpected="x")
