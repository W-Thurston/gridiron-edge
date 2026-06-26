# src/gridiron_edge/api/schemas/prop_shop.py
"""Schemas for per-prop multi-book endpoints (/props/{prop_id}/shop).

Currently blocked on multi-book odds ingest; responses return null
shapes with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class PropBookLine(BaseModel):
    """A single book's line and price for a prop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    book: str | None = Field(default=None)
    line: float | None = Field(default=None)
    price: int | None = Field(default=None, description="American odds.")
    is_best_over: bool | None = Field(default=None)
    is_best_under: bool | None = Field(default=None)


class PropShop(BaseResponse):
    """Response for GET /props/{prop_id}/shop."""

    prop_id: str
    books: list[PropBookLine] | None = Field(default=None)
    best_over: PropBookLine | None = Field(default=None)
    best_under: PropBookLine | None = Field(default=None)
