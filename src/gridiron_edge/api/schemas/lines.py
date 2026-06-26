# src/gridiron_edge/api/schemas/lines.py
"""Schemas for line-shopping endpoints (/lines and /lines/{game_id}).

These endpoints are currently blocked on multi-book odds ingest;
responses return null shapes with structured `_meta.field_status`
entries pointing at the blocker. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class BookLine(BaseModel):
    """Per-book line for a single market.

    Mirrors the prototype's per-cell book line shape (book id, line,
    price, optional best-price highlight).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    book: str | None = Field(default=None, description="Book identifier (e.g. 'dk', 'fd', 'pin').")
    line: float | None = Field(
        default=None, description="Line value (spread, total, or moneyline)."
    )
    price: int | None = Field(default=None, description="American odds.")
    is_best: bool | None = Field(
        default=None, description="True if this is the best price on the market."
    )


class LineRow(BaseModel):
    """A single row in the /lines grid — one matchup across multiple books."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str | None = Field(default=None)
    market: str | None = Field(default=None, description="'spread', 'total', 'ml', etc.")
    fair_line: float | None = Field(default=None, description="Gridiron Edge fair value.")
    books: list[BookLine] | None = Field(default=None)


class SteamMove(BaseModel):
    """A detected sharp-money line move."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str | None = Field(default=None)
    book: str | None = Field(default=None)
    description: str | None = Field(default=None)
    rationale: str | None = Field(default=None)


class ArbitrageOpportunity(BaseModel):
    """A cross-book lock-in opportunity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    book_a: str | None = Field(default=None)
    side_a: str | None = Field(default=None)
    book_b: str | None = Field(default=None)
    side_b: str | None = Field(default=None)
    edge_pct: float | None = Field(default=None, description="Locked-in edge in percent.")


class LineDetail(BaseResponse):
    """Response for GET /lines/{game_id}."""

    game_id: str
    market: str | None = Field(default=None)
    books: list[BookLine] | None = Field(default=None)
    movement: list[dict] | None = Field(default=None, description="Time-series of consensus line.")
    steam_moves: list[SteamMove] | None = Field(default=None)
    arbitrage: list[ArbitrageOpportunity] | None = Field(default=None)
