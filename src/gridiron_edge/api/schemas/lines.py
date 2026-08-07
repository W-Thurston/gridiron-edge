"""Schemas for current multi-book line shopping."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse

MarketName = Literal["moneyline", "spread", "total"]
MarketSide = Literal["away", "home", "over", "under"]
GuidanceStatus = Literal[
    "available",
    "model_unavailable",
    "uncertainty_unavailable",
]


class LineOutcomeGuidance(BaseModel):
    """Selected-product guidance for one game, market, and side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    side: MarketSide
    model_status: GuidanceStatus
    model_value: float | None = None
    playable_line: float | None = None
    reference_odds: int | None = None
    fair_american_odds: int | None = None
    product_id: str | None = None
    product_run_id: str | None = None


class LineOffer(BaseModel):
    """One exact sportsbook quote with deterministic comparison flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    provider_event_id: str | None = None
    sportsbook: str
    sportsbook_updated_at: datetime | None = None
    market_fetched_at: datetime
    commence_time: datetime | None = None
    is_live: bool
    market: MarketName
    side: MarketSide
    line: float | None = None
    american_odds: int
    is_best_line: bool
    is_best_price: bool
    model_status: GuidanceStatus = "model_unavailable"
    model_value: float | None = None
    model_probability: float | None = None
    expected_value: float | None = None
    is_model_approved: bool | None = None
    is_best_model_approved_offer: bool = False
    product_id: str | None = None
    product_run_id: str | None = None


class LineShoppingGame(BaseModel):
    """One scheduled game and every current quote in the requested scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str
    season: str
    week: int
    game_date: str
    away_team: str
    home_team: str
    commence_time: datetime | None = None
    offers: list[LineOffer] = Field(default_factory=list)
    guidance: list[LineOutcomeGuidance] = Field(default_factory=list)


class LineShoppingList(BaseListResponse[LineShoppingGame]):
    """Current slate-wide sportsbook comparison response."""

    season: str
    week: int
    market: MarketName | None = None
    sportsbooks: tuple[str, ...] = ()
    market_fetched_at: tuple[datetime, ...] = ()
