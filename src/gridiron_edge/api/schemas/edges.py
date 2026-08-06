# src/gridiron_edge/api/schemas/edges.py

"""Schemas for /edges."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.market.edge import EdgeStrength
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeResultState,
)


class EdgeProvenanceResponse(BaseModel):
    """Prediction, product, and market provenance supplied by the service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    win_event_ids: tuple[str, ...] = ()
    win_run_ids: tuple[str, ...] = ()
    win_model_names: tuple[str, ...] = ()
    win_model_types: tuple[str, ...] = ()
    total_event_ids: tuple[str, ...] = ()
    total_run_ids: tuple[str, ...] = ()
    total_model_names: tuple[str, ...] = ()
    total_model_types: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    product_run_ids: tuple[str, ...] = ()
    market_providers: tuple[str, ...] = ()
    market_sportsbooks: tuple[str, ...] = ()
    market_fetched_at: tuple[datetime, ...] = ()


class EdgeDiagnosticsResponse(BaseModel):
    """Complete service-provided diagnostics for one weekly edge result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str
    week: int
    prediction_game_count: int
    market_game_count: int
    matched_game_count: int
    complete_moneyline_count: int = Field(
        description="Complete sportsbook-game moneyline pairs.",
    )
    complete_spread_count: int = Field(
        description="Complete sportsbook-game spread pairs.",
    )
    complete_total_count: int = Field(
        description="Complete sportsbook-game total pairs.",
    )
    eligible_market_count: int = Field(
        description="Total complete sportsbook-game market families.",
    )
    calculated_edge_count: int
    positive_edge_count: int
    filtered_edge_count: int
    state: EdgeResultState
    blockers: tuple[EdgeDiagnosticBlocker, ...] = ()
    provenance: EdgeProvenanceResponse = Field(
        default_factory=EdgeProvenanceResponse,
    )


class EdgeRow(BaseModel):
    """A single edge in the ranked edge report.

    One row per provider-event-sportsbook-game-market-side offer.
    Moneyline rows have ``point_edge`` and ``cover_prob`` as null;
    spread/total rows populate them. Fields mirror ``_REPORT_COLUMNS`` in
    ``market.recommendations`` and preserve exact quote provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Quote provenance
    provider: str | None = None
    provider_event_id: str | None = None
    sportsbook: str | None = None
    market_fetched_at: datetime | None = None
    sportsbook_updated_at: datetime | None = None
    commence_time: datetime | None = None

    # Game context
    game_id: str
    game_date: str | None = None
    season: str | None = None
    week: int | None = None
    away_team: str
    home_team: str

    # Model identity
    model_key: str = Field(
        description="Composite {model_name}_{model_type} identifier.",
    )
    confidence_tier: str | None = Field(
        default=None,
        description="'Low', 'Moderate', or 'High' — from prediction row.",
    )

    # Market
    market_type: str = Field(
        description="'moneyline', 'spread', or 'total'.",
    )
    side: str = Field(
        description="'home', 'away', 'over', or 'under'.",
    )
    model_value: float | None = None
    market_value: float | None = Field(
        default=None,
        description=(
            "Market context value: no-vig implied probability for "
            "moneyline, home-team spread for spread, or market total "
            "for total."
        ),
    )
    american_odds: int = Field(
        description=("American price used to calculate EV and Kelly for this edge."),
    )
    point_edge: float | None = Field(
        default=None,
        description="Points of edge (spread/total). Null for moneyline.",
    )
    cover_prob: float | None = Field(
        default=None,
        description="Model-implied cover probability. Null for moneyline.",
    )

    # Bet economics
    ev: float
    edge_strength: EdgeStrength = Field(
        description="'strong', 'moderate', 'lean', or 'no_edge'.",
    )
    kelly_frac: float | None = None
    kelly_stake: float | None = None


class EdgeList(BaseListResponse[EdgeRow]):
    """Response for GET /edges."""

    season: str | None = Field(default=None)
    week: int | None = Field(default=None)
    min_ev: float | None = Field(
        default=None,
        description="Minimum EV threshold applied to the report.",
    )
    bankroll: float | None = Field(
        default=None,
        description=(
            "Bankroll basis used to calculate kelly_stake. "
            "None means dollar sizing was not requested."
        ),
    )

    kelly_multiplier: float | None = Field(
        default=None,
        description=("Fraction of full Kelly applied when calculating kelly_stake."),
    )
    diagnostics: EdgeDiagnosticsResponse = Field(
        description="Complete diagnostics returned by the unified edge service.",
    )
