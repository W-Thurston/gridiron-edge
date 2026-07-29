# src/gridiron_edge/api/schemas/edges.py

"""Schemas for /edges."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse


class EdgeRow(BaseModel):
    """A single edge in the ranked edge report.

    One row per (game, market_type, side) triple. Moneyline rows have
    ``point_edge`` and ``cover_prob`` as null; spread/total rows populate
    them. Fields mirror ``_REPORT_COLUMNS`` in
    ``market.recommendations``, with team names as short codes after
    loader-side normalization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    edge_strength: str = Field(
        description="'strong', 'moderate', or 'weak'.",
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
