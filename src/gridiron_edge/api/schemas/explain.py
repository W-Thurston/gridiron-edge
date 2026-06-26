# src/gridiron_edge/api/schemas/explain.py
"""Schemas for per-game explainability endpoints (/games/{game_id}/explain).

Currently blocked on the scenario engine; responses return null shapes
with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ExplainFactor(BaseModel):
    """A single factor in the win-prob waterfall."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str | None = Field(default=None, description="Stable factor identifier.")
    label: str | None = Field(default=None, description="Human-readable label.")
    description: str | None = Field(default=None, description="One-line rationale.")
    delta: float | None = Field(
        default=None,
        description="Percentage-point contribution to win prob.",
    )
    is_baseline: bool | None = Field(default=None)
    is_adjustable: bool | None = Field(default=None)


class CredibleBand(BaseModel):
    """90% credible interval around the headline probability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: float | None = Field(default=None)
    lo: float | None = Field(default=None)
    hi: float | None = Field(default=None)


class ExplainDistribution(BaseModel):
    """Simulated outcome distribution for the explain view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int | None = Field(default=None, description="Number of sims (e.g. 2000).")
    mean_margin: float | None = Field(
        default=None,
        description="Expected margin in points (home - away).",
    )
    sd: float | None = Field(default=None, description="Standard deviation of margin.")


class GameExplain(BaseResponse):
    """Response for GET /games/{game_id}/explain."""

    game_id: str
    headline_win_prob: float | None = Field(default=None)
    band: CredibleBand | None = Field(default=None)
    factors: list[ExplainFactor] | None = Field(default=None)
    distribution: ExplainDistribution | None = Field(default=None)
    market_implied: float | None = Field(
        default=None,
        description="De-vigged market implied prob for the favored side.",
    )
