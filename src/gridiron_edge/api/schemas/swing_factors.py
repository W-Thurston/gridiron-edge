# src/gridiron_edge/api/schemas/swing_factors.py
"""Schemas for per-game swing-factor endpoints (/games/{game_id}/swing-factors).

Currently blocked on feature attribution; responses return null shapes
with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class SwingFactor(BaseModel):
    """A single swing-factor entry — tag, description, leaning team."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str | None = Field(default=None, description="Short category (e.g. 'Run game').")
    text: str | None = Field(default=None, description="One-sentence rationale.")
    leans_to: str | None = Field(
        default=None,
        description="Team abbreviation favored by this factor, or None for neutral.",
    )


class GameSwingFactors(BaseResponse):
    """Response for GET /games/{game_id}/swing-factors."""

    game_id: str
    factors: list[SwingFactor] | None = Field(default=None)
