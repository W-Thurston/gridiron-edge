# src/gridiron_edge/api/schemas/prop_reasoning.py
"""Schemas for per-prop reasoning endpoints (/props/{prop_id}/reasoning).

Currently blocked on feature attribution; responses return null shapes
with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ReasoningEntry(BaseModel):
    """A single 'why the model leans X' entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str | None = Field(default=None, description="Short category (e.g. 'Volume').")
    text: str | None = Field(default=None, description="One-sentence rationale.")
    weight: str | None = Field(
        default=None,
        description="Qualitative strength: 'high', 'med', 'low'.",
    )


class PropReasoning(BaseResponse):
    """Response for GET /props/{prop_id}/reasoning."""

    prop_id: str
    lean: str | None = Field(default=None, description="'OVER', 'UNDER', or 'No Edge'.")
    entries: list[ReasoningEntry] | None = Field(default=None)
