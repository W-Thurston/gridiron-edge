# src/gridiron_edge/api/schemas/injuries.py
"""Schemas for per-game injury endpoints (/games/{game_id}/injuries).

Currently blocked on an injury data source decision (ROADMAP §5.3);
responses return null shapes with structured `_meta.field_status`
entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class InjuryReport(BaseModel):
    """A single player injury entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    team: str | None = Field(default=None)
    player: str | None = Field(default=None)
    position: str | None = Field(default=None)
    status: str | None = Field(
        default=None,
        description="'OUT', 'Doubtful', 'Questionable', 'Probable', 'Active'.",
    )
    note: str | None = Field(default=None, description="Free-text status note.")


class GameInjuries(BaseResponse):
    """Response for GET /games/{game_id}/injuries."""

    game_id: str
    reports: list[InjuryReport] | None = Field(default=None)
