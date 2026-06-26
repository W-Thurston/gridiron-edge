# src/gridiron_edge/api/schemas/news.py
"""Schemas for news wire endpoints (/news and /news/alerts).

Currently blocked on news ingest; responses return null shapes with
structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NewsItem(BaseModel):
    """A single news wire entry.

    Matches the prototype's `items` shape: timestamp, team tag,
    category, title, body, and betting impact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str | None = Field(default=None, description="ISO timestamp or human-readable time.")
    team: str | None = Field(default=None, description="Team abbreviation, 'MKT', or 'NFL'.")
    category: str | None = Field(
        default=None,
        description="'injury', 'lineup', 'market', 'weather'.",
    )
    title: str | None = Field(default=None)
    body: str | None = Field(default=None)
    betting_impact: str | None = Field(default=None)
    priority: str | None = Field(default=None, description="'high', 'med', 'low'.")
