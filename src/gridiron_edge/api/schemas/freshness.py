# src/gridiron_edge/api/schemas/freshness.py
"""The market freshness context served beside quotes and edges (``DECISIONS.md`` D80)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MarketFreshness(BaseModel):
    """The server clock and age limit a client needs to say how old the quotes are."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated_at: datetime = Field(
        description="Server time (UTC) the response was built; the age is measured from it.",
    )
    max_age_seconds: int = Field(
        description="Quotes older than this are stale: edges are withheld.",
    )
