# src/gridiron_edge/api/routes/live.py
"""Routes for live game endpoints.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at the live state ingest gap. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.live import LiveGame, LiveGameSummary

router = APIRouter(prefix="/live", tags=["live"])


_LIST_META = ResponseMeta().with_blocked("items", *Blocker.LIVE_STATE)


@router.get(
    "",
    response_model=BaseListResponse[LiveGameSummary],
    summary="List of in-progress games.",
)
def list_live() -> BaseListResponse[LiveGameSummary]:
    """Return an empty list until live state ingest lands."""
    return BaseListResponse[LiveGameSummary](
        items=[],
        total=0,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=_LIST_META,
    )


@router.get(
    "/{game_id}",
    response_model=LiveGame,
    summary="Live game state for a single matchup.",
)
def get_live(game_id: str) -> LiveGame:
    """Return a null-shape detail until live state ingest lands."""
    meta = ResponseMeta()
    for field in (
        "status",
        "score",
        "clock",
        "possession",
        "down_distance",
        "yard_line",
        "live_win_prob",
        "drives",
        "odds",
    ):
        meta = meta.with_blocked(field, *Blocker.LIVE_STATE)
    # pyrefly: ignore [unexpected-keyword]
    return LiveGame(game_id=game_id, response_meta=meta)
