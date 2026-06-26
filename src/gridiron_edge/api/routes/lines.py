# src/gridiron_edge/api/routes/lines.py
"""Routes for line-shopping endpoints.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at the multi-book odds ingest gap. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.lines import LineDetail, LineRow

router = APIRouter(prefix="/lines", tags=["lines"])


_LIST_META = ResponseMeta().with_blocked("items", *Blocker.MULTI_BOOK)


@router.get(
    "",
    response_model=BaseListResponse[LineRow],
    summary="List of matchups with cross-book line grids.",
)
def list_lines() -> BaseListResponse[LineRow]:
    """Return an empty list until multi-book odds ingest lands."""
    return BaseListResponse[LineRow](
        items=[],
        total=0,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=_LIST_META,
    )


@router.get(
    "/{game_id}",
    response_model=LineDetail,
    summary="Line detail for a single matchup.",
)
def get_line(game_id: str) -> LineDetail:
    """Return a null-shape detail until multi-book odds ingest lands."""
    meta = ResponseMeta()
    for field in ("market", "books", "movement", "steam_moves", "arbitrage"):
        meta = meta.with_blocked(field, *Blocker.MULTI_BOOK)
    # pyrefly: ignore [unexpected-keyword]
    return LineDetail(game_id=game_id, response_meta=meta)
