# src/gridiron_edge/api/routes/comparables.py
"""Routes for per-game historical comparables.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at comparables retrieval. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.comparables import GameComparables

router = APIRouter(prefix="/games", tags=["comparables"])


@router.get(
    "/{game_id}/comparables",
    response_model=GameComparables,
    summary="Historical games similar to a single matchup.",
)
def get_game_comparables(game_id: str) -> GameComparables:
    """Return null comparables until comparables retrieval lands."""
    meta = ResponseMeta()
    for field in (
        "comparables",
        "sample_size",
        "favorite_win_rate",
        "favorite_cover_rate",
    ):
        meta = meta.with_blocked(field, *Blocker.COMPARABLES)
    # pyrefly: ignore [unexpected-keyword]
    return GameComparables(game_id=game_id, response_meta=meta)
