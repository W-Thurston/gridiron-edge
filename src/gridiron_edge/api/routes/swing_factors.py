# src/gridiron_edge/api/routes/swing_factors.py
"""Routes for per-game swing factors.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at feature attribution. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.swing_factors import GameSwingFactors

router = APIRouter(prefix="/games", tags=["swing-factors"])


@router.get(
    "/{game_id}/swing-factors",
    response_model=GameSwingFactors,
    summary="Top factors driving a single matchup's model lean.",
)
def get_game_swing_factors(game_id: str) -> GameSwingFactors:
    """Return null swing factors until feature attribution lands."""
    meta = ResponseMeta().with_blocked(
        "factors",
        *Blocker.FEATURE_ATTRIBUTION,
    )
    # pyrefly: ignore [unexpected-keyword]
    return GameSwingFactors(game_id=game_id, response_meta=meta)
