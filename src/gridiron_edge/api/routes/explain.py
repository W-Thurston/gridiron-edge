# src/gridiron_edge/api/routes/explain.py
"""Routes for per-game win-probability explainability.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at the scenario engine. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.explain import GameExplain

router = APIRouter(prefix="/games", tags=["explain"])


@router.get(
    "/{game_id}/explain",
    response_model=GameExplain,
    summary="Win-probability factor decomposition with credible band.",
)
def get_game_explain(game_id: str) -> GameExplain:
    """Return a null explainability shape until the scenario engine lands."""
    meta = ResponseMeta()
    for field in (
        "headline_win_prob",
        "band",
        "factors",
        "distribution",
        "market_implied",
    ):
        meta = meta.with_blocked(field, *Blocker.SCENARIO_ENGINE)
    # pyrefly: ignore [unexpected-keyword]
    return GameExplain(game_id=game_id, response_meta=meta)
