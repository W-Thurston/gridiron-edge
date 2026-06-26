# src/gridiron_edge/api/routes/injuries.py
"""Routes for per-game injury reports.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at the injury data source decision. See ROADMAP §5.3
and §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.injuries import GameInjuries

router = APIRouter(prefix="/games", tags=["injuries"])


@router.get(
    "/{game_id}/injuries",
    response_model=GameInjuries,
    summary="Injury reports for both teams in a single matchup.",
)
def get_game_injuries(game_id: str) -> GameInjuries:
    """Return a null explainability shape until the scenario engine lands."""
    meta = ResponseMeta().with_blocked("reports", *Blocker.INJURY_DATA)
    # pyrefly: ignore [unexpected-keyword]
    return GameInjuries(game_id=game_id, response_meta=meta)
