# src/gridiron_edge/api/routes/comparables.py
"""Routes for per-game historical comparables.

Serializes persisted comparable-games retrieval evidence for the game's
selected Win forecast event when it exists (ROADMAP.md Tier 3 #10, U14;
`DECISIONS.md` D54).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import load_comparable_games_for_event, load_game
from gridiron_edge.api.schemas.comparables import GameComparables
from gridiron_edge.api.serializers.comparables import serialize_game_comparables
from gridiron_edge.evaluation.comparable_games_evidence_store import (
    AmbiguousComparableGamesError,
)

router = APIRouter(prefix="/games", tags=["comparables"])

_LOGISTIC_MODEL_TYPE = "logistic"
_WIN_AVAILABLE_STATUS = "available"


@router.get(
    "/{game_id}/comparables",
    response_model=GameComparables,
    summary="Historical games similar to a single matchup.",
)
def get_game_comparables(settings: SettingsDep, game_id: str) -> GameComparables:
    """Return one selected game's comparable-games shape."""
    row = load_game(settings, game_id=game_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")

    batch = None
    ambiguous_evidence = False
    if (
        str(row["win_status"]) == _WIN_AVAILABLE_STATUS
        and row.get("win_model_type") == _LOGISTIC_MODEL_TYPE
    ):
        try:
            batch = load_comparable_games_for_event(
                settings,
                event_id=str(row["win_event_id"]),
            )
        except AmbiguousComparableGamesError:
            ambiguous_evidence = True

    return serialize_game_comparables(
        row,
        batch,
        ambiguous_evidence=ambiguous_evidence,
    )
