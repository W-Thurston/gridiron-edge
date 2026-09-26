# src/gridiron_edge/api/routes/explain.py
"""Routes for per-game win-probability explainability.

Serializes persisted Logistic explanation evidence for the game's selected
Win forecast event when it exists. `band`, `distribution`, and
`market_implied` remain null pending the scenario engine.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import load_game, load_logistic_explanation_for_event
from gridiron_edge.api.schemas.explain import GameExplain
from gridiron_edge.api.serializers.explain import serialize_game_explain
from gridiron_edge.evaluation.logistic_explanation_evidence_store import (
    AmbiguousLogisticExplanationError,
)

router = APIRouter(prefix="/games", tags=["explain"])

_LOGISTIC_MODEL_TYPE = "logistic"
_WIN_AVAILABLE_STATUS = "available"


@router.get(
    "/{game_id}/explain",
    response_model=GameExplain,
    summary="Win-probability factor decomposition with credible band.",
)
def get_game_explain(settings: SettingsDep, game_id: str) -> GameExplain:
    """Return one selected game's explainability shape."""
    row = load_game(settings, game_id=game_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")

    explanation_event = None
    ambiguous_evidence = False
    if (
        str(row["win_status"]) == _WIN_AVAILABLE_STATUS
        and row.get("win_model_type") == _LOGISTIC_MODEL_TYPE
    ):
        try:
            explanation_event = load_logistic_explanation_for_event(
                settings,
                event_id=str(row["win_event_id"]),
            )
        except AmbiguousLogisticExplanationError:
            ambiguous_evidence = True

    return serialize_game_explain(
        row,
        explanation_event,
        ambiguous_evidence=ambiguous_evidence,
    )
