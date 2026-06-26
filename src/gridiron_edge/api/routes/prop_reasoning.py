# src/gridiron_edge/api/routes/prop_reasoning.py
"""Routes for per-prop model reasoning.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at feature attribution. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.prop_reasoning import PropReasoning

router = APIRouter(prefix="/props", tags=["prop-reasoning"])


@router.get(
    "/{prop_id}/reasoning",
    response_model=PropReasoning,
    summary="Factor-level rationale behind the model's lean on a single prop.",
)
def get_prop_reasoning(prop_id: str) -> PropReasoning:
    """Return null reasoning entries until feature attribution lands."""
    meta = ResponseMeta()
    for field in ("lean", "entries"):
        meta = meta.with_blocked(field, *Blocker.FEATURE_ATTRIBUTION)
    # pyrefly: ignore [unexpected-keyword]
    return PropReasoning(prop_id=prop_id, response_meta=meta)
