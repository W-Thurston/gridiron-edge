# src/gridiron_edge/api/routes/prop_shop.py
"""Routes for per-prop multi-book shopping.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at multi-book odds ingest. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.prop_shop import PropShop

router = APIRouter(prefix="/props", tags=["prop-shop"])


@router.get(
    "/{prop_id}/shop",
    response_model=PropShop,
    summary="Cross-book line and price comparison for a single prop.",
)
def get_prop_shop(prop_id: str) -> PropShop:
    """Return null cross-book lines until multi-book odds ingest lands."""
    meta = ResponseMeta()
    for field in ("books", "best_over", "best_under"):
        meta = meta.with_blocked(field, *Blocker.MULTI_BOOK)
    # pyrefly: ignore [unexpected-keyword]
    return PropShop(prop_id=prop_id, response_meta=meta)
