# src/gridiron_edge/api/routes/news.py
"""Routes for news wire endpoints.

Responses are currently null shapes with structured `_meta.field_status`
entries pointing at the news ingest gap. See ROADMAP §9.5.
"""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.news import NewsItem

router = APIRouter(prefix="/news", tags=["news"])


_LIST_META = ResponseMeta().with_blocked("items", *Blocker.NEWS_INGEST)


@router.get(
    "",
    response_model=BaseListResponse[NewsItem],
    summary="News and lineup feed.",
)
def list_news() -> BaseListResponse[NewsItem]:
    """Return an empty feed until news ingest lands."""
    return BaseListResponse[NewsItem](
        items=[],
        total=0,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=_LIST_META,
    )


@router.get(
    "/alerts",
    response_model=BaseListResponse[NewsItem],
    summary="Alerts relevant to active positions.",
)
def list_alerts() -> BaseListResponse[NewsItem]:
    """Return an empty alerts list until news ingest lands."""
    return BaseListResponse[NewsItem](
        items=[],
        total=0,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=_LIST_META,
    )
