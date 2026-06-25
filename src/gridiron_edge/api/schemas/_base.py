# src/gridiron_edge/api/schemas/_base.py
"""Base response models with the `_meta` envelope.

All API response schemas inherit from `BaseResponse` (single object) or
`BaseListResponse` (list payload). The envelope is optional but always
typed; per D14, unpopulated fields return `null` and their status lives
in `_meta.field_status`.

Per D16, list endpoints surface blocked-list state through the same
`field_status` mechanism keyed on `"items"`, not via a separate envelope
field. This module provides only the shapes; routes and concrete schemas
build on top.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.meta import ResponseMeta


class BaseResponse(BaseModel):
    """Base for single-object API response schemas.

    Subclasses add their own typed fields. The `_meta` envelope is
    optional; when present it documents which fields are unpopulated
    and why (per D14).

    Configured as:
    - `frozen=True`: responses are immutable once constructed; safe to
      share and cache.
    - `extra="forbid"`: unknown fields raise `ValidationError` at
      construction. Surfaces typos as test failures rather than silent
      data loss.
    - `populate_by_name=True`: construction code can use either
      `response_meta=...` or `_meta=...`; the wire shape is always
      `_meta`.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    response_meta: ResponseMeta | None = Field(
        default=None,
        alias="_meta",
        description="Optional metadata describing unpopulated fields (per D14).",
    )


T = TypeVar("T", bound=BaseModel)


class BaseListResponse[T: BaseModel](BaseResponse):
    """Base for list-shaped API responses.

    Wraps the list payload in an `items` field so the envelope has a
    consistent attach point. `total` is optional and signals when the
    response represents a subset (paginated or filtered) of a larger
    collection.

    Per D16, blocked-list endpoints mark `_meta.field_status["items"]`
    rather than introducing a separate `list_status` envelope field.

    Example:
        >>> BaseListResponse[GameSummary](items=[...], total=13)
        >>> BaseListResponse[LineRow](  # blocked
        ...     items=[],
        ...     total=0,
        ...     response_meta=ResponseMeta().with_blocked(
        ...         "items",
        ...         *Blocker.MULTI_BOOK,
        ...     ),
        ... )
    """

    items: list[T] = Field(
        default_factory=list,
        description="List payload. May be empty when blocked (see `_meta`).",
    )
    total: int | None = Field(
        default=None,
        description=(
            "Total count when applicable. None implies the response contains "
            "all items; an integer implies a subset of a larger collection."
        ),
    )
