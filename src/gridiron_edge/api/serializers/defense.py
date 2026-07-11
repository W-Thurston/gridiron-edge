# src/gridiron_edge/api/serializers/defense.py

"""Serializer for /defense/{team}/allowed."""

from __future__ import annotations

from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.defense import DefenseAllowedResponse


def serialize_defense_allowed(
    *,
    team: str,
    position: str,
    stat_type: str,
    cohorts: dict[str, dict],
) -> DefenseAllowedResponse:
    """Build the defense-allowed response.

    Empty cohorts → cohorts null + field_status marking the blocker
    (opponent_allowed_by_position), consistent with the compare-player
    defense rows.
    """
    if not cohorts:
        meta = ResponseMeta().with_blocked(
            "cohorts",
            *Unavailable.OPPONENT_ALLOWED_BY_POSITION,
        )
        return DefenseAllowedResponse(
            team=team,
            position=position,
            stat_type=stat_type,
            cohorts=None,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    return DefenseAllowedResponse(
        team=team,
        position=position,
        stat_type=stat_type,
        cohorts=cohorts,
    )
