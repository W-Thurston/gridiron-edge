# src/gridiron_edge/api/serializers/comparables.py

"""Serializer for GET /games/{game_id}/comparables.

Owns `_meta.field_status` construction per D18. `comparables`,
`sample_size`, `favorite_win_rate`, and `favorite_cover_rate` serialize a
persisted `ComparableGamesBatch` when one exists for the game's selected
Win forecast event; otherwise all four are null with the field_status entry
that explains why (win prediction unavailable, no corpus for this
champion's model type yet, no batch generated yet for this run, or ambiguous
evidence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridiron_edge.api.meta import Blocker, ResponseMeta, Unavailable
from gridiron_edge.api.schemas.comparables import ComparableFactor, ComparableGame, GameComparables

if TYPE_CHECKING:
    from gridiron_edge.evaluation.comparable_games_evidence import ComparableGamesBatch

_LOGISTIC_MODEL_TYPE = "logistic"
_WIN_AVAILABLE_STATUS = "available"

_BLOCKED_FIELDS = ("comparables", "sample_size", "favorite_win_rate", "favorite_cover_rate")


def _build_comparables(batch: ComparableGamesBatch) -> list[ComparableGame]:
    """Build the comparable-game list from one persisted retrieval batch."""
    return [
        ComparableGame(
            game_id=match.game_id,
            rank=match.rank,
            distance=match.distance,
            season=match.season,
            week=match.week,
            game_date=match.game_date,
            away_team=match.away_team,
            home_team=match.home_team,
            away_score=match.away_score,
            home_score=match.home_score,
            favorite_team=match.favorite_team,
            spread_magnitude=match.spread_magnitude,
            favorite_won=match.favorite_won,
            favorite_covered=match.favorite_covered,
            top_contributing_features=[
                ComparableFactor(
                    feature_name=item.feature_name,
                    query_value=item.query_value,
                    candidate_value=item.candidate_value,
                    squared_difference=item.squared_difference,
                )
                for item in match.top_contributing_features
            ],
        )
        for match in batch.matches
    ]


def serialize_game_comparables(
    row: dict,
    batch: ComparableGamesBatch | None,
    *,
    ambiguous_evidence: bool = False,
) -> GameComparables:
    """Build the /comparables response from one selected weekly-product row.

    `batch` must already be resolved by the caller (looked up by the row's
    own `win_event_id` when `win_model_type == "logistic"`) since this
    serializer performs no lookups of its own. `ambiguous_evidence` is set
    when the caller's lookup found more than one persisted batch claiming
    the same event — for example the event was retrieved against two
    different corpus generations (before and after a champion retrain) —
    which is distinct from no evidence existing at all.
    """
    game_id = str(row["game_id"])
    win_status = str(row["win_status"])
    win_model_type = row.get("win_model_type")

    meta = ResponseMeta()

    if win_status != _WIN_AVAILABLE_STATUS:
        for field in _BLOCKED_FIELDS:
            meta = meta.with_blocked(field, *Unavailable.NO_WIN_FORECAST)
        return GameComparables(
            game_id=game_id,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if win_model_type != _LOGISTIC_MODEL_TYPE:
        for field in _BLOCKED_FIELDS:
            meta = meta.with_blocked(field, *Blocker.COMPARABLES)
        return GameComparables(
            game_id=game_id,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if ambiguous_evidence:
        for field in _BLOCKED_FIELDS:
            meta = meta.with_blocked(field, *Unavailable.AMBIGUOUS_COMPARABLE_EVIDENCE)
        return GameComparables(
            game_id=game_id,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if batch is None:
        for field in _BLOCKED_FIELDS:
            meta = meta.with_blocked(field, *Unavailable.NO_COMPARABLE_EVIDENCE)
        return GameComparables(
            game_id=game_id,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    return GameComparables(
        game_id=game_id,
        comparables=_build_comparables(batch),
        sample_size=batch.sample_size,
        favorite_win_rate=batch.favorite_win_rate,
        favorite_cover_rate=batch.favorite_cover_rate,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
