# src/gridiron_edge/api/serializers/explain.py

"""Serializer for GET /games/{game_id}/explain.

Owns `_meta.field_status` construction per D18. `factors` serializes a
persisted `LogisticExplanationEvent` when one exists for the game's selected
Win forecast event; otherwise it is null with the field_status entry that
explains why (win prediction unavailable, non-Logistic champion, or no
explanation batch generated yet for this run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from gridiron_edge.api.meta import Blocker, ResponseMeta, Unavailable
from gridiron_edge.api.schemas.explain import ExplainFactor, GameExplain

if TYPE_CHECKING:
    from gridiron_edge.evaluation.logistic_explanation_evidence import (
        LogisticExplanationEvent,
    )

_LOGISTIC_MODEL_TYPE = "logistic"
_WIN_AVAILABLE_STATUS = "available"


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _build_factors(event: LogisticExplanationEvent) -> list[ExplainFactor]:
    """Build the factor list from one persisted explanation event.

    The intercept is its own leading factor (`is_baseline=True`); the
    remaining factors follow the event's own persisted contribution order.
    """
    factors = [
        ExplainFactor(
            key="intercept",
            label="intercept",
            log_odds_contribution=event.intercept,
            coefficient=None,
            transformed_value=None,
            is_baseline=True,
            is_adjustable=False,
        )
    ]
    factors.extend(
        ExplainFactor(
            key=contribution.feature_name,
            label=contribution.feature_name,
            log_odds_contribution=contribution.contribution,
            coefficient=contribution.coefficient,
            transformed_value=contribution.transformed_value,
            is_baseline=False,
            is_adjustable=False,
        )
        for contribution in event.contributions
    )
    return factors


def serialize_game_explain(
    row: dict,
    explanation_event: LogisticExplanationEvent | None,
    *,
    ambiguous_evidence: bool = False,
) -> GameExplain:
    """Build the /explain response from one selected weekly-product row.

    `explanation_event` must already be resolved by the caller (looked up by
    the row's own `win_event_id` when `win_model_type == "logistic"`) since
    this serializer performs no lookups of its own. `ambiguous_evidence` is
    set when the caller's lookup found more than one persisted batch
    claiming the same event — the immutable store is create-only with no
    "current" selection concept, so this can genuinely happen and is a
    distinct condition from no evidence existing at all.
    """
    game_id = str(row["game_id"])
    win_status = str(row["win_status"])
    win_model_type = row.get("win_model_type")

    meta = ResponseMeta()
    meta = meta.with_blocked("band", *Blocker.SCENARIO_ENGINE)
    meta = meta.with_blocked("distribution", *Blocker.SCENARIO_ENGINE)
    meta = meta.with_blocked("market_implied", *Blocker.SCENARIO_ENGINE)

    if win_status != _WIN_AVAILABLE_STATUS:
        meta = meta.with_blocked("headline_win_prob", *Unavailable.NO_WIN_FORECAST)
        meta = meta.with_blocked("factors", *Unavailable.NO_WIN_FORECAST)
        return GameExplain(
            game_id=game_id,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    headline_win_prob = _none_if_nan(row.get("home_win_prob"))
    factors: list[ExplainFactor] | None = None

    if win_model_type != _LOGISTIC_MODEL_TYPE:
        meta = meta.with_blocked("factors", *Blocker.FEATURE_ATTRIBUTION)
    elif ambiguous_evidence:
        meta = meta.with_blocked("factors", *Unavailable.AMBIGUOUS_EXPLANATION_EVIDENCE)
    elif explanation_event is None:
        meta = meta.with_blocked("factors", *Unavailable.NO_EXPLANATION_EVIDENCE)
    else:
        factors = _build_factors(explanation_event)

    return GameExplain(
        game_id=game_id,
        headline_win_prob=headline_win_prob,
        factors=factors,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
