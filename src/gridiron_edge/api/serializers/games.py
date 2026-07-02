# src/gridiron_edge/api/serializers/games.py

"""Serializers for /games and /games/{game_id}.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.games import (
    GameDetail,
    GameList,
    GameSummary,
    PredictionBlock,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _derive_day_of_week(game_date: str | float | None) -> str | None:
    """Parse an ISO date string and return the weekday name.

    Returns None for missing, NaN, or unparseable input. Serializer-side
    derivation (not loader-side) keeps display concerns out of the domain
    loader.
    """
    game_date = _none_if_nan(game_date)
    if not game_date:
        return None
    try:
        return date.fromisoformat(str(game_date)).strftime("%A")
    except (ValueError, TypeError):
        return None


def _build_prediction_block(row: dict) -> PredictionBlock | None:
    """Extract champion-prediction fields from a loader row.

    Returns None if every prediction field is None/NaN (the loader
    returned a game with no champion data — should not normally happen
    once the loader has resolved the champion, but defensive).
    """
    fields: dict[str, Any] = {
        "home_win_prob": _none_if_nan(row.get("home_win_prob")),
        "away_win_prob": _none_if_nan(row.get("away_win_prob")),
        "home_win_lo": _none_if_nan(row.get("win_prob_lo")),
        "home_win_hi": _none_if_nan(row.get("win_prob_hi")),
        "confidence_tier": _none_if_nan(row.get("confidence_tier")),
        "model_spread": _none_if_nan(row.get("model_spread")),
        "model_total": _none_if_nan(row.get("model_total")),
        "projected_home_score": _none_if_nan(row.get("projected_home_score")),
        "projected_away_score": _none_if_nan(row.get("projected_away_score")),
    }
    if all(v is None for v in fields.values()):
        return None
    return PredictionBlock(**fields)


def serialize_game_summary(row: dict) -> GameSummary:
    """Convert one loader row (Series-as-dict) to GameSummary."""
    prediction: PredictionBlock | None = _build_prediction_block(row)
    return GameSummary(
        game_id=str(row["game_id"]),
        game_date=_none_if_nan(row.get("game_date")),
        week=_none_if_nan(row.get("week")),
        season=_none_if_nan(row.get("season")),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        prediction=prediction,
    )


def serialize_games_list(
    rows: DataFrame,
    *,
    season: str | None,
    week: int | None,
) -> GameList:
    """Build the /games list response from a loader DataFrame.

    Empty rows return an empty items list; per D14, no field_status is
    attached because the concrete-empty case is legitimate (no games
    yet scheduled, or filter matched nothing).
    """
    items: list[GameSummary] = [serialize_game_summary(r.to_dict()) for _, r in rows.iterrows()]
    return GameList(
        items=items,
        total=len(items),
        season=season,
        week=week,
    )


def serialize_game_detail(row: dict) -> GameDetail:
    """Build the /games/{game_id} response.

    Populated fields come from the loader row. Pending/blocked fields
    are marked in _meta.field_status per D14. See PLAN.md substep 5c
    for the field-scope rationale.
    """
    prediction: PredictionBlock | None = _build_prediction_block(row)
    game_date = _none_if_nan(row.get("game_date"))
    day_of_week: str | None = _derive_day_of_week(game_date)

    meta = ResponseMeta()
    # Backend work planned: schedule join for kick/venue, weather join.
    meta: ResponseMeta = meta.with_pending("kick")
    meta = meta.with_pending("venue")
    meta = meta.with_pending("weather")
    # Tier 3 additive datasets (per PLAN.md W8 Tier 3).
    meta = meta.with_pending("team_comparison")
    meta = meta.with_pending("top_prop_edges")
    # Blocked on upstream workstreams.
    meta = meta.with_blocked("swing_factors", *Blocker.FEATURE_ATTRIBUTION)
    meta = meta.with_blocked("injuries", *Blocker.INJURY_DATA)

    return GameDetail(
        game_id=str(row["game_id"]),
        game_date=game_date,
        week=_none_if_nan(row.get("week")),
        season=_none_if_nan(row.get("season")),
        day_of_week=day_of_week,
        kick=None,
        venue=None,
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        weather=None,
        prediction=prediction,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
