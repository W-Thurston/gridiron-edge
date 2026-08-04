# src/gridiron_edge/api/serializers/games.py
"""Serializers for the selected, schedule-complete weekly game product."""

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
    ProjectedScoreBlock,
    SpreadPredictionBlock,
    TotalPredictionBlock,
    WinPredictionBlock,
)


def _none_if_nan(value: Any) -> Any:  # noqa: ANN401
    """Return None for pandas-null values and preserve concrete values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _text(row: dict, key: str) -> str | None:
    """Return one nullable persisted value as text."""
    value = _none_if_nan(row.get(key))
    return None if value is None else str(value)


def _derive_day_of_week(game_date: str | float | None) -> str | None:
    """Parse an ISO date string and return the weekday name."""
    normalized = _none_if_nan(game_date)
    if not normalized:
        return None
    try:
        return date.fromisoformat(str(normalized)).strftime("%A")
    except (ValueError, TypeError):
        return None


def _build_win_block(row: dict) -> WinPredictionBlock:
    return WinPredictionBlock(
        status=str(row["win_status"]),
        selection_status=_text(row, "win_selection_status"),
        away_win_prob=_none_if_nan(row.get("away_win_prob")),
        home_win_prob=_none_if_nan(row.get("home_win_prob")),
        model_name=_text(row, "win_model_name"),
        model_type=_text(row, "win_model_type"),
        event_id=_text(row, "win_event_id"),
        run_id=_text(row, "win_run_id"),
        generated_at=_text(row, "win_generated_at"),
        role=_text(row, "win_role"),
    )


def _build_spread_block(row: dict) -> SpreadPredictionBlock:
    return SpreadPredictionBlock(
        status=str(row["spread_status"]),
        model_spread=_none_if_nan(row.get("model_spread")),
        uncertainty=_none_if_nan(row.get("spread_uncertainty")),
        source_event_id=_text(row, "spread_source_event_id"),
        model_name=_text(row, "spread_model_name"),
        model_type=_text(row, "spread_model_type"),
        calibration_key=_text(row, "spread_calibration_key"),
        calibration_updated_at=_text(row, "spread_calibration_updated_at"),
    )


def _build_total_block(row: dict) -> TotalPredictionBlock:
    return TotalPredictionBlock(
        status=str(row["total_status"]),
        selection_status=_text(row, "total_selection_status"),
        model_total=_none_if_nan(row.get("model_total")),
        uncertainty=_none_if_nan(row.get("total_uncertainty")),
        model_name=_text(row, "total_model_name"),
        model_type=_text(row, "total_model_type"),
        event_id=_text(row, "total_event_id"),
        run_id=_text(row, "total_run_id"),
        generated_at=_text(row, "total_generated_at"),
        role=_text(row, "total_role"),
        uncertainty_trained_at=_text(row, "total_uncertainty_trained_at"),
    )


def _build_projected_score_block(row: dict) -> ProjectedScoreBlock:
    return ProjectedScoreBlock(
        status=str(row["projected_score_status"]),
        home=_none_if_nan(row.get("projected_home_score")),
        away=_none_if_nan(row.get("projected_away_score")),
    )


def serialize_game_summary(row: dict) -> GameSummary:
    """Convert one persisted weekly-product row to a list item."""
    return GameSummary(
        game_id=str(row["game_id"]),
        game_date=_text(row, "game_date"),
        week=_none_if_nan(row.get("week")),
        season=_text(row, "season"),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        win=_build_win_block(row),
        spread=_build_spread_block(row),
        total=_build_total_block(row),
        projected_score=_build_projected_score_block(row),
    )


def serialize_games_list(
    rows: DataFrame,
    *,
    season: str | None,
    week: int | None,
) -> GameList:
    """Serialize every selected scheduled row without prediction filtering."""
    items = [serialize_game_summary(row.to_dict()) for _, row in rows.iterrows()]
    return GameList(items=items, total=len(items), season=season, week=week)


def serialize_game_detail(
    row: dict,
    team_comparison: dict[str, dict] | None = None,
) -> GameDetail:
    """Build detail from one persisted selected-product row."""
    game_date = _text(row, "game_date")
    meta = ResponseMeta()
    if _none_if_nan(row.get("game_time")) is None:
        meta = meta.with_pending("kick")
    if _none_if_nan(row.get("stadium")) is None:
        meta = meta.with_pending("venue")
    meta = meta.with_pending("weather")
    if team_comparison is None:
        meta = meta.with_pending("team_comparison")
    meta = meta.with_pending("top_prop_edges")
    meta = meta.with_blocked("swing_factors", *Blocker.FEATURE_ATTRIBUTION)
    meta = meta.with_blocked("injuries", *Blocker.INJURY_DATA)

    return GameDetail(
        game_id=str(row["game_id"]),
        game_date=game_date,
        week=_none_if_nan(row.get("week")),
        season=_text(row, "season"),
        day_of_week=_text(row, "game_day_of_week") or _derive_day_of_week(game_date),
        kick=_text(row, "game_time"),
        venue=_text(row, "stadium"),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        win=_build_win_block(row),
        spread=_build_spread_block(row),
        total=_build_total_block(row),
        projected_score=_build_projected_score_block(row),
        weather=None,
        team_comparison=team_comparison,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
