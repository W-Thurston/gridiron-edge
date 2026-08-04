# src/gridiron_edge/api/routes/games.py
"""Schedule-complete game list and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pandas import DataFrame

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    format_team_cohort_splits,
    load_game,
    load_games_for_week,
    load_team_cohort_splits_df,
    resolve_current_season_week,
)
from gridiron_edge.api.schemas.games import GameDetail, GameList
from gridiron_edge.api.serializers.games import (
    serialize_game_detail,
    serialize_games_list,
)

router = APIRouter(prefix="/games", tags=["games"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
    week: int | None,
) -> tuple[str, int]:
    """Return season and week, defaulting to the current API scope."""
    resolved_season, resolved_week = resolve_current_season_week(settings)
    return (season or resolved_season, week or resolved_week)


@router.get("", response_model=GameList)
def list_games(
    settings: SettingsDep,
    season: str | None = Query(
        default=None,
        description="Season, e.g. '2026-2027'. Defaults to current.",
    ),
    week: int | None = Query(
        default=None,
        description="Week number. Defaults to current.",
    ),
) -> GameList:
    """Return every row in the explicitly selected weekly product."""
    resolved_season, resolved_week = _resolve_scope(settings, season, week)
    try:
        rows = load_games_for_week(
            settings,
            season=resolved_season,
            week=resolved_week,
        )
    except FileNotFoundError:
        rows = DataFrame()

    return serialize_games_list(
        rows,
        season=resolved_season,
        week=resolved_week,
    )


@router.get("/{game_id}", response_model=GameDetail)
def get_game(
    settings: SettingsDep,
    game_id: str,
) -> GameDetail:
    """Return one selected scheduled game regardless of prediction status."""
    row = load_game(settings, game_id=game_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown game_id: {game_id}",
        )

    cohort_splits_df = load_team_cohort_splits_df(settings)
    team_comparison: dict[str, dict] | None = None
    away_team = str(row["away_team"])
    home_team = str(row["home_team"])
    away_splits = format_team_cohort_splits(cohort_splits_df, away_team)
    home_splits = format_team_cohort_splits(cohort_splits_df, home_team)
    if away_splits is not None or home_splits is not None:
        team_comparison = {}
        if away_splits is not None:
            team_comparison[away_team] = away_splits
        if home_splits is not None:
            team_comparison[home_team] = home_splits

    return serialize_game_detail(row, team_comparison=team_comparison)
