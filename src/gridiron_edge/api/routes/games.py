# src/gridiron_edge/api/routes/games.py

"""Game list and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_game,
    load_games_for_week,
    resolve_current_season_week,
)
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.games import GameDetail, GameList
from gridiron_edge.api.serializers.games import (
    serialize_game_detail,
    serialize_games_list,
)
from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

router = APIRouter(prefix="/games", tags=["games"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
    week: int | None,
) -> tuple[str, int]:
    """Return (season, week), defaulting to current when not provided."""
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
    """Return champion-model predictions for all games in (season, week).

    When the champion manifest is missing, returns an empty items list
    with _meta.field_status["items"] marked NO_CHAMPION_MANIFEST.
    """
    resolved_season, resolved_week = _resolve_scope(settings, season, week)

    try:
        rows = load_games_for_week(
            settings,
            season=resolved_season,
            week=resolved_week,
        )
    except ChampionNotFoundError:
        meta = ResponseMeta().with_blocked(
            "items",
            *Unavailable.NO_CHAMPION_MANIFEST,
        )
        return GameList(
            season=resolved_season,
            week=resolved_week,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

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
    """Return champion-model prediction and metadata for one game.

    Returns 404 if the game_id is not in the prediction archive.
    Returns 200 with the champion-dependent fields null when the
    champion manifest is missing (structured field_status per D14).
    """
    try:
        row = load_game(settings, game_id=game_id)
    except ChampionNotFoundError:
        meta = ResponseMeta()
        meta = meta.with_blocked("prediction", *Unavailable.NO_CHAMPION_MANIFEST)
        return GameDetail(
            game_id=game_id,
            away_team="",
            home_team="",
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown game_id: {game_id}",
        )

    return serialize_game_detail(row)
