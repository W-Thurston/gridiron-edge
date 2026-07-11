# src/gridiron_edge/api/routes/players.py

"""Player endpoints — per-game stat history for charts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_player_history,
    load_players_list,
    player_stat_columns,
)
from gridiron_edge.api.schemas.players import (
    PlayerHistoryResponse,
    PlayersListResponse,
)
from gridiron_edge.api.serializers.players import (
    serialize_player_history,
    serialize_players_list,
)

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=PlayersListResponse)
def list_players(
    settings: SettingsDep,
    season: int | None = Query(
        default=None,
        description="Season int (e.g. 2025). Defaults to latest.",
    ),
) -> PlayersListResponse:
    """Return skill players active in a season (for the Compare picker)."""
    payload = load_players_list(settings, season=season)
    if payload is None:
        return PlayersListResponse(items=[], total=0)
    return serialize_players_list(payload)


@router.get("/{player_id}/history", response_model=PlayerHistoryResponse)
def get_player_history(
    settings: SettingsDep,
    player_id: str,
    stat: str = Query(
        description="Stat key, e.g. 'rush_yards', 'rec_yards', 'pass_yards'.",
    ),
    season: int | None = Query(
        default=None,
        description="Season int (e.g. 2024). Defaults to player's latest.",
    ),
    limit: int | None = Query(
        default=None,
        description="Return only the last N games (most recent weeks).",
    ),
) -> PlayerHistoryResponse:
    """Return a player's per-game stat series for one season.

    Powers per-game bar charts (Compare Player-vs-Defense), the
    PlayerProp 12-game history chart, and the PlayersExplorer L6
    sparkline.

    - Unknown stat key: 404 (lists valid keys).
    - Player not found / no rows for season: 404.
    """
    valid = player_stat_columns()
    if stat not in valid:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown stat '{stat}'. Valid stats: {', '.join(valid)}.",
        )

    payload = load_player_history(
        settings,
        player_id=player_id,
        stat=stat,
        season=season,
        limit=limit,
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No history for player '{player_id}' stat '{stat}'.",
        )

    return serialize_player_history(payload)
