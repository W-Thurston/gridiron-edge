# src/gridiron_edge/api/routes/compare.py

"""Team-vs-team and player-vs-defense comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_elo_state_df,
    load_games_df,
    load_team_name_map,
    resolve_current_season_week,
)
from gridiron_edge.api.schemas.compare import CompareTeamsResponse
from gridiron_edge.api.serializers.compare import serialize_compare_teams

router = APIRouter(prefix="/compare", tags=["compare"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
) -> tuple[str, int]:
    """Return (season, as_of_week), defaulting to current when needed.

    Lazy: only reads the games CSV when a default is actually needed.
    """
    if season is not None:
        games = load_games_df(settings)
        season_games = games.loc[games["YEAR"] == season, "WEEK_NUM"]
        as_of_week = int(season_games.max()) if not season_games.empty else 0
        return (season, as_of_week)
    return resolve_current_season_week(settings)


@router.get("/teams", response_model=CompareTeamsResponse)
def compare_teams(
    settings: SettingsDep,
    team_a: str = Query(
        description="Short-code team abbreviation, e.g. 'KC'.",
    ),
    team_b: str = Query(
        description="Short-code team abbreviation, e.g. 'LAC'.",
    ),
    season: str | None = Query(
        default=None,
        description="Season, e.g. '2026-2027'. Defaults to current.",
    ),
) -> CompareTeamsResponse:
    """Return team-vs-team comparison across stats.

    Populated stats: Elo rating, rank, season record.
    Scaffolded stats: off/def decomposition, trend, schedule difficulty,
    playoff probability, cohort splits, per-stat percentile ranks — all
    marked in ``_meta.field_status`` per D14.

    Returns 404 if either abbreviation is unknown.
    """
    long_to_short = load_team_name_map(settings)
    short_to_long = {v: k for k, v in long_to_short.items()}

    if team_a.upper() not in short_to_long:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team abbreviation for team_a: {team_a}",
        )
    if team_b.upper() not in short_to_long:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team abbreviation for team_b: {team_b}",
        )

    elo = load_elo_state_df(settings)
    games = load_games_df(settings)
    resolved_season, as_of_week = _resolve_scope(settings, season)

    return serialize_compare_teams(
        elo,
        games,
        long_to_short,
        team_a_short=team_a,
        team_b_short=team_b,
        season=resolved_season,
        as_of_week=as_of_week,
    )
