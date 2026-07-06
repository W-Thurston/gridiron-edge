# src/gridiron_edge/api/routes/teams.py

"""Team ranking and profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pandas import DataFrame

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    compute_elo_deltas,
    format_team_cohort_splits,
    load_elo_state_df,
    load_games_df,
    load_team_cohort_splits_df,
    load_team_name_map,
    load_team_percentiles_df,
    resolve_current_season_week,
    team_metadata_lookup,
)
from gridiron_edge.api.schemas.teams import TeamProfile, TeamRankingsList
from gridiron_edge.api.serializers.teams import (
    serialize_team_profile,
    serialize_team_rankings,
)

router = APIRouter(prefix="/teams", tags=["teams"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
) -> tuple[str, int]:
    """Return (season, as_of_week) for the request, defaulting to current."""
    if season is None:
        return resolve_current_season_week(settings)

    games: DataFrame = load_games_df(settings)
    season_games = games.loc[games["YEAR"] == season, "WEEK_NUM"]
    as_of_week: int = int(season_games.max()) if not season_games.empty else 0
    return (season, as_of_week)


@router.get("", response_model=TeamRankingsList)
def list_teams(
    settings: SettingsDep,
    season: str | None = Query(
        default=None,
        description="Season to rank against, e.g. '2025-2026'. Defaults to current.",
    ),
) -> TeamRankingsList:
    """Return power rankings for all teams in the given season."""
    elo: DataFrame = load_elo_state_df(settings)
    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)
    percentiles: DataFrame = load_team_percentiles_df(settings)
    trends: DataFrame = compute_elo_deltas(elo, long_to_short)
    team_metadata = team_metadata_lookup(settings)
    resolved_season, as_of_week = _resolve_scope(settings, season)

    return serialize_team_rankings(
        elo,
        games,
        long_to_short,
        resolved_season,
        as_of_week,
        percentiles,
        trends,
        team_metadata,
    )


@router.get("/{abbr}", response_model=TeamProfile)
def get_team(
    settings: SettingsDep,
    abbr: str,
    season: str | None = Query(
        default=None,
        description="Season to profile against, e.g. '2025-2026'. Defaults to current.",
    ),
) -> TeamProfile:
    """Return per-team profile with ratings, record, and history."""
    long_to_short: dict[str, str] = load_team_name_map(settings)
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}

    if abbr.upper() not in short_to_long:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team abbreviation: {abbr}",
        )

    elo: DataFrame = load_elo_state_df(settings)
    games: DataFrame = load_games_df(settings)
    percentiles: DataFrame = load_team_percentiles_df(settings)
    trends: DataFrame = compute_elo_deltas(elo, long_to_short)
    cohort_splits_df: DataFrame = load_team_cohort_splits_df(settings)
    cohort_splits = format_team_cohort_splits(cohort_splits_df, abbr.upper())
    team_metadata = team_metadata_lookup(settings)
    resolved_season, as_of_week = _resolve_scope(settings, season)

    return serialize_team_profile(
        abbr,
        elo,
        games,
        long_to_short,
        resolved_season,
        as_of_week,
        percentiles,
        trends,
        team_metadata,
        cohort_splits=cohort_splits,
    )
