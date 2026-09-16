# src/gridiron_edge/api/routes/teams.py

"""Team ranking and profile endpoints."""

from __future__ import annotations

from typing import Literal

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
    load_weekly_elo_forecast,
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
    *,
    elo: DataFrame,
) -> tuple[str, int]:
    """Return (season, as_of_week) for the request, defaulting to current.

    Explicit preseason scopes fall back to the requested season's Elo state
    when no completed games exist. A truly unavailable season remains Week 0.
    """
    if season is None:
        return resolve_current_season_week(settings)

    games: DataFrame = load_games_df(settings)
    season_games = games.loc[games["YEAR"] == season, "WEEK_NUM"]
    if not season_games.empty:
        return (season, int(season_games.max()))

    season_elo = elo.loc[elo["NFL_YEAR"] == season, "NFL_WEEK"]
    as_of_week: int = int(season_elo.max()) if not season_elo.empty else 0
    return (season, as_of_week)


def _resolve_rating_timeline_scope(
    *,
    elo: DataFrame,
    games: DataFrame,
    season: str,
    as_of_week: int,
) -> tuple[int, int]:
    """Return completed-through and entering-week rating boundaries."""
    season_games = games.loc[games["YEAR"] == season, :]
    completed = season_games.dropna(subset=["AWAY_SCORE", "HOME_SCORE"])
    completed_through = min(22, int(completed["WEEK_NUM"].max())) if not completed.empty else 0
    available = elo.loc[
        (elo["NFL_YEAR"] == season) & (elo["NFL_WEEK"].between(1, 22)),
        "NFL_WEEK",
    ]
    if available.empty:
        return (completed_through, max(1, min(22, as_of_week)))
    desired = min(22, completed_through + 1)
    available_weeks = sorted(int(value) for value in available.unique())
    eligible = [week for week in available_weeks if week <= desired]
    current_rating_week = max(eligible) if eligible else min(available_weeks)
    return (completed_through, current_rating_week)


@router.get("", response_model=TeamRankingsList)
def list_teams(
    settings: SettingsDep,
    season: str = Query(
        default=None,
        description="Season to rank against, e.g. '2025-2026'. Defaults to current.",
    ),
) -> TeamRankingsList:
    """Return power rankings for all teams in the given season."""
    elo: DataFrame = load_elo_state_df(settings)
    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)
    percentiles: DataFrame = load_team_percentiles_df(settings)
    team_metadata = team_metadata_lookup(settings)

    resolved_season, as_of_week = _resolve_scope(
        settings,
        season,
        elo=elo,
    )
    trends: DataFrame = compute_elo_deltas(
        elo,
        long_to_short,
        season=resolved_season,
    )

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
    season: str = Query(
        default=None,
        description="Season to profile against, e.g. '2025-2026'. Defaults to current.",
    ),
    rating_range: Literal["season", "recent"] = Query(
        default="season",
        description="Historical power-rating timeline range.",
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
    cohort_splits_df: DataFrame = load_team_cohort_splits_df(settings)
    cohort_splits = format_team_cohort_splits(
        cohort_splits_df,
        abbr.upper(),
    )
    team_metadata = team_metadata_lookup(settings)

    resolved_season, as_of_week = _resolve_scope(
        settings,
        season,
        elo=elo,
    )
    trends: DataFrame = compute_elo_deltas(
        elo,
        long_to_short,
        season=resolved_season,
    )
    completed_through_week, current_rating_week = _resolve_rating_timeline_scope(
        elo=elo,
        games=games,
        season=resolved_season,
        as_of_week=as_of_week,
    )

    forecast_load = load_weekly_elo_forecast(
        settings,
        season=resolved_season,
        forecast_origin_week=current_rating_week,
        team_abbr=abbr,
    )

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
        completed_through_week=completed_through_week,
        current_rating_week=current_rating_week,
        timeline_range=rating_range,
        forecast_load=forecast_load,
    )
