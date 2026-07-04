# src/gridiron_edge/api/routes/compare.py

"""Team-vs-team and player-vs-defense comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pandas import DataFrame

from gridiron_edge.api._prop_id import decode_prop_id, resolve_opponent_from_game_id
from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    format_team_cohort_splits,
    load_elo_state_df,
    load_games_df,
    load_opponent_allowed_for_prop,
    load_prop,
    load_team_cohort_splits_df,
    load_team_name_map,
    load_team_percentiles_df,
    resolve_current_season_week,
)
from gridiron_edge.api.schemas.compare import (
    ComparePlayerResponse,
    CompareTeamsResponse,
)
from gridiron_edge.api.serializers.compare import (
    serialize_compare_player,
    serialize_compare_teams,
)
from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

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
    long_to_short: dict[str, str] = load_team_name_map(settings)
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}

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

    elo: DataFrame = load_elo_state_df(settings)
    games: DataFrame = load_games_df(settings)
    percentiles: DataFrame = load_team_percentiles_df(settings)
    cohort_splits_df = load_team_cohort_splits_df(settings)

    # Build cohort_splits dict for both teams.
    cohort_splits: dict[str, dict] | None = None
    a_splits = format_team_cohort_splits(cohort_splits_df, team_a.upper())
    b_splits = format_team_cohort_splits(cohort_splits_df, team_b.upper())
    if a_splits is not None or b_splits is not None:
        cohort_splits = {}
        if a_splits is not None:
            cohort_splits[team_a.upper()] = a_splits
        if b_splits is not None:
            cohort_splits[team_b.upper()] = b_splits

    resolved_season, as_of_week = _resolve_scope(settings, season)

    return serialize_compare_teams(
        elo,
        games,
        long_to_short,
        team_a_short=team_a,
        team_b_short=team_b,
        season=resolved_season,
        as_of_week=as_of_week,
        percentiles=percentiles,
        cohort_splits=cohort_splits,
    )


@router.get("/player/{prop_id}", response_model=ComparePlayerResponse)
def compare_player(
    settings: SettingsDep,
    prop_id: str,
) -> ComparePlayerResponse:
    """Return projection-vs-defense comparison for one prop.

    Projection-side fields (mean, std, lo_90, hi_90) populate from the
    champion model's archive row. Defense-side fields (avg allowed,
    rank vs position, L5 avg, red zone rate) are entirely blocked
    pending opponent-allowed-by-position aggregation (ROADMAP §9 Tier 3).

    - Malformed or unknown prop_id: 404.
    - Champion for this stat_type not resolved: 200 with projection
      block null and field_status marking blocked.
    - Prop not in archive: 404.
    """
    game_id, player_id, stat_type = decode_prop_id(prop_id)

    try:
        row = load_prop(
            settings,
            game_id=game_id,
            player_id=player_id,
            stat_type=stat_type,
        )
    except ChampionNotFoundError:
        # Fabricate a minimal row: projection fields null, identity
        # fields from the decoded prop_id. Serializer marks blocked.
        row = {
            "game_id": game_id,
            "player_id": player_id,
            "player_name": "",
            "position": "",
            "team": "",
            "stat_type": stat_type,
            "model_name": stat_type,
            "model_type": "",
            "season": None,
            "week": None,
            "predicted_mean": None,
            "predicted_std": None,
            "lo_90": None,
            "hi_90": None,
        }
        return serialize_compare_player(row)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prop not found: {prop_id}",
        )

    # Determine opponent from game_id and player's team.
    opponent = resolve_opponent_from_game_id(
        str(row["game_id"]),
        str(row["team"]),
    )

    opponent_allowed: dict[str, dict] | None = None
    if opponent is not None:
        opponent_allowed = load_opponent_allowed_for_prop(
            settings,
            opponent_team=opponent,
            position=str(row["position"]),
            stat_type=stat_type,
        )

    return serialize_compare_player(row, opponent_allowed=opponent_allowed)
