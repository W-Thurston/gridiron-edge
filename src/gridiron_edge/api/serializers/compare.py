# src/gridiron_edge/api/serializers/compare.py

"""Serializers for /compare endpoints.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from pandas import DataFrame

from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.compare import ComparePlayerResponse, CompareTeamsResponse, StatRow
from gridiron_edge.api.serializers.teams import _compute_record, _latest_ratings


def _rating_for_team(
    elo: DataFrame,
    long_name: str,
    season: str,
    as_of_week: int,
) -> tuple[float | None, int | None]:
    """Return (rating, rank) for a team from the latest Elo state.

    Returns (None, None) if the team has no rating for the requested
    season/week.
    """
    latest = _latest_ratings(elo, season, as_of_week)
    if latest.empty:
        return None, None

    ranked = latest.sort_values("ELO", ascending=False).reset_index(drop=True)
    team_row = ranked.loc[ranked["NFL_TEAM"] == long_name]
    if team_row.empty:
        return None, None

    # pyrefly: ignore [bad-argument-type]
    rank = int(team_row.index[0]) + 1
    rating = float(team_row.iloc[0]["ELO"])
    return rating, rank


def _record_string(games: DataFrame, long_name: str) -> str:
    """Format a team's season record as 'W-L-T' string."""
    record = _compute_record(games, long_name)
    return f"{record.wins}-{record.losses}-{record.ties}"


def serialize_compare_teams(
    elo: DataFrame,
    games: DataFrame,
    long_to_short: dict[str, str],
    *,
    team_a_short: str,
    team_b_short: str,
    season: str,
    as_of_week: int,
) -> CompareTeamsResponse:
    """Build the /compare/teams response.

    Populated stats: rating, rank, record.
    Scaffolded stats: off_rating, def_rating, trend, schedule_difficulty,
    playoff_probability, cohort splits, percentile ranks — via field_status.
    """
    short_to_long = {v: k for k, v in long_to_short.items()}
    long_a = short_to_long.get(team_a_short.upper())
    long_b = short_to_long.get(team_b_short.upper())

    # Route should have validated abbreviations before this call. Defensive
    # fallback: use the short code as a placeholder if the map miss.
    long_a = long_a or team_a_short
    long_b = long_b or team_b_short

    season_games = games.loc[games["YEAR"] == season, :]

    rating_a, rank_a = _rating_for_team(elo, long_a, season, as_of_week)
    rating_b, rank_b = _rating_for_team(elo, long_b, season, as_of_week)

    stats: list[StatRow] = [
        StatRow(
            key="rating",
            label="Elo Rating",
            unit="elo",
            team_a_value=rating_a,
            team_b_value=rating_b,
        ),
        StatRow(
            key="rank",
            label="Rank",
            unit="rank",
            team_a_value=rank_a,
            team_b_value=rank_b,
        ),
        StatRow(
            key="record",
            label="Record",
            unit="record",
            team_a_value=_record_string(season_games, long_a),
            team_b_value=_record_string(season_games, long_b),
        ),
        StatRow(
            key="off_rating",
            label="Offensive Rating",
            unit="elo",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="def_rating",
            label="Defensive Rating",
            unit="elo",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="trend",
            label="Rating Trend (7d)",
            unit="raw",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="schedule_difficulty",
            label="Schedule Difficulty",
            unit="raw",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="playoff_probability",
            label="Playoff Probability",
            unit="pct",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="cohort_splits",
            label="Cohort Splits (Season/L4/Home/Away)",
            unit="raw",
            team_a_value=None,
            team_b_value=None,
        ),
        StatRow(
            key="percentile_ranks",
            label="Per-Stat Percentile Ranks",
            unit="pct",
            team_a_value=None,
            team_b_value=None,
        ),
    ]

    meta = ResponseMeta()
    # Blocked on upstream workstreams.
    meta = meta.with_blocked("off_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("def_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("trend", *Unavailable.NO_PRIOR_SNAPSHOT)
    # Pending on Tier 3 additive datasets.
    meta = meta.with_pending("schedule_difficulty")
    meta = meta.with_pending("playoff_probability")
    meta = meta.with_pending("cohort_splits")
    meta = meta.with_pending("percentile_ranks")

    return CompareTeamsResponse(
        season=season,
        team_a=team_a_short.upper(),
        team_b=team_b_short.upper(),
        stats=stats,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def serialize_compare_player(row: dict) -> ComparePlayerResponse:
    """Build the /compare/player/{prop_id} response.

    Populated stats come from the archive row's projection fields.
    Defense-side stats are entirely blocked pending
    opponent-allowed-by-position aggregation (Tier 3 additive dataset).
    """
    from gridiron_edge.api.schemas.compare import (
        ComparePlayerResponse,
        PlayerVsDefenseRow,
    )
    from gridiron_edge.api.serializers.props import (
        _build_model_key,
        _build_prop_id,
        _none_if_nan,
        _season_int_to_str,
    )

    projection_mean = _none_if_nan(row.get("predicted_mean"))
    projection_std = _none_if_nan(row.get("predicted_std"))
    projection_lo = _none_if_nan(row.get("lo_90"))
    projection_hi = _none_if_nan(row.get("hi_90"))

    stats: list[PlayerVsDefenseRow] = [
        PlayerVsDefenseRow(
            key="mean",
            label="Projected Mean",
            unit="yards",
            projection_value=projection_mean,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="std",
            label="Uncertainty (std)",
            unit="yards",
            projection_value=projection_std,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="lo_90",
            label="10th Percentile",
            unit="yards",
            projection_value=projection_lo,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="hi_90",
            label="90th Percentile",
            unit="yards",
            projection_value=projection_hi,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="avg_allowed",
            label="Defense: Avg Allowed",
            unit="yards",
            projection_value=None,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="rank_against_position",
            label="Defense: Rank vs Position",
            unit="rank",
            projection_value=None,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="last_5_games_avg",
            label="Defense: L5 Avg Allowed",
            unit="yards",
            projection_value=None,
            defense_value=None,
        ),
        PlayerVsDefenseRow(
            key="red_zone_rate_allowed",
            label="Defense: Red Zone Rate Allowed",
            unit="pct",
            projection_value=None,
            defense_value=None,
        ),
    ]

    meta = ResponseMeta()
    # Defense-side rows: all blocked on Tier 3 additive dataset.
    meta = meta.with_blocked("avg_allowed", *Unavailable.OPPONENT_ALLOWED_BY_POSITION)
    meta = meta.with_blocked("rank_against_position", *Unavailable.OPPONENT_ALLOWED_BY_POSITION)
    meta = meta.with_blocked("last_5_games_avg", *Unavailable.OPPONENT_ALLOWED_BY_POSITION)
    meta = meta.with_blocked("red_zone_rate_allowed", *Unavailable.OPPONENT_ALLOWED_BY_POSITION)

    return ComparePlayerResponse(
        prop_id=_build_prop_id(row),
        game_id=str(row["game_id"]),
        season=_season_int_to_str(row.get("season")),
        week=_none_if_nan(row.get("week")),
        player_id=str(row["player_id"]),
        player_name=str(row["player_name"]),
        position=str(row["position"]),
        team=str(row["team"]),
        stat_type=str(row["stat_type"]),
        model_key=_build_model_key(row),
        stats=stats,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
