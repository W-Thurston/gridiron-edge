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


def _pct_for_team(
    percentiles: DataFrame,
    team_abbr: str,
    column: str,
) -> float | None:
    """Return a specific percentile value for a team, or None if missing."""
    if percentiles.empty:
        return None
    match = percentiles.loc[percentiles["team_abbr"] == team_abbr]
    if match.empty:
        return None
    val = match.iloc[0].get(column)
    if val is None:
        return None
    try:
        import pandas as pd

        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)


def serialize_compare_teams(
    elo: DataFrame,
    games: DataFrame,
    long_to_short: dict[str, str],
    *,
    team_a_short: str,
    team_b_short: str,
    season: str,
    as_of_week: int,
    percentiles: DataFrame,
    cohort_splits: dict[str, dict] | None = None,
) -> CompareTeamsResponse:
    """Build the /compare/teams response.

    Populated stats: rating, rank, record.
    Scaffolded stats: off_rating, def_rating, trend, schedule_difficulty,
    playoff_probability, cohort splits, percentile ranks — via field_status.
    """
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}
    long_a: str | None = short_to_long.get(team_a_short.upper())
    long_b: str | None = short_to_long.get(team_b_short.upper())

    # Route should have validated abbreviations before this call. Defensive
    # fallback: use the short code as a placeholder if the map miss.
    long_a = long_a or team_a_short
    long_b = long_b or team_b_short

    season_games = games.loc[games["YEAR"] == season, :]

    rating_a, rank_a = _rating_for_team(elo, long_a, season, as_of_week)
    rating_b, rank_b = _rating_for_team(elo, long_b, season, as_of_week)

    # Precompute percentiles for both teams (all four stats).
    a: str = team_a_short.upper()
    b: str = team_b_short.upper()

    stats: list[StatRow] = [
        StatRow(
            key="rating",
            label="Elo Rating",
            unit="elo",
            team_a_value=rating_a,
            team_b_value=rating_b,
            team_a_pct=_pct_for_team(percentiles, a, "rating_pct"),
            team_b_pct=_pct_for_team(percentiles, b, "rating_pct"),
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
            key="avg_wins",
            label="Projected Avg Wins",
            unit="raw",
            team_a_value=None,
            team_b_value=None,
            team_a_pct=_pct_for_team(percentiles, a, "avg_wins_pct"),
            team_b_pct=_pct_for_team(percentiles, b, "avg_wins_pct"),
        ),
        StatRow(
            key="make_playoffs",
            label="Playoff Probability",
            unit="pct",
            team_a_value=None,
            team_b_value=None,
            team_a_pct=_pct_for_team(percentiles, a, "make_playoffs_pct"),
            team_b_pct=_pct_for_team(percentiles, b, "make_playoffs_pct"),
        ),
        StatRow(
            key="win_sb",
            label="Super Bowl Win Probability",
            unit="pct",
            team_a_value=None,
            team_b_value=None,
            team_a_pct=_pct_for_team(percentiles, a, "win_sb_pct"),
            team_b_pct=_pct_for_team(percentiles, b, "win_sb_pct"),
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
            key="cohort_splits",
            label="Cohort Splits (Season/L4/Home/Away)",
            unit="raw",
            team_a_value=None,
            team_b_value=None,
        ),
    ]

    meta: ResponseMeta = ResponseMeta()
    # Blocked on upstream workstreams.
    meta = meta.with_blocked("off_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("def_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("trend", *Unavailable.NO_PRIOR_SNAPSHOT)
    # Pending required derived datasets.
    meta = meta.with_pending("schedule_difficulty")
    if cohort_splits is None:
        meta = meta.with_pending("cohort_splits")
    # Note: avg_wins, make_playoffs, win_sb now have percentiles populated
    # via team_a_pct / team_b_pct, so they're no longer marked pending here.
    # The team_*_value fields remain null (raw values not shown on compare).
    # percentile_ranks scaffold row removed — per-row percentiles replace it.

    return CompareTeamsResponse(
        season=season,
        team_a=team_a_short.upper(),
        team_b=team_b_short.upper(),
        stats=stats,
        cohort_splits=cohort_splits,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def serialize_compare_player(
    row: dict,
    *,
    opponent_allowed: dict[str, dict] | None = None,
) -> ComparePlayerResponse:
    """Build the /compare/player/{prop_id} response.

    Populated stats come from the archive row's projection fields.
    Defense-side stats are entirely blocked pending
    opponent-allowed-by-position aggregation.
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
            defense_value=_get_defense_stat(opponent_allowed, "season", "avg_allowed"),
        ),
        PlayerVsDefenseRow(
            key="rank_against_position",
            label="Defense: Rank vs Position",
            unit="rank",
            projection_value=None,
            defense_value=_get_defense_stat(opponent_allowed, "season", "rank_against_position"),
        ),
        PlayerVsDefenseRow(
            key="last_4_games_avg",
            label="Defense: L4 Avg Allowed",
            unit="yards",
            projection_value=None,
            defense_value=_get_defense_stat(opponent_allowed, "l4", "avg_allowed"),
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
    # red_zone_rate_allowed always blocked (requires PBP-derived aggregation).
    meta = meta.with_blocked(
        "red_zone_rate_allowed",
        *Unavailable.OPPONENT_ALLOWED_BY_POSITION,
    )

    # Other defense rows: block only when data is missing (no artifact
    # for this opponent/position/stat combination).
    if opponent_allowed is None or not opponent_allowed:
        meta = meta.with_blocked(
            "avg_allowed",
            *Unavailable.OPPONENT_ALLOWED_BY_POSITION,
        )
        meta = meta.with_blocked(
            "rank_against_position",
            *Unavailable.OPPONENT_ALLOWED_BY_POSITION,
        )
        meta = meta.with_blocked(
            "last_4_games_avg",
            *Unavailable.OPPONENT_ALLOWED_BY_POSITION,
        )

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


def _get_defense_stat(
    opponent_allowed: dict[str, dict] | None,
    cohort: str,
    field: str,
) -> float | int | None:
    """Extract a specific stat from the opponent_allowed dict.

    Returns None if the dict is None/empty or the cohort/field isn't present.
    """
    if opponent_allowed is None:
        return None
    cohort_data = opponent_allowed.get(cohort)
    if cohort_data is None:
        return None
    return cohort_data.get(field)
