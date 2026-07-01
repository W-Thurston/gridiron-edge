# src/gridiron_edge/api/serializers/teams.py

"""Serializers for /teams and /teams/{abbr}.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import Blocker, ResponseMeta, Unavailable
from gridiron_edge.api.schemas.teams import (
    RatingHistoryPoint,
    RecentResult,
    TeamProfile,
    TeamRankingRow,
    TeamRankingsList,
    TeamRecord,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _compute_record(games: DataFrame, team_long_name: str) -> TeamRecord:
    """Count wins/losses/ties for a team in the given games slice."""
    if games.empty:
        return TeamRecord()

    won_mask = games["WINNER"] == team_long_name
    lost_mask = games["LOSER"] == team_long_name

    # Ties: score equal AND team is one of the two participants.
    tie_mask = (games["PTS_WINNER"] == games["PTS_LOSER"]) & (won_mask | lost_mask)

    return TeamRecord(
        wins=(won_mask & ~tie_mask).sum(),
        losses=(lost_mask & ~tie_mask).sum(),
        ties=tie_mask.sum(),
    )


def _latest_ratings(elo: DataFrame, season: str, week: int) -> DataFrame:
    """Filter Elo state to the latest week ≤ `week` per team within `season`."""
    if elo.empty:
        return elo

    scope = elo.loc[
        (elo["NFL_YEAR"] == season) & (elo["NFL_WEEK"] <= week),
        :,
    ]
    if scope.empty:
        return scope

    return scope.sort_values(["NFL_TEAM", "NFL_WEEK"]).groupby("NFL_TEAM", as_index=False).tail(1)


def serialize_team_rankings(
    elo: DataFrame,
    games: DataFrame,
    long_to_short: dict[str, str],
    season: str,
    as_of_week: int,
) -> TeamRankingsList:
    """Build the /teams power rankings response."""
    latest = _latest_ratings(elo, season, as_of_week)

    if latest.empty:
        meta = ResponseMeta().with_blocked("items", *Unavailable.NO_EVALUATION_DATA)
        return TeamRankingsList(
            season=season,
            as_of_week=as_of_week,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    ranked = latest.sort_values("ELO", ascending=False).reset_index(drop=True)
    season_games = games.loc[games["YEAR"] == season, :]

    rows: list[TeamRankingRow] = []
    for rank_idx, (_, r) in enumerate(ranked.iterrows()):
        long_name = r["NFL_TEAM"]
        abbr = long_to_short.get(long_name, long_name[:3].upper())
        rows.append(
            TeamRankingRow(
                abbr=abbr,
                name=long_name,
                rating=_none_if_nan(r["ELO"]),
                rank=rank_idx + 1,
                record=_compute_record(season_games, long_name),
            ),
        )

    # Trend, off_rating, def_rating are null for every row.
    # Mark once at the items-level path.
    meta = ResponseMeta()
    meta = meta.with_blocked("items.trend", *Unavailable.NO_PRIOR_SNAPSHOT)
    meta = meta.with_blocked("items.off_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("items.def_rating", *Unavailable.OFF_DEF_DECOMPOSITION)

    return TeamRankingsList(
        season=season,
        as_of_week=as_of_week,
        items=rows,
        total=len(rows),
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def _serialize_result(
    row: pd.Series,
    team_long_name: str,
    long_to_short: dict[str, str],
) -> RecentResult:
    """Translate one row of games to a RecentResult from `team_long_name`'s POV.

    GAME_LOCATION semantics (verified against data):
    - "@" → the WINNER played on the road.
    - "H" → the WINNER played at home.

    So if the team is the WINNER and location is "@", team was away.
    If the team is the LOSER and location is "@", team was home.
    """
    won = row["WINNER"] == team_long_name
    winner_at_home = row["GAME_LOCATION"] == "H"
    is_home = winner_at_home if won else not winner_at_home

    if won:
        opponent_long = row["LOSER"]
        score_for = int(row["PTS_WINNER"])
        score_against = int(row["PTS_LOSER"])
    else:
        opponent_long = row["WINNER"]
        score_for = int(row["PTS_LOSER"])
        score_against = int(row["PTS_WINNER"])

    # Ties: same score for both sides.
    result = "T" if score_for == score_against else "W" if won else "L"

    opponent_short = long_to_short.get(opponent_long, opponent_long[:3].upper())

    return RecentResult(
        week=int(row["WEEK_NUM"]),
        date=str(row.get("GAME_DATE", "")),
        opponent=opponent_short,
        is_home=is_home,
        result=result,
        score_for=score_for,
        score_against=score_against,
    )


def serialize_team_profile(
    abbr: str,
    elo: DataFrame,
    games: DataFrame,
    long_to_short: dict[str, str],
    season: str,
    as_of_week: int,
) -> TeamProfile:
    """Build the /teams/{abbr} response."""
    short_to_long = {v: k for k, v in long_to_short.items()}
    long_name = short_to_long.get(abbr.upper())

    if long_name is None:
        # Unknown abbreviation. The route will 404 before this is reached,
        # but return a defensive shape if called directly.
        meta = ResponseMeta().with_blocked("name", *Unavailable.NO_EVALUATION_DATA)
        return TeamProfile(
            abbr=abbr,
            name=abbr,
            season=season,
            as_of_week=as_of_week,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    # ------------------------------------------------------------------
    # Rating + rank (from latest week ≤ as_of_week within season)
    # ------------------------------------------------------------------
    latest = _latest_ratings(elo, season, as_of_week)
    if latest.empty:
        rating: float | None = None
        rank: int | None = None
    else:
        ranked = latest.sort_values("ELO", ascending=False).reset_index(drop=True)
        team_row = ranked.loc[ranked["NFL_TEAM"] == long_name]
        if team_row.empty:
            rating, rank = None, None
        else:
            rank = int(team_row.index[0]) + 1
            rating = float(team_row.iloc[0]["ELO"])

    # ------------------------------------------------------------------
    # Record within the season
    # ------------------------------------------------------------------
    season_games = games.loc[games["YEAR"] == season, :]
    record = _compute_record(season_games, long_name)

    # ------------------------------------------------------------------
    # Rating history for this team within the season
    # ------------------------------------------------------------------
    hist = elo.loc[
        (elo["NFL_TEAM"] == long_name) & (elo["NFL_YEAR"] == season),
        ["NFL_WEEK", "ELO"],
    ].sort_values("NFL_WEEK")
    history = (
        [
            RatingHistoryPoint(week=int(r["NFL_WEEK"]), rating=float(r["ELO"]))
            for _, r in hist.iterrows()
        ]
        if not hist.empty
        else None
    )

    # ------------------------------------------------------------------
    # Recent results (last 6 games of the season)
    # ------------------------------------------------------------------
    team_games = (
        season_games.loc[
            (season_games["WINNER"] == long_name) | (season_games["LOSER"] == long_name),
            :,
        ]
        .sort_values("WEEK_NUM")
        .tail(6)
    )
    recent = (
        [_serialize_result(r, long_name, long_to_short) for _, r in team_games.iterrows()]
        if not team_games.empty
        else None
    )

    # ------------------------------------------------------------------
    # Field-status metadata
    # ------------------------------------------------------------------
    meta = ResponseMeta()
    meta = meta.with_blocked("trend", *Unavailable.NO_PRIOR_SNAPSHOT)
    meta = meta.with_blocked("off_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("def_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_pending("schedule_difficulty")
    meta = meta.with_pending("playoff_probability")
    meta = meta.with_pending("situational_splits")
    meta = meta.with_blocked("top_players", *Blocker.WAR)

    return TeamProfile(
        abbr=abbr.upper(),
        name=long_name,
        season=season,
        as_of_week=as_of_week,
        rating=rating,
        rank=rank,
        record=record,
        rating_history=history,
        recent_results=recent,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
