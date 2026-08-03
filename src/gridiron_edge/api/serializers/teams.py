# src/gridiron_edge/api/serializers/teams.py

"""Serializers for /teams and /teams/{abbr}.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pandas import DataFrame, Series

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


def _trend_for_team(
    trends: DataFrame,
    team_abbr: str,
) -> float | None:
    """Return the Elo trend (delta from prior week) for a team.

    Returns None if trends DataFrame is empty or the team isn't found.
    """
    if trends.empty:
        return None
    match = trends.loc[trends["team_abbr"] == team_abbr]
    if match.empty:
        return None
    return _none_if_nan(match.iloc[0].get("elo_delta"))


def _percentile_for_team(
    percentiles: DataFrame,
    team_abbr: str,
) -> dict[str, float | None]:
    """Extract the four percentile fields for a team.

    Returns dict with rating_pct, avg_wins_pct, make_playoffs_pct,
    win_sb_pct keys. All None if the team isn't found or percentiles
    DataFrame is empty.
    """
    empty: dict[str, float | None] = {
        "rating_pct": None,
        "avg_wins_pct": None,
        "make_playoffs_pct": None,
        "win_sb_pct": None,
    }
    if percentiles.empty:
        return empty

    match: DataFrame | Series = percentiles.loc[percentiles["team_abbr"] == team_abbr]
    if match.empty:
        return empty

    row: Series | Any = match.iloc[0]
    return {
        "rating_pct": _none_if_nan(row.get("rating_pct")),
        "avg_wins_pct": _none_if_nan(row.get("avg_wins_pct")),
        "make_playoffs_pct": _none_if_nan(row.get("make_playoffs_pct")),
        "win_sb_pct": _none_if_nan(row.get("win_sb_pct")),
    }


def _compute_record(
    games: DataFrame,
    team_long_name: str,
) -> TeamRecord:
    """Count canonical Away/Home results for one team."""
    if games.empty:
        return TeamRecord()

    away_mask: Series[bool] = games["AWAY_TEAM"] == team_long_name
    home_mask: Series[bool] = games["HOME_TEAM"] == team_long_name
    participant_mask: Series[bool] = away_mask | home_mask

    away_scores = pd.to_numeric(
        games["AWAY_SCORE"],
        errors="coerce",
    )
    home_scores = pd.to_numeric(
        games["HOME_SCORE"],
        errors="coerce",
    )
    completed_mask = away_scores.notna() & home_scores.notna()

    away_wins = away_mask & completed_mask & (away_scores > home_scores)
    home_wins = home_mask & completed_mask & (home_scores > away_scores)
    away_losses = away_mask & completed_mask & (away_scores < home_scores)
    home_losses = home_mask & completed_mask & (home_scores < away_scores)
    ties = participant_mask & completed_mask & (away_scores == home_scores)

    return TeamRecord(
        wins=int((away_wins | home_wins).sum()),
        losses=int((away_losses | home_losses).sum()),
        ties=int(ties.sum()),
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
    percentiles: DataFrame,
    trends: DataFrame,
    team_metadata: dict[str, dict],
) -> TeamRankingsList:
    """Build the /teams power rankings response."""
    latest: DataFrame = _latest_ratings(elo, season, as_of_week)

    if latest.empty:
        meta: ResponseMeta = ResponseMeta().with_blocked("items", *Unavailable.NO_EVALUATION_DATA)
        return TeamRankingsList(
            season=season,
            as_of_week=as_of_week,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    ranked: DataFrame = latest.sort_values("ELO", ascending=False).reset_index(drop=True)
    season_games = games.loc[games["YEAR"] == season, :]

    rows: list[TeamRankingRow] = []
    for rank_idx, (_, r) in enumerate(ranked.iterrows()):
        long_name = r["NFL_TEAM"]
        abbr = long_to_short.get(long_name, long_name[:3].upper())
        pcts = _percentile_for_team(percentiles, abbr)
        team_meta = team_metadata.get(long_name, {})
        rows.append(
            TeamRankingRow(
                abbr=abbr,
                name=long_name,
                city=team_meta.get("city"),
                conference=team_meta.get("conference"),
                division=team_meta.get("division"),
                primary_color=team_meta.get("primary_color"),
                secondary_color=team_meta.get("secondary_color"),
                rating=_none_if_nan(r["ELO"]),
                rank=rank_idx + 1,
                record=_compute_record(season_games, long_name),
                trend=_trend_for_team(trends, abbr),
                rating_pct=pcts["rating_pct"],
                avg_wins_pct=pcts["avg_wins_pct"],
                make_playoffs_pct=pcts["make_playoffs_pct"],
                win_sb_pct=pcts["win_sb_pct"],
            ),
        )

    # Trend, off_rating, def_rating are null for every row.
    # Mark once at the items-level path.
    meta = ResponseMeta()
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
    """Serialize one canonical game from the requested team's view."""
    is_away = row["AWAY_TEAM"] == team_long_name
    is_designated_home = row["HOME_TEAM"] == team_long_name

    if not is_away and not is_designated_home:
        raise ValueError(
            f"Team {team_long_name!r} is not a participant in game {row.get('GAME_ID', '')!r}."
        )

    if is_away:
        opponent_long = str(row["HOME_TEAM"])
        score_for = int(row["AWAY_SCORE"])
        score_against = int(row["HOME_SCORE"])
    else:
        opponent_long = str(row["AWAY_TEAM"])
        score_for = int(row["HOME_SCORE"])
        score_against = int(row["AWAY_SCORE"])

    if score_for > score_against:
        result: Literal["L", "T", "W"] = "W"
    elif score_for < score_against:
        result = "L"
    else:
        result = "T"

    is_neutral = int(row.get("IS_NEUTRAL_SITE", 0)) == 1
    is_home = is_designated_home and not is_neutral

    opponent_short = long_to_short.get(
        opponent_long,
        opponent_long[:3].upper(),
    )

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
    percentiles: DataFrame,
    trends: DataFrame,
    team_metadata: dict[str, dict],
    cohort_splits: dict[str, dict] | None = None,
) -> TeamProfile:
    """Build the /teams/{abbr} response."""
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}
    long_name: str | None = short_to_long.get(abbr.upper())

    if long_name is None:
        # Unknown abbreviation. The route will 404 before this is reached,
        # but return a defensive shape if called directly.
        meta: ResponseMeta = ResponseMeta().with_blocked("name", *Unavailable.NO_EVALUATION_DATA)
        return TeamProfile(
            abbr=abbr,
            name=abbr,
            season=season,
            as_of_week=as_of_week,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    # ------------------------------------------------------------------
    # Team metadata (colors, city, conference, division)
    # ------------------------------------------------------------------
    team_meta = team_metadata.get(long_name, {})

    # ------------------------------------------------------------------
    # Rating + rank (from latest week ≤ as_of_week within season)
    # ------------------------------------------------------------------
    latest: DataFrame = _latest_ratings(elo, season, as_of_week)
    if latest.empty:
        rating: float | None = None
        rank: int | None = None
    else:
        ranked: DataFrame = latest.sort_values("ELO", ascending=False).reset_index(drop=True)
        team_row: DataFrame = ranked.loc[ranked["NFL_TEAM"] == long_name, :]
        if team_row.empty:
            rating, rank = None, None
        else:
            # pyrefly: ignore [bad-argument-type]
            rank = int(team_row.index[0]) + 1
            rating = float(team_row.iloc[0]["ELO"])

    # ------------------------------------------------------------------
    # Record within the season
    # ------------------------------------------------------------------
    season_games = games.loc[games["YEAR"] == season, :]
    record: TeamRecord = _compute_record(season_games, long_name)

    # ------------------------------------------------------------------
    # Rating history for this team within the season
    # ------------------------------------------------------------------
    hist = elo.loc[
        (elo["NFL_TEAM"] == long_name) & (elo["NFL_YEAR"] == season),
        ["NFL_WEEK", "ELO"],
    ].sort_values("NFL_WEEK")
    history: list[RatingHistoryPoint] | None = (
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
            (season_games["AWAY_TEAM"] == long_name) | (season_games["HOME_TEAM"] == long_name),
            :,
        ]
        .dropna(
            subset=[
                "AWAY_SCORE",
                "HOME_SCORE",
            ]
        )
        .sort_values("WEEK_NUM")
        .tail(6)
    )
    recent: list[RecentResult] | None = (
        [_serialize_result(r, long_name, long_to_short) for _, r in team_games.iterrows()]
        if not team_games.empty
        else None
    )

    # ------------------------------------------------------------------
    # Field-status metadata
    # ------------------------------------------------------------------
    meta = ResponseMeta()
    meta = meta.with_blocked("off_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_blocked("def_rating", *Unavailable.OFF_DEF_DECOMPOSITION)
    meta = meta.with_pending("schedule_difficulty")
    meta = meta.with_pending("playoff_probability")
    if cohort_splits is None:
        meta = meta.with_pending("cohort_splits")
    meta = meta.with_blocked("top_players", *Blocker.WAR)

    pcts: dict[str, float | None] = _percentile_for_team(percentiles, abbr.upper())

    return TeamProfile(
        abbr=abbr.upper(),
        name=long_name,
        city=team_meta.get("city"),
        conference=team_meta.get("conference"),
        division=team_meta.get("division"),
        primary_color=team_meta.get("primary_color"),
        secondary_color=team_meta.get("secondary_color"),
        season=season,
        as_of_week=as_of_week,
        rating=rating,
        rank=rank,
        record=record,
        trend=_trend_for_team(trends, abbr.upper()),
        rating_pct=pcts["rating_pct"],
        avg_wins_pct=pcts["avg_wins_pct"],
        make_playoffs_pct=pcts["make_playoffs_pct"],
        win_sb_pct=pcts["win_sb_pct"],
        rating_history=history,
        recent_results=recent,
        cohort_splits=cohort_splits,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )
