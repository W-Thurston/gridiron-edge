# src/gridiron_edge/api/serializers/projections.py

"""Serializer for /projections.

Per D17, hand-written. Per D18, owns _meta construction.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.api.loaders import ProjectionGridData
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.projections import (
    ProjectionGridResponse,
    ProjectionGridTeam,
    ProjectionGridWeek,
    ProjectionsList,
    TeamProjectionRow,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def serialize_projections(
    df: DataFrame,
    long_to_short: dict[str, str],
    season: str,
    computed_at: str | None,
    n_simulations: int | None,
) -> ProjectionsList:
    """Build the /projections response from the projections summary CSV.

    Maps CSV columns to schema fields:
        TEAM → abbr (already short)
        AVG_WINS → avg_wins
        P_MAKE_PLAYOFFS → make_playoffs
        P_REACH_DIV → reach_div
        P_REACH_CONF → reach_conf
        P_REACH_SB → reach_sb
        P_WIN_SB → win_sb
        elo_delta → elo_delta  (Elo rating change from prior same-season week)
    """
    # Invert long_to_short for name resolution: {abbr → long_name}
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}

    meta = ResponseMeta()

    if df.empty:
        # No projections CSV or empty file. Mark items as unavailable.
        meta: ResponseMeta = meta.with_blocked("items", *Unavailable.NO_PROJECTIONS_DATA)
        return ProjectionsList(
            season=season,
            computed_at=computed_at,
            n_simulations=n_simulations,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    # Sort by SB win probability descending.
    df_sorted: DataFrame = df.sort_values("P_WIN_SB", ascending=False).reset_index(drop=True)

    rows: list[TeamProjectionRow] = [
        TeamProjectionRow(
            abbr=str(row["TEAM"]),
            name=short_to_long.get(str(row["TEAM"]), str(row["TEAM"])),
            avg_wins=_none_if_nan(row.get("AVG_WINS")),
            make_playoffs=_none_if_nan(row.get("P_MAKE_PLAYOFFS")),
            reach_div=_none_if_nan(row.get("P_REACH_DIV")),
            reach_conf=_none_if_nan(row.get("P_REACH_CONF")),
            reach_sb=_none_if_nan(row.get("P_REACH_SB")),
            win_sb=_none_if_nan(row.get("P_WIN_SB")),
            elo_delta=_none_if_nan(row.get("elo_delta")),
        )
        for _, row in df_sorted.iterrows()
    ]

    # Clinched and eliminated remain pending for every projection row.
    meta = meta.with_pending("items.clinched")
    meta = meta.with_pending("items.eliminated")

    # Elo movement requires a prior week in the same season. Week 1 and
    # equivalent no-history states therefore have no prior snapshot.
    elo_delta = df_sorted.get("elo_delta")
    if elo_delta is None or not elo_delta.notna().any():
        meta = meta.with_blocked(
            "items.elo_delta",
            *Unavailable.NO_PRIOR_SNAPSHOT,
        )

    return ProjectionsList(
        season=season,
        computed_at=computed_at,
        n_simulations=n_simulations,
        items=rows,
        total=len(rows),
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def serialize_projection_grid(
    data: ProjectionGridData,
) -> ProjectionGridResponse:
    """Build the 32 x 18 weekly projection-grid response.

    The loader supplies season-scoped static sources. This serializer owns
    team-week state classification, matchup perspective, actual-result
    resolution, and field-status metadata.

    A missing scheduled game is a bye only when the schedule source is
    available. If the schedule source is unavailable, team-week rows are
    explicitly marked unavailable instead.
    """
    meta = ResponseMeta()

    if data.probabilities.empty:
        meta = meta.with_blocked(
            "items",
            *Unavailable.NO_PROJECTIONS_DATA,
        )
        return ProjectionGridResponse(
            season=data.season or None,
            completed_through_week=data.completed_through_week,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if not data.schedule_available:
        meta = meta.with_blocked(
            "items.weeks",
            *Unavailable.NO_SCHEDULE_DATA,
        )

    short_to_long = {short: long_name for long_name, short in data.long_to_short.items()}

    schedule_by_team_week = _index_grid_schedule(
        data.schedule,
        data.long_to_short,
    )
    games_by_id = _index_completed_games(
        data.games,
        data.long_to_short,
    )

    rows: list[ProjectionGridTeam] = []

    for _, probability_row in data.probabilities.iterrows():
        abbr = str(probability_row["TEAM"])
        weeks = [
            _serialize_grid_week(
                probability_row=probability_row,
                abbr=abbr,
                week=week,
                schedule_by_team_week=schedule_by_team_week,
                games_by_id=games_by_id,
                schedule_available=data.schedule_available,
            )
            for week in range(1, 19)
        ]

        rows.append(
            ProjectionGridTeam(
                abbr=abbr,
                name=short_to_long.get(abbr, abbr),
                weeks=weeks,
            )
        )

    rows.sort(key=lambda row: row.name)

    return ProjectionGridResponse(
        season=data.season or None,
        completed_through_week=data.completed_through_week,
        regular_season_weeks=18,
        items=rows,
        total=len(rows),
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def _index_grid_schedule(
    schedule: DataFrame,
    long_to_short: dict[str, str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Index each scheduled game from both teams' perspectives."""
    if schedule.empty:
        return {}

    required = {
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
        "GAME_ID",
    }
    if not required.issubset(schedule.columns):
        return {}

    indexed: dict[tuple[str, int], dict[str, Any]] = {}

    for row in schedule.itertuples(index=False):
        week = int(str(row.WEEK_NUM))
        away_long = str(row.AWAY_TEAM)
        home_long = str(row.HOME_TEAM)
        away = data_value(long_to_short.get(away_long))
        home = data_value(long_to_short.get(home_long))

        if away is None or home is None:
            continue

        common = {
            "game_id": data_value(getattr(row, "GAME_ID", None)),
            "game_date": data_value(getattr(row, "GAME_DATE", None)),
            "game_time": data_value(getattr(row, "GAMETIME", None)),
        }

        indexed[(away, week)] = {
            **common,
            "opponent": home,
            "is_home": False,
        }
        indexed[(home, week)] = {
            **common,
            "opponent": away,
            "is_home": True,
        }

    return indexed


def _index_completed_games(
    games: DataFrame,
    long_to_short: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Index completed regular-season games by canonical game ID."""
    if games.empty or "GAME_ID" not in games.columns:
        return {}

    indexed: dict[str, dict[str, Any]] = {}

    for row in games.itertuples(index=False):
        game_id = str(row.GAME_ID)

        winner = data_value(getattr(row, "WINNER", None))
        loser = data_value(getattr(row, "LOSER", None))

        indexed[game_id] = {
            "winner": winner,
            "loser": loser,
            "winner_abbr": (long_to_short.get(winner) if winner is not None else None),
            "loser_abbr": (long_to_short.get(loser) if loser is not None else None),
            "win_or_tie": _none_if_nan(getattr(row, "WIN_OR_TIE", None)),
        }

    return indexed


def _serialize_grid_week(
    *,
    probability_row: Series,
    abbr: str,
    week: int,
    schedule_by_team_week: dict[
        tuple[str, int],
        dict[str, Any],
    ],
    games_by_id: dict[str, dict[str, Any]],
    schedule_available: bool,
) -> ProjectionGridWeek:
    """Serialize one team/week using schedule and completed-game context."""
    schedule_game = schedule_by_team_week.get((abbr, week))
    probability = _grid_probability(probability_row, week)

    if schedule_game is None:
        state = "bye" if schedule_available else "unavailable"
        return ProjectionGridWeek(
            week=week,
            state=state,
        )

    game_id = schedule_game["game_id"]
    completed_game = games_by_id.get(game_id) if game_id is not None else None

    if completed_game is not None:
        return ProjectionGridWeek(
            week=week,
            state="played",
            opponent=schedule_game["opponent"],
            is_home=schedule_game["is_home"],
            game_id=game_id,
            game_date=schedule_game["game_date"],
            game_time=schedule_game["game_time"],
            win_probability=probability,
            actual_result=_actual_result(
                abbr=abbr,
                completed_game=completed_game,
            ),
        )

    if probability is None:
        return ProjectionGridWeek(
            week=week,
            state="unavailable",
            opponent=schedule_game["opponent"],
            is_home=schedule_game["is_home"],
            game_id=game_id,
            game_date=schedule_game["game_date"],
            game_time=schedule_game["game_time"],
        )

    return ProjectionGridWeek(
        week=week,
        state="projected",
        opponent=schedule_game["opponent"],
        is_home=schedule_game["is_home"],
        game_id=game_id,
        game_date=schedule_game["game_date"],
        game_time=schedule_game["game_time"],
        win_probability=probability,
    )


def _grid_probability(
    probability_row: Series,
    week: int,
) -> float | None:
    """Read a weekly win-probability column from a season-grid row."""
    column = f"W{week:02d}_WIN_P"
    value = probability_row.get(column)
    result = _none_if_nan(value)

    if result is None:
        return None

    return float(result)


def _actual_result(
    *,
    abbr: str,
    completed_game: dict[str, Any],
) -> Literal["W", "L", "T"] | None:
    """Resolve W/L/T for one team from a completed-game record."""
    tie_value = completed_game.get("win_or_tie")
    if tie_value is not None and float(tie_value) == 0.5:
        return "T"

    winner = completed_game.get("winner")
    loser = completed_game.get("loser")

    # Completed-games sources use long names. The schedule index has already
    # established the row team's short code, so callers add short-name fields
    # before invoking this helper.
    if completed_game.get("winner_abbr") == abbr:
        return "W"
    if completed_game.get("loser_abbr") == abbr:
        return "L"

    if winner == abbr:
        return "W"
    if loser == abbr:
        return "L"

    return None


def data_value(value: object) -> str | None:
    """Return string source data or None for missing scalar values."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)
