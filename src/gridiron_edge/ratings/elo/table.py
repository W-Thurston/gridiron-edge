"""Elo state table construction.

Builds the canonical ``NFL_Team_Elo.csv`` used by downstream predict,
features, and viz modules. Delegates the simulation to the canonical
:mod:`ratings.elo.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.console import console
from gridiron_edge.core.constants import (
    EXPANSION_TEAMS as EXPANSION_START_YEAR,
)
from gridiron_edge.ratings.elo.simulator import (
    EloSimulationResult,
    transition_to_next_season,
)


@dataclass(frozen=True)
class EloTableConfig:
    """Configuration parameters for Elo table construction."""

    k: float = 20.0
    initial_elo: float = 1500.0
    expansion_elo: float = 1300.0
    offseason_regress_frac: float = 1 / 3.0
    divisor: float = 480.0


def _next_season_label(year: str) -> str:
    """Derive the next season label from one historical season label."""
    parts: list[str] = year.split("-")

    if len(parts) != 2:
        raise ValueError(f"Invalid NFL season label {year!r}. Expected format YYYY-YYYY.")

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid NFL season label {year!r}. Expected numeric years.") from exc

    if end != start + 1:
        raise ValueError(
            f"Invalid NFL season label {year!r}. "
            "Ending year must be one greater than starting year."
        )

    return f"{end}-{end + 1}"


def _max_week_for_year(games: pd.DataFrame, year: str) -> int:
    """Return the maximum week number for a given season."""
    subset = games.loc[games["YEAR"] == year, "WEEK_NUM"]
    return int(subset.max()) if not subset.empty else 1


def _add_next_season_week_one(
    elo: dict[tuple[str, str, int], float],
    *,
    games: DataFrame,
    sorted_years: list[str],
    teams_by_year: dict[str, set[str]],
    cfg: EloTableConfig,
) -> None:
    """Append one deterministic synthetic next-season Week 1 state.

    The transition begins from the final postgame state of the latest
    historical season. Returning teams receive the same offseason
    regression used between historical seasons. Expansion teams whose
    configured start season matches the derived next season receive the
    configured expansion rating.

    The input Elo mapping is updated in place. Existing historical rows
    are not altered.
    """
    if not sorted_years:
        return

    latest_year: str = sorted_years[-1]
    next_year: str = _next_season_label(latest_year)

    max_week: int = _max_week_for_year(
        games,
        latest_year,
    )
    final_state_week: int = max_week + 1

    returning_teams: set[str] = teams_by_year.get(
        latest_year,
        set(),
    )

    final_ratings: dict[str, float] = {
        team: elo.get(
            (
                team,
                latest_year,
                final_state_week,
            ),
            cfg.initial_elo,
        )
        for team in returning_teams
    }

    transitioned: dict[str, float] = transition_to_next_season(
        final_ratings,
        returning_teams=returning_teams,
        expansion_start=EXPANSION_START_YEAR,
        next_year=next_year,
        regress_frac=cfg.offseason_regress_frac,
        initial_elo=cfg.initial_elo,
        expansion_elo=cfg.expansion_elo,
    )

    for team, rating in transitioned.items():
        key = (
            team,
            next_year,
            1,
        )
        if key not in elo:
            elo[key] = rating


def build_elo_state_table_all_years(
    games: pd.DataFrame,
    *,
    cfg: EloTableConfig | None = None,
) -> pd.DataFrame:
    """Build the full Elo state table from historical game results."""
    from gridiron_edge.evaluation.tune import _prepare_games
    from gridiron_edge.ratings.elo.simulator import simulate_elo_history

    cfg = cfg or EloTableConfig()

    games_prepared, sorted_years, teams_by_year = _prepare_games(games)

    result: EloSimulationResult = simulate_elo_history(
        games_prepared,
        sorted_years,
        teams_by_year,
        EXPANSION_START_YEAR,
        k_early=cfg.k,
        k_mid=cfg.k,
        k_week18=cfg.k,
        k_post=cfg.k,
        divisor=cfg.divisor,
        regress_frac=cfg.offseason_regress_frac,
        initial_elo=cfg.initial_elo,
        expansion_elo=cfg.expansion_elo,
    )

    elo_dict: dict[tuple[str, str, int], float] = dict(result.elo)

    _add_next_season_week_one(
        elo_dict,
        games=games_prepared,
        sorted_years=sorted_years,
        teams_by_year=teams_by_year,
        cfg=cfg,
    )

    rows: list[dict[str, float | int | str]] = [
        {"NFL_TEAM": team, "NFL_YEAR": year, "NFL_WEEK": week, "ELO": elo}
        for (team, year, week), elo in elo_dict.items()
    ]

    if not rows:
        return pd.DataFrame(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"])

    df_out: DataFrame = (
        pd.DataFrame(rows).sort_values(["NFL_YEAR", "NFL_WEEK", "NFL_TEAM"]).reset_index(drop=True)
    )

    if console.verbose:
        n_teams: int = df_out["NFL_TEAM"].nunique()
        n_seasons: int = df_out["NFL_YEAR"].nunique()
        print(f"  Elo table: {len(df_out):,} rows  {n_teams} teams  {n_seasons} seasons")

    return df_out


def update_elo_state_incremental(
    elo_state_existing: pd.DataFrame,
    games: pd.DataFrame,
    *,
    cfg: EloTableConfig | None = None,
) -> pd.DataFrame:
    """Incrementally update an existing Elo state table with new game results.

    No new completed games (e.g., offseason — prior season already baked in,
    next season not yet played) → existing state is already current; return
    it unchanged.
    """
    if games.empty:
        return elo_state_existing
    return build_elo_state_table_all_years(games, cfg=cfg)
