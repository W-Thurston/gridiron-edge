"""Elo state table construction.

Builds the canonical ``NFL_Team_Elo.csv`` used by downstream predict,
features, and viz modules. Delegates the simulation to the canonical
:mod:`ratings.elo.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.console import console
from gridiron_edge.core.constants import EXPANSION_TEAMS as EXPANSION_START_YEAR
from gridiron_edge.ratings.elo.simulator import EloSimulationResult


@dataclass(frozen=True)
class EloTableConfig:
    """Configuration parameters for Elo table construction."""

    k: float = 20.0
    initial_elo: float = 1500.0
    expansion_elo: float = 1300.0
    offseason_regress_frac: float = 1 / 3.0
    divisor: float = 480.0


def _build_years(df: pd.DataFrame) -> list[str]:
    max_year = df["YEAR"].max()
    if df.loc[df["YEAR"] == max_year, "WEEK_NUM"].max() == 22:
        now: datetime = datetime.now(tz=UTC)
        return [*sorted(df["YEAR"].unique().tolist()), f"{now.year}-{now.year + 1}"]
    return sorted(df["YEAR"].unique().tolist())


def _max_week_for_year(games: pd.DataFrame, year: str) -> int:
    """Return the maximum week number for a given season."""
    subset = games.loc[games["YEAR"] == year, "WEEK_NUM"]
    return int(subset.max()) if not subset.empty else 1


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

    nfl_years: list[str] = _build_years(games)
    if nfl_years and nfl_years[-1] not in sorted_years:
        next_year: str = nfl_years[-1]
        prev_year: str = sorted_years[-1]
        for team in teams_by_year.get(prev_year, set()):
            key_prev: tuple[str, str, int] = (
                team,
                prev_year,
                _max_week_for_year(games_prepared, prev_year),
            )
            key_next = (team, next_year, 1)
            if key_next not in elo_dict and key_prev in elo_dict:
                elo_dict[key_next] = elo_dict[key_prev]

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
