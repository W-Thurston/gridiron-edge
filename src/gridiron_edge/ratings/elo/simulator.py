"""Canonical Elo history simulator.

Single source of truth for constructing Elo state and per-game
predictions from historical games. Replaces the structurally duplicate
simulators previously embedded in ``ratings/elo/table.py`` and
``evaluation/tune.py``.

Returns an :class:`EloSimulationResult` containing both:

- the full Elo dict keyed by ``(team, year, week)``, used by the state
  table builder, and
- the per-game predictions used by the tuner and the Elo predictor.

Numba kernels in ``sim/_engine.py`` and ``sim/playoffs.py`` are
intentionally not affected: they operate on numeric Elo vectors at
simulation time, and their parity with this module's math is pinned by
``tests/unit/ratings/test_elo_core.py::TestPythonNumbaParity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.ratings.elo.core import (
    DEFAULT_ELO_DIVISOR,
    elo_win_probability,
    update_elo,
)

_DEFAULT_INITIAL_ELO: float = 1500.0
_DEFAULT_EXPANSION_ELO: float = 1300.0


@dataclass(frozen=True)
class EloSimulationResult:
    """Output of :func:`simulate_elo_history`.

    Attributes:
        elo: Mapping ``(team, year, week) -> rating``. Used by the state
            table builder.
        away_probs: Predicted away-team win probability per scored game.
        away_outcomes: Actual away-team outcome per scored game
            (``1.0`` win, ``0.0`` loss, ``0.5`` tie).
        game_seasons: Season label per scored game.
        game_ids: Canonical ``GAME_ID`` per scored game.
    """

    elo: dict[tuple[str, str, int], float] = field(default_factory=dict)
    away_probs: list[float] = field(default_factory=list)
    away_outcomes: list[float] = field(default_factory=list)
    game_seasons: list[str] = field(default_factory=list)
    game_ids: list[str] = field(default_factory=list)


def _k_for_week(
    week: int,
    k_early: float,
    k_mid: float,
    k_week18: float,
    k_post: float,
) -> float:
    """Return the K-factor for the given week number."""
    if 1 <= week <= 4:
        return k_early
    if 5 <= week <= 17:
        return k_mid
    if week == 18:
        return k_week18
    if 19 <= week <= 22:
        return k_post
    return k_mid


def transition_to_next_season(
    final_ratings: dict[str, float],
    *,
    returning_teams: set[str],
    expansion_start: dict[str, str],
    next_year: str,
    regress_frac: float,
    initial_elo: float = _DEFAULT_INITIAL_ELO,
    expansion_elo: float = _DEFAULT_EXPANSION_ELO,
) -> dict[str, float]:
    """Build deterministic Week 1 ratings for the next season.

    Returning teams regress toward their own mean using the same policy
    applied between historical seasons. Teams whose recorded expansion
    season is next_year enter at expansion_elo.

    The function performs no date lookup, schedule loading, mutation, or
    rating update.
    """
    transitioned: dict[str, float] = {}

    if returning_teams:
        ratings = [
            final_ratings.get(
                team,
                initial_elo,
            )
            for team in returning_teams
        ]
        season_mean = sum(ratings) / len(ratings)

        for team in returning_teams:
            current = final_ratings.get(
                team,
                initial_elo,
            )
            transitioned[team] = season_mean * regress_frac + current * (1.0 - regress_frac)

    for team, start_year in expansion_start.items():
        if start_year == next_year:
            transitioned[team] = expansion_elo

    return transitioned


def simulate_elo_history(  # noqa: PLR0912, PLR0915
    games: pd.DataFrame,
    sorted_years: list[str],
    teams_by_year: dict[str, set[str]],
    expansion_start: dict[str, str],
    *,
    k_early: float,
    k_mid: float,
    k_week18: float,
    k_post: float,
    divisor: float = DEFAULT_ELO_DIVISOR,
    regress_frac: float,
    initial_elo: float = _DEFAULT_INITIAL_ELO,
    expansion_elo: float = _DEFAULT_EXPANSION_ELO,
) -> EloSimulationResult:
    """Simulate Elo across the full historical games DataFrame.

    Args:
        games: Prepared games DataFrame. Must contain
            ``YEAR, WEEK_NUM, WINNER, LOSER, WIN_OR_TIE, GAME_LOCATION,
            GAME_ID`` columns.
        sorted_years: Chronologically ordered season labels.
        teams_by_year: Season label to active team name set.
        expansion_start: Team name to first active season label.
        k_early: K-factor for weeks 1-4.
        k_mid: K-factor for weeks 5-17.
        k_week18: K-factor for week 18.
        k_post: K-factor for weeks 19-22.
        divisor: Win-probability divisor passed to :func:`update_elo`
            and :func:`elo_win_probability`.
        regress_frac: Offseason regression fraction toward the league
            mean.
        initial_elo: Starting rating for non-expansion teams.
        expansion_elo: Starting rating for expansion franchises in their
            inaugural season.

    Returns:
        Populated :class:`EloSimulationResult`.
    """
    elo: dict[tuple[str, str, int], float] = {}

    if sorted_years:
        first_year: str = sorted_years[0]
        for team in teams_by_year.get(first_year, set()):
            elo[(team, first_year, 1)] = initial_elo

    away_probs: list[float] = []
    away_outcomes: list[float] = []
    game_seasons: list[str] = []
    game_ids: list[str] = []

    games_idx = games.groupby(["YEAR", "WEEK_NUM"])

    for yr_idx, curr_year in enumerate(sorted_years):
        next_year: str | None = sorted_years[yr_idx + 1] if yr_idx < len(sorted_years) - 1 else None
        teams_this_season: set[str] = teams_by_year.get(curr_year, set())

        season_games = games.loc[games["YEAR"] == curr_year]
        weeks_with_games = sorted(season_games["WEEK_NUM"].unique().tolist())
        if not weeks_with_games:
            continue
        max_week = max(weeks_with_games)

        for wk in range(1, max_week + 1):
            k: float = _k_for_week(wk, k_early, k_mid, k_week18, k_post)

            try:
                week_df = games_idx.get_group((curr_year, wk))
            except KeyError:
                week_df = pd.DataFrame()

            for _, row in week_df.iterrows():
                winner = str(row["WINNER"])
                loser = str(row["LOSER"])
                win_or_tie = float(row["WIN_OR_TIE"])

                w_elo: float = elo.get((winner, curr_year, wk), initial_elo)
                l_elo: float = elo.get((loser, curr_year, wk), initial_elo)

                away_team: str = winner if row["GAME_LOCATION"] == _AWAY_WIN_LOCATION else loser
                away_elo: float = w_elo if away_team == winner else l_elo
                home_elo: float = l_elo if away_team == winner else w_elo

                away_prob, _ = elo_win_probability(away_elo, home_elo, divisor=divisor)
                if win_or_tie == 0.5:
                    outcome = 0.5
                elif away_team == winner:
                    outcome = 1.0
                else:
                    outcome = 0.0

                away_probs.append(away_prob)
                away_outcomes.append(outcome)
                game_seasons.append(curr_year)
                game_ids.append(str(row.get("GAME_ID", "")))

                winner_new, loser_new = update_elo(
                    w_elo,
                    l_elo,
                    win_or_tie=win_or_tie,
                    k=k,
                    divisor=divisor,
                )

                is_last_week = wk == max_week
                if is_last_week and next_year is not None:
                    elo[(winner, next_year, 1)] = winner_new
                    elo[(loser, next_year, 1)] = loser_new
                else:
                    elo[(winner, curr_year, wk + 1)] = winner_new
                    elo[(loser, curr_year, wk + 1)] = loser_new

            is_last_week = wk == max_week
            for team in teams_this_season:
                curr_key: tuple[str, str, int] = (team, curr_year, wk)
                if is_last_week and next_year is not None:
                    next_key = (team, next_year, 1)
                    if next_key not in elo:
                        elo[next_key] = elo.get(curr_key, initial_elo)
                elif not is_last_week:
                    next_key: tuple[str, str, int] = (team, curr_year, wk + 1)
                    if next_key not in elo:
                        elo[next_key] = elo.get(curr_key, initial_elo)

        if next_year is not None:
            returning: set[str] = teams_this_season & teams_by_year.get(
                next_year,
                set(),
            )
            final_ratings: dict[str, float] = {
                team: elo.get(
                    (team, next_year, 1),
                    initial_elo,
                )
                for team in returning
            }

            transitioned: dict[str, float] = transition_to_next_season(
                final_ratings,
                returning_teams=returning,
                expansion_start=expansion_start,
                next_year=next_year,
                regress_frac=regress_frac,
                initial_elo=initial_elo,
                expansion_elo=expansion_elo,
            )

            for team, rating in transitioned.items():
                elo[(team, next_year, 1)] = rating

    return EloSimulationResult(
        elo=elo,
        away_probs=away_probs,
        away_outcomes=away_outcomes,
        game_seasons=game_seasons,
        game_ids=game_ids,
    )
