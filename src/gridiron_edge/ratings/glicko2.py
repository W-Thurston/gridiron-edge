# src/gridiron_edge/ratings/glicko2.py

"""Team Glicko-2 ratings recomputed from complete game history.

Glicko-2 tracks a rating, a rating deviation (RD, the uncertainty of the
rating) and a volatility (how erratic the team's results are). One weekly
rating period covers every game in a (season, week); a team without a game
that week (a bye, or an eliminated playoff team) takes the standard no-game
step, which only widens RD.

The ratings are path-dependent from 1999 but are recomputed in memory rather
than persisted, so no separate lineage artifact exists; the games history is
validated as complete and contiguous instead (``DECISIONS.md`` D69).

A value for a (season, week) is the state after every completed game with a
strictly earlier (season, week). A week beyond the last completed game
therefore gets the last state, carried forward unchanged (D69). Crossing into a
new season applies one offseason transition.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Final, NamedTuple

from gridiron_edge.core.constants import EXPANSION_TEAMS
from gridiron_edge.ratings.history import (
    CutoffKey,
    away_outcome,
    prepare_history,
    season_label,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas import DataFrame

#: Conversion between the public rating scale and the Glicko-2 internal scale.
GLICKO2_SCALE: Final[float] = 173.7178
_CENTER: Final[float] = 1500.0
_CONVERGENCE: Final[float] = 1e-6


@dataclass(frozen=True)
class Glicko2Config:
    """Glicko-2 parameters (``DECISIONS.md`` D70); regression and expansion mirror team Elo."""

    tau: float = 0.5
    offseason_periods: int = 35
    regress_frac: float = 1 / 3
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_volatility: float = 0.06
    expansion_rating: float = 1300.0


class Glicko2State(NamedTuple):
    """One team's rating on the public (Elo-like) scale."""

    rating: float
    rd: float
    volatility: float


# Internal scale: (mu, phi, sigma).
type _Internal = tuple[float, float, float]
type _Result = tuple[float, float, float]  # (opponent mu, opponent phi, score)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi**2)


def _expected(mu: float, mu_opp: float, phi_opp: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_opp) * (mu - mu_opp)))


def _new_volatility(phi: float, sigma: float, delta: float, v: float, tau: float) -> float:
    """Solve for the new volatility (Glickman step 5, Illinois algorithm)."""
    a = math.log(sigma * sigma)
    phi_sq = phi * phi

    def f(x: float) -> float:
        ex = math.exp(x)
        return ex * (delta * delta - phi_sq - v - ex) / (2.0 * (phi_sq + v + ex) ** 2) - (x - a) / (
            tau * tau
        )

    lower = a
    if delta * delta > phi_sq + v:
        upper = math.log(delta * delta - phi_sq - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
        upper = a - k * tau

    f_lower, f_upper = f(lower), f(upper)
    while abs(upper - lower) > _CONVERGENCE:
        c = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_c = f(c)
        if f_c * f_upper <= 0.0:
            lower, f_lower = upper, f_upper
        else:
            f_lower /= 2.0
        upper, f_upper = c, f_c
    return math.exp(lower / 2.0)


def glicko2_step(
    mu: float,
    phi: float,
    sigma: float,
    results: list[_Result],
    *,
    tau: float,
) -> _Internal:
    """Advance one rating period on the Glicko-2 internal scale.

    With no results the period only widens RD: ``phi' = sqrt(phi^2 + sigma^2)``.
    """
    if not results:
        return mu, math.sqrt(phi * phi + sigma * sigma), sigma

    v_inv = 0.0
    improvement = 0.0
    for mu_j, phi_j, score in results:
        e = _expected(mu, mu_j, phi_j)
        v_inv += _g(phi_j) ** 2 * e * (1.0 - e)
        improvement += _g(phi_j) * (score - e)
    v = 1.0 / v_inv
    delta = v * improvement

    new_sigma = _new_volatility(phi, sigma, delta, v, tau)
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * improvement
    return new_mu, new_phi, new_sigma


def expected_score(team: Glicko2State, opponent: Glicko2State) -> float:
    """Return ``team``'s expected score against ``opponent`` (no home term)."""
    return _expected(
        (team.rating - _CENTER) / GLICKO2_SCALE,
        (opponent.rating - _CENTER) / GLICKO2_SCALE,
        opponent.rd / GLICKO2_SCALE,
    )


def _public(state: _Internal) -> Glicko2State:
    mu, phi, sigma = state
    return Glicko2State(mu * GLICKO2_SCALE + _CENTER, phi * GLICKO2_SCALE, sigma)


class Glicko2History:
    """Full Glicko-2 replay with as-of lookup by (season, week)."""

    def __init__(self, games: DataFrame, config: Glicko2Config | None = None) -> None:
        self._cfg = config or Glicko2Config()
        history = prepare_history(games)
        self._keys: list[CutoffKey] = []
        self._after: list[dict[str, _Internal]] = []
        self._replay(history)

    def _default(self, team: str, season: int) -> _Internal:
        cfg = self._cfg
        if EXPANSION_TEAMS.get(team) == season_label(season):
            rating = cfg.expansion_rating
        else:
            rating = cfg.initial_rating
        return (
            (rating - _CENTER) / GLICKO2_SCALE,
            cfg.initial_rd / GLICKO2_SCALE,
            cfg.initial_volatility,
        )

    def _transition(self, state: dict[str, _Internal], season: int) -> dict[str, _Internal]:
        """Apply the offseason into ``season``: mean regression, RD widening."""
        cfg = self._cfg
        mean = sum(mu for mu, _, _ in state.values()) / len(state) if state else 0.0
        cap = cfg.initial_rd / GLICKO2_SCALE
        moved: dict[str, _Internal] = {}
        for team, (mu, phi, sigma) in state.items():
            widened = math.sqrt(phi * phi + cfg.offseason_periods * sigma * sigma)
            moved[team] = (
                mean * cfg.regress_frac + mu * (1.0 - cfg.regress_frac),
                min(widened, cap),
                sigma,
            )
        label = season_label(season)
        for team, start in EXPANSION_TEAMS.items():
            if start == label:
                moved[team] = self._default(team, season)
        return moved

    def _replay(self, history: DataFrame) -> None:
        cfg = self._cfg
        state: dict[str, _Internal] = {}
        previous_season: int | None = None

        for (season, week), week_games in history.groupby(["SEASON", "WEEK"], sort=True):
            season, week = int(season), int(week)  # noqa: PLW2901
            if previous_season is not None and season != previous_season:
                state = self._transition(state, season)
            previous_season = season

            results: dict[str, list[_Result]] = {}
            for away, home, away_score, home_score in zip(
                week_games["AWAY_TEAM"].astype(str),
                week_games["HOME_TEAM"].astype(str),
                week_games["AWAY_SCORE"].to_numpy(dtype=float),
                week_games["HOME_SCORE"].to_numpy(dtype=float),
                strict=True,
            ):
                for team in (away, home):
                    state.setdefault(team, self._default(team, season))
                a_mu, a_phi, _ = state[away]
                h_mu, h_phi, _ = state[home]
                score = away_outcome(float(away_score), float(home_score))
                results.setdefault(away, []).append((h_mu, h_phi, score))
                results.setdefault(home, []).append((a_mu, a_phi, 1.0 - score))

            state = {
                team: glicko2_step(mu, phi, sigma, results.get(team, []), tau=cfg.tau)
                for team, (mu, phi, sigma) in state.items()
            }
            self._keys.append((season, week))
            self._after.append(state)

    def state_as_of(
        self,
        season: int,
        week: int,
        teams: Iterable[str],
    ) -> dict[str, Glicko2State]:
        """Return each team's state before ``(season, week)``'s games.

        Reflects only completed games with a strictly earlier (season, week).

        Raises:
            ValueError: If ``season`` is more than one season past the last
                completed game (the offseason transition is defined once).
        """
        index = bisect_left(self._keys, (season, week)) - 1
        if index < 0:
            state: dict[str, _Internal] = {}
        else:
            state = self._after[index]
            last_season = self._keys[index][0]
            if season > last_season + 1:
                raise ValueError(
                    f"Cannot look ahead to season {season_label(season)}: "
                    f"last completed game is in {season_label(last_season)}."
                )
            if season > last_season:
                state = self._transition(state, season)
        return {
            team: _public(state[team] if team in state else self._default(team, season))
            for team in teams
        }
