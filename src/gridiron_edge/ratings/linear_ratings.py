# src/gridiron_edge/ratings/linear_ratings.py

"""Massey, Colley, and Keener ratings recomputed per (season, week) cutoff.

All three are linear-algebra rankings with no probability model. They are pure
functions of the completed games in a window, so nothing is persisted and no
lineage is needed (``DECISIONS.md`` D69).

Window for a cutoff: completed games of the target season strictly before the
target week at weight 1, plus every game of the prior season at
``prior_season_weight``. A week beyond the last completed game therefore sees
the same games as the next week. A team with no game in the window gets the
neutral value of each method (Massey 0, Colley 0.5, Keener 1.0), so a
season's first week and an expansion team are never null.

- Massey: ridge-regularised least squares on capped point margins.
- Colley: weighted Colley matrix on wins and losses only; ties count half.
- Keener: Perron eigenvector of the smoothed, skewed, games-normalised score
  matrix, scaled to a league mean of 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from gridiron_edge.ratings.history import prepare_history

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray
    from pandas import DataFrame

LINEAR_RATING_COLUMNS: Final[tuple[str, str, str]] = ("MASSEY", "COLLEY", "KEENER")
NEUTRAL_RATINGS: Final[dict[str, float]] = {"MASSEY": 0.0, "COLLEY": 0.5, "KEENER": 1.0}

_POWER_ITERATIONS: Final[int] = 2000
_POWER_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True)
class LinearRatingConfig:
    """Window weighting and regularisation, fit on 1999-2022 only (``DECISIONS.md`` D70)."""

    prior_season_weight: float = 0.25
    margin_cap: float = 21.0
    massey_ridge: float = 4.0
    keener_floor: float = 1e-3


def _massey(
    away: NDArray[np.integer],
    home: NDArray[np.integer],
    margin: NDArray[np.floating],
    weight: NDArray[np.floating],
    *,
    n_teams: int,
    cfg: LinearRatingConfig,
) -> NDArray[np.floating]:
    capped = np.clip(margin, -cfg.margin_cap, cfg.margin_cap)
    normal = cfg.massey_ridge * np.eye(n_teams)
    rhs = np.zeros(n_teams)
    np.add.at(normal, (away, away), weight)
    np.add.at(normal, (home, home), weight)
    np.add.at(normal, (away, home), -weight)
    np.add.at(normal, (home, away), -weight)
    np.add.at(rhs, away, weight * capped)
    np.add.at(rhs, home, -weight * capped)
    return np.linalg.solve(normal, rhs)


def _colley(
    away: NDArray[np.integer],
    home: NDArray[np.integer],
    away_result: NDArray[np.floating],
    weight: NDArray[np.floating],
    n_teams: int,
) -> NDArray[np.floating]:
    matrix = 2.0 * np.eye(n_teams)
    rhs = np.ones(n_teams)
    np.add.at(matrix, (away, away), weight)
    np.add.at(matrix, (home, home), weight)
    np.add.at(matrix, (away, home), -weight)
    np.add.at(matrix, (home, away), -weight)
    np.add.at(rhs, away, weight * (away_result - 0.5))
    np.add.at(rhs, home, weight * (0.5 - away_result))
    return np.linalg.solve(matrix, rhs)


def _keener_skew(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return 0.5 + np.sign(x - 0.5) * np.sqrt(np.abs(2.0 * x - 1.0)) / 2.0


def _keener(
    away: NDArray[np.integer],
    home: NDArray[np.integer],
    away_points: NDArray[np.floating],
    home_points: NDArray[np.floating],
    weight: NDArray[np.floating],
    *,
    n_teams: int,
    cfg: LinearRatingConfig,
) -> NDArray[np.floating]:
    scored = np.zeros((n_teams, n_teams))
    np.add.at(scored, (away, home), weight * away_points)
    np.add.at(scored, (home, away), weight * home_points)
    games = np.zeros(n_teams)
    np.add.at(games, away, weight)
    np.add.at(games, home, weight)

    played = (scored + scored.T) > 0.0
    share = (scored + 1.0) / (scored + scored.T + 2.0)
    strength = np.where(played, _keener_skew(share), 0.0)
    strength = strength / np.maximum(games, 1.0)[:, None]
    strength = strength + cfg.keener_floor * (1.0 - np.eye(n_teams))
    # Adding a multiple of the identity keeps the Perron vector but makes the
    # iteration aperiodic (a two-team matrix has eigenvalues +/- lambda).
    strength = strength + strength.max() * np.eye(n_teams)

    vector = np.ones(n_teams)
    for _ in range(_POWER_ITERATIONS):
        updated = strength @ vector
        updated = updated / updated.sum()
        if np.abs(updated - vector / vector.sum()).max() < _POWER_TOLERANCE:
            vector = updated
            break
        vector = updated
    return vector * n_teams / vector.sum()


def solve_linear_ratings(window: DataFrame, cfg: LinearRatingConfig) -> DataFrame:
    """Rate every team appearing in a weighted game window.

    Args:
        window: Games with ``AWAY_TEAM``, ``HOME_TEAM``, ``AWAY_SCORE``,
            ``HOME_SCORE`` and a positive ``WEIGHT``.
        cfg: Regularisation and margin-cap parameters.

    Returns:
        One row per team with ``MASSEY``, ``COLLEY`` and ``KEENER``.
    """
    teams = sorted(set(window["AWAY_TEAM"]) | set(window["HOME_TEAM"]))
    index = {team: i for i, team in enumerate(teams)}
    away = window["AWAY_TEAM"].map(index).to_numpy(dtype=np.intp)
    home = window["HOME_TEAM"].map(index).to_numpy(dtype=np.intp)
    away_points = window["AWAY_SCORE"].to_numpy(dtype=float)
    home_points = window["HOME_SCORE"].to_numpy(dtype=float)
    weight = window["WEIGHT"].to_numpy(dtype=float)
    away_result = np.where(
        away_points > home_points, 1.0, np.where(away_points == home_points, 0.5, 0.0)
    )
    n_teams = len(teams)

    return pd.DataFrame(
        {
            "MASSEY": _massey(
                away, home, away_points - home_points, weight, n_teams=n_teams, cfg=cfg
            ),
            "COLLEY": _colley(away, home, away_result, weight, n_teams),
            "KEENER": _keener(
                away, home, away_points, home_points, weight, n_teams=n_teams, cfg=cfg
            ),
        },
        index=pd.Index(teams, name="TEAM"),
    )


class LinearRatings:
    """Massey/Colley/Keener ratings as of a (season, week) cutoff."""

    def __init__(self, games: DataFrame, config: LinearRatingConfig | None = None) -> None:
        self._cfg = config or LinearRatingConfig()
        self._history = prepare_history(games)

    def as_of(self, season: int, week: int, teams: Iterable[str]) -> DataFrame:
        """Rate ``teams`` using only completed games before ``(season, week)``.

        Teams with no game in the window get the neutral value of each method.
        """
        history = self._history
        current = (history["SEASON"] == season) & (history["WEEK"] < week)
        prior = history["SEASON"] == season - 1
        weight = np.where(current, 1.0, np.where(prior, self._cfg.prior_season_weight, 0.0))
        window = history.loc[weight > 0.0].assign(WEIGHT=weight[weight > 0.0])

        requested = list(dict.fromkeys(teams))
        neutral = pd.DataFrame(
            {
                column: [NEUTRAL_RATINGS[column]] * len(requested)
                for column in LINEAR_RATING_COLUMNS
            },
            index=pd.Index(requested, name="TEAM"),
        )
        if window.empty:
            return neutral
        rated = solve_linear_ratings(window, self._cfg)
        return rated.reindex(requested).fillna(neutral)
