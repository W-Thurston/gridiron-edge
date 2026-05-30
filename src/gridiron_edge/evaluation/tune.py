# src/gridiron_edge/evaluation/tune.py

"""Elo parameter grid search - elo_v2 (flat K) and elo_v3 (zone-based K).

Finds the combination of parameters that minimises Brier score on a
held-out set of seasons. Two search modes are supported:

  elo_v2 (flat K):
    Parameters: k, divisor, regress_frac
    100 combinations (5 x 5 x 4)

  elo_v3 (zone-based K):
    Parameters: k_early, k_mid, k_week18, k_post, divisor, regress_frac
    K varies by week zone - early season, mid season, week 18, postseason.

The search engine uses a dict-based Elo simulation (~50x faster than the
production DataFrame-based table builder) that replicates the exact same
algorithm: same expansion team handling, same offseason regression, same
bye-week forward-fill.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
from logging import Logger
from pathlib import Path
import time
from typing import Final

import pandas as pd
from tqdm import tqdm

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.core.constants import EXPANSION_TEAMS as _EXPANSION_START
from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders
from gridiron_edge.ratings.elo.core import elo_win_probability as _elo_win_probability

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

# Imported from core.constants — single source of truth shared with _shared.py.
# Update HOLDOUT_SEASONS in core/constants.py at the start of each new season.

# ---------------------------------------------------------------------------
# elo_v2 grid (flat K)
# ---------------------------------------------------------------------------

K_VALUES: Final[list[float]] = [15.0, 20.0, 25.0, 32.0, 40.0]
DIVISOR_VALUES: Final[list[float]] = [350.0, 400.0, 450.0, 480.0, 550.0]
REGRESS_VALUES: Final[list[float]] = [0.2, 0.33, 0.4, 0.5]

# ---------------------------------------------------------------------------
# elo_v3 grid (zone-based K) — comprehensive
# ---------------------------------------------------------------------------

# All four K zones use the same symmetric range — no prior assumptions about
# which zones should be higher or lower. The data determines the direction.
# k_week18 extends down to 0 since near-zero K is theoretically sound for
# a structurally noisy week (starter rest, locked seedings).
#
# Grid size: 6^3 * 7 * 7 * 6 = 63,504 combinations ~ 7.9h at 0.45s each.
# Divisor extends down to 280 and up to 520 to ensure the optimum is not
# at a boundary — elo_v2 found 350 winning with the previous floor at 350.
# Regress extends down to 0.1 to test near-zero regression.
K_EARLY_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
K_MID_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
K_WEEK18_VALUES: Final[list[float]] = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
K_POST_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
DIVISOR_VALUES_V3: Final[list[float]] = [280.0, 320.0, 360.0, 400.0, 440.0, 480.0, 520.0]
REGRESS_VALUES_V3: Final[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Week zone boundaries
_WEEKS_EARLY: Final[frozenset[int]] = frozenset(range(1, 5))
_WEEKS_MID: Final[frozenset[int]] = frozenset(range(5, 18))
_WEEK_18: Final[frozenset[int]] = frozenset([18])
_WEEKS_POST: Final[frozenset[int]] = frozenset(range(19, 23))

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

# _EXPANSION_START is imported from core.constants as EXPANSION_TEAMS.
# See core/constants.py for the canonical expansion franchise start seasons.

_VALID_WEEKS: Final[frozenset[int]] = frozenset(range(1, 23))
_INITIAL_ELO: Final[float] = 1500.0
_EXPANSION_ELO: Final[float] = 1300.0

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneResult:
    """Result for a single elo_v2 flat-K parameter combination."""

    k: float
    divisor: float
    regress_frac: float
    train_brier: float
    holdout_brier: float
    overfit_gap: float
    train_games: int
    holdout_games: int
    elapsed_s: float


@dataclass(frozen=True)
class TuneResultV3:
    """Result for a single elo_v3 zone-based K parameter combination."""

    k_early: float
    k_mid: float
    k_week18: float
    k_post: float
    divisor: float
    regress_frac: float
    train_brier: float
    holdout_brier: float
    overfit_gap: float
    train_games: int
    holdout_games: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Fast Elo engine
# ---------------------------------------------------------------------------


def _k_for_week(
    week: int,
    k_early: float,
    k_mid: float,
    k_week18: float,
    k_post: float,
) -> float:
    """Return the K-factor for the given week number."""
    if week in _WEEKS_EARLY:
        return k_early
    if week in _WEEKS_MID:
        return k_mid
    if week in _WEEK_18:
        return k_week18
    if week in _WEEKS_POST:
        return k_post
    return k_mid


def _simulate_and_score(  # noqa: PLR0912, PLR0915
    games: pd.DataFrame,
    sorted_years: list[str],
    teams_by_year: dict[str, set[str]],
    expansion_start: dict[str, str],
    *,
    k_early: float,
    k_mid: float,
    k_week18: float,
    k_post: float,
    divisor: float,
    regress_frac: float,
    initial_elo: float = _INITIAL_ELO,
    expansion_elo: float = _EXPANSION_ELO,
) -> tuple[list[float], list[float], list[str], list[str]]:
    """Run the full Elo simulation and return per-game predictions.

    Accepts zone-based K parameters. For a flat-K simulation, pass the
    same value for k_early, k_mid, k_week18, and k_post.

    Args:
        games: Canonical games DataFrame.
        sorted_years: Chronologically ordered season labels.
        teams_by_year: Season label to set of active team names.
        expansion_start: Team name to first active season.
        k_early: K-factor for weeks 1-4.
        k_mid: K-factor for weeks 5-17.
        k_week18: K-factor for week 18.
        k_post: K-factor for weeks 19-22.
        divisor: Win-probability divisor.
        regress_frac: Offseason regression fraction.
        initial_elo: Starting Elo for all teams in year 0, week 1.
        expansion_elo: Starting Elo for expansion teams.

    Returns:
        Tuple of (away_probs, away_outcomes, seasons, game_ids).
    """
    elo: dict[tuple[str, str, int], float] = {}

    first_year = sorted_years[0]
    for team in teams_by_year.get(first_year, set()):
        elo[(team, first_year, 1)] = initial_elo

    games_idx = games.groupby(["YEAR", "WEEK_NUM"])

    away_probs: list[float] = []
    away_outcomes: list[float] = []
    game_seasons: list[str] = []
    game_ids: list[str] = []

    for yr_idx, curr_year in enumerate(sorted_years):
        next_year = sorted_years[yr_idx + 1] if yr_idx < len(sorted_years) - 1 else None
        teams_this_season = teams_by_year.get(curr_year, set())

        try:
            season_games = games.loc[games["YEAR"] == curr_year]
        except KeyError:
            continue

        weeks_with_games = sorted(season_games["WEEK_NUM"].unique().tolist())
        if not weeks_with_games:
            continue
        max_week = max(weeks_with_games)

        for wk in range(1, max_week + 1):
            k = _k_for_week(wk, k_early, k_mid, k_week18, k_post)

            try:
                week_df = games_idx.get_group((curr_year, wk))
            except KeyError:
                week_df = pd.DataFrame()

            for _, row in week_df.iterrows():
                winner = str(row["WINNER"])
                loser = str(row["LOSER"])
                win_or_tie = float(row["WIN_OR_TIE"])

                w_elo = elo.get((winner, curr_year, wk), initial_elo)
                l_elo = elo.get((loser, curr_year, wk), initial_elo)

                away_team = winner if row["GAME_LOCATION"] == _AWAY_WIN_LOCATION else loser
                away_elo = w_elo if away_team == winner else l_elo
                home_elo = l_elo if away_team == winner else w_elo

                if wk in _VALID_WEEKS:
                    away_prob, _ = _elo_win_probability(away_elo, home_elo, divisor=divisor)
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

                p_win, _ = _elo_win_probability(w_elo, l_elo, divisor=divisor)
                score_w = win_or_tie
                delta = k * (score_w - p_win)
                new_w = w_elo + delta
                new_l = l_elo - delta

                is_last_week = wk == max_week
                if is_last_week and next_year is not None:
                    elo[(winner, next_year, 1)] = new_w
                    elo[(loser, next_year, 1)] = new_l
                else:
                    elo[(winner, curr_year, wk + 1)] = new_w
                    elo[(loser, curr_year, wk + 1)] = new_l

            all_teams_this_year = teams_by_year.get(curr_year, set())
            is_last_week = wk == max_week

            for team in all_teams_this_year:
                curr_key = (team, curr_year, wk)
                if is_last_week and next_year is not None:
                    next_key = (team, next_year, 1)
                    if next_key not in elo:
                        elo[next_key] = elo.get(curr_key, initial_elo)
                elif not is_last_week:
                    next_key = (team, curr_year, wk + 1)
                    if next_key not in elo:
                        elo[next_key] = elo.get(curr_key, initial_elo)

        if next_year is not None:
            returning = teams_this_season & teams_by_year.get(next_year, set())
            next_elos = [elo.get((t, next_year, 1), initial_elo) for t in returning]
            if next_elos:
                season_mean = sum(next_elos) / len(next_elos)
                for team in returning:
                    key = (team, next_year, 1)
                    current = elo.get(key, initial_elo)
                    elo[key] = season_mean * regress_frac + current * (1.0 - regress_frac)

            for team, start in expansion_start.items():
                if start == next_year:
                    elo[(team, next_year, 1)] = expansion_elo

    return away_probs, away_outcomes, game_seasons, game_ids


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _brier(probs: list[float], outcomes: list[float]) -> float:
    """Brier score for a list of predictions and outcomes."""
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes, strict=False)) / len(probs)


def _split_train_holdout(
    away_probs: list[float],
    away_outcomes: list[float],
    game_seasons: list[str],
    holdout: frozenset[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Split predictions into train and holdout lists."""
    train_p: list[float] = []
    train_o: list[float] = []
    holdout_p: list[float] = []
    holdout_o: list[float] = []
    for prob, outcome, season in zip(away_probs, away_outcomes, game_seasons, strict=False):
        if season in holdout:
            holdout_p.append(prob)
            holdout_o.append(outcome)
        else:
            train_p.append(prob)
            train_o.append(outcome)
    return train_p, train_o, holdout_p, holdout_o


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_games(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, set[str]]]:
    """Clean and index the games DataFrame for the simulator."""
    df = games.copy()
    df = df.loc[df["WIN_OR_TIE"].notna()].copy()
    df["WINNER"] = df["WINNER"].astype(str)
    df["LOSER"] = df["LOSER"].astype(str)

    sorted_years = sorted(df["YEAR"].unique().tolist())

    teams_by_year: dict[str, set[str]] = {}
    for year in sorted_years:
        yr_df = df.loc[df["YEAR"] == year]
        teams_by_year[year] = set(yr_df["WINNER"].tolist()) | set(yr_df["LOSER"].tolist())

    return df, sorted_years, teams_by_year


# ---------------------------------------------------------------------------
# elo_v2 grid search (flat K)
# ---------------------------------------------------------------------------


def run_grid_search(
    *,
    repo: Path | None = None,
    k_values: list[float] | None = None,
    divisor_values: list[float] | None = None,
    regress_values: list[float] | None = None,
    holdout_seasons: frozenset[str] | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Run the elo_v2 flat-K parameter grid search.

    Args:
        repo: Repository root. Defaults to ``repo_root()``.
        k_values: K-factor values to search. Defaults to ``K_VALUES``.
        divisor_values: Divisor values. Defaults to ``DIVISOR_VALUES``.
        regress_values: Regression fractions. Defaults to ``REGRESS_VALUES``.
        holdout_seasons: Seasons held out. Defaults to ``HOLDOUT_SEASONS``.
        save_path: If provided, write the full results DataFrame to this
            path as Parquet after the search completes. Useful for long
            runs where terminal output may be lost.

    Returns:
        DataFrame of ``TuneResult`` rows sorted by ``holdout_brier`` ascending.
    """
    resolved_repo = repo or repo_root()
    games_raw = loaders.load_games(resolved_repo)
    games, sorted_years, teams_by_year = _prepare_games(games_raw)

    ks = k_values or K_VALUES
    divs = divisor_values or DIVISOR_VALUES
    regs = regress_values or REGRESS_VALUES
    holdout = holdout_seasons or HOLDOUT_SEASONS

    grid = list(itertools.product(ks, divs, regs))
    n_total = len(grid)
    logger.info("elo_v2 grid search: %d combinations  holdout=%s", n_total, sorted(holdout))

    results: list[TuneResult] = []
    best_so_far = float("inf")

    bar = tqdm(grid, desc="elo_v2 tune", unit="combo", ncols=80)
    for k, divisor, regress_frac in bar:
        t0 = time.perf_counter()

        away_probs, away_outcomes, game_seasons, _ = _simulate_and_score(
            games,
            sorted_years,
            teams_by_year,
            _EXPANSION_START,
            k_early=k,
            k_mid=k,
            k_week18=k,
            k_post=k,
            divisor=divisor,
            regress_frac=regress_frac,
        )

        train_p, train_o, holdout_p, holdout_o = _split_train_holdout(
            away_probs, away_outcomes, game_seasons, holdout
        )

        train_brier = _brier(train_p, train_o)
        holdout_brier = _brier(holdout_p, holdout_o)
        elapsed = time.perf_counter() - t0

        best_so_far = min(best_so_far, holdout_brier)

        bar.set_postfix(best=f"{best_so_far:.5f}", refresh=False)

        results.append(
            TuneResult(
                k=k,
                divisor=divisor,
                regress_frac=regress_frac,
                train_brier=round(train_brier, 6),
                holdout_brier=round(holdout_brier, 6),
                overfit_gap=round(holdout_brier - train_brier, 6),
                train_games=len(train_p),
                holdout_games=len(holdout_p),
                elapsed_s=round(elapsed, 2),
            )
        )

    df_results = (
        pd.DataFrame([vars(r) for r in results]).sort_values("holdout_brier").reset_index(drop=True)
    )
    best = df_results.iloc[0]
    logger.info(
        "Best elo_v2: k=%.0f  divisor=%.0f  regress=%.2f  holdout=%.5f  gap=%.5f",
        best["k"],
        best["divisor"],
        best["regress_frac"],
        best["holdout_brier"],
        best["overfit_gap"],
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(save_path, index=False)
        logger.info("elo_v2 results saved to %s", save_path)
    return df_results


# ---------------------------------------------------------------------------
# elo_v3 grid search (zone-based K)
# ---------------------------------------------------------------------------


def run_grid_search_v3(
    *,
    repo: Path | None = None,
    k_early_values: list[float] | None = None,
    k_mid_values: list[float] | None = None,
    k_week18_values: list[float] | None = None,
    k_post_values: list[float] | None = None,
    divisor_values: list[float] | None = None,
    regress_values: list[float] | None = None,
    holdout_seasons: frozenset[str] | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Run the elo_v3 zone-based K parameter grid search.

    Searches all combinations of four zone K-factors alongside divisor and
    regression fraction. Full search is 4 x 3 x 4 x 4 x 6 x 4 = 4,608
    combinations. At ~0.45s each this takes roughly 35 minutes. Consider
    narrowing any axis via the keyword arguments if you want a faster run.

    Args:
        repo: Repository root.
        k_early_values: K for weeks 1-4. Defaults to ``K_EARLY_VALUES``.
        k_mid_values: K for weeks 5-17. Defaults to ``K_MID_VALUES``.
        k_week18_values: K for week 18. Defaults to ``K_WEEK18_VALUES``.
        k_post_values: K for weeks 19-22. Defaults to ``K_POST_VALUES``.
        divisor_values: Divisor values. Defaults to ``DIVISOR_VALUES_V3``.
        regress_values: Regression fractions. Defaults to ``REGRESS_VALUES_V3``.
        holdout_seasons: Seasons held out. Defaults to ``HOLDOUT_SEASONS``.
        save_path: If provided, write the full results DataFrame to this
            path as Parquet after the search completes.

    Returns:
        DataFrame of ``TuneResultV3`` rows sorted by ``holdout_brier`` ascending.
    """
    resolved_repo = repo or repo_root()
    games_raw = loaders.load_games(resolved_repo)
    games, sorted_years, teams_by_year = _prepare_games(games_raw)

    ke_vals = k_early_values or K_EARLY_VALUES
    km_vals = k_mid_values or K_MID_VALUES
    kw_vals = k_week18_values or K_WEEK18_VALUES
    kp_vals = k_post_values or K_POST_VALUES
    divs = divisor_values or DIVISOR_VALUES_V3
    regs = regress_values or REGRESS_VALUES_V3
    holdout = holdout_seasons or HOLDOUT_SEASONS

    grid = list(itertools.product(ke_vals, km_vals, kw_vals, kp_vals, divs, regs))
    n_total = len(grid)
    logger.info("elo_v3 grid search: %d combinations  holdout=%s", n_total, sorted(holdout))

    results: list[TuneResultV3] = []
    best_so_far = float("inf")

    bar = tqdm(grid, desc="elo_v3 tune", unit="combo", ncols=80)
    for k_early, k_mid, k_week18, k_post, divisor, regress_frac in bar:
        t0 = time.perf_counter()

        away_probs, away_outcomes, game_seasons, _ = _simulate_and_score(
            games,
            sorted_years,
            teams_by_year,
            _EXPANSION_START,
            k_early=k_early,
            k_mid=k_mid,
            k_week18=k_week18,
            k_post=k_post,
            divisor=divisor,
            regress_frac=regress_frac,
        )

        train_p, train_o, holdout_p, holdout_o = _split_train_holdout(
            away_probs, away_outcomes, game_seasons, holdout
        )

        train_brier = _brier(train_p, train_o)
        holdout_brier = _brier(holdout_p, holdout_o)
        elapsed = time.perf_counter() - t0

        best_so_far = min(best_so_far, holdout_brier)

        bar.set_postfix(best=f"{best_so_far:.5f}", refresh=False)

        results.append(
            TuneResultV3(
                k_early=k_early,
                k_mid=k_mid,
                k_week18=k_week18,
                k_post=k_post,
                divisor=divisor,
                regress_frac=regress_frac,
                train_brier=round(train_brier, 6),
                holdout_brier=round(holdout_brier, 6),
                overfit_gap=round(holdout_brier - train_brier, 6),
                train_games=len(train_p),
                holdout_games=len(holdout_p),
                elapsed_s=round(elapsed, 2),
            )
        )

    df_results = (
        pd.DataFrame([vars(r) for r in results]).sort_values("holdout_brier").reset_index(drop=True)
    )
    best = df_results.iloc[0]
    logger.info(
        "Best elo_v3: k_early=%.0f k_mid=%.0f k_w18=%.0f k_post=%.0f "
        "divisor=%.0f regress=%.2f  holdout=%.5f  gap=%.5f",
        best["k_early"],
        best["k_mid"],
        best["k_week18"],
        best["k_post"],
        best["divisor"],
        best["regress_frac"],
        best["holdout_brier"],
        best["overfit_gap"],
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(save_path, index=False)
        logger.info("elo_v3 results saved to %s", save_path)
    return df_results


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------


def best_params(results: pd.DataFrame) -> dict[str, float]:
    """Extract best flat-K parameters from elo_v2 grid search results.

    Args:
        results: Output of ``run_grid_search()``.

    Returns:
        Dict with keys ``k``, ``divisor``, ``regress_frac``.
    """
    row = results.iloc[0]
    return {
        "k": float(row["k"]),
        "divisor": float(row["divisor"]),
        "regress_frac": float(row["regress_frac"]),
    }


def best_params_v3(results: pd.DataFrame) -> dict[str, float]:
    """Extract best zone-based K parameters from elo_v3 grid search results.

    Args:
        results: Output of ``run_grid_search_v3()``.

    Returns:
        Dict with keys ``k_early``, ``k_mid``, ``k_week18``, ``k_post``,
        ``divisor``, ``regress_frac``.
    """
    row = results.iloc[0]
    return {
        "k_early": float(row["k_early"]),
        "k_mid": float(row["k_mid"]),
        "k_week18": float(row["k_week18"]),
        "k_post": float(row["k_post"]),
        "divisor": float(row["divisor"]),
        "regress_frac": float(row["regress_frac"]),
    }
