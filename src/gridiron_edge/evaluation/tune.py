"""Elo parameter grid search - flat K and zone-K variants.

Finds the combination of parameters that minimises Brier score on a
held-out set of seasons. Two search modes are supported:

  flat_k (one K-factor for every week):
    Parameters: k, divisor, regress_frac
    100 combinations (5 x 5 x 4)

  zone_k (K varies by week zone):
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

# pyrefly: ignore [untyped-import]
from tqdm import tqdm

from gridiron_edge.core.constants import EXPANSION_TEAMS as _EXPANSION_START
from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

# Imported from core.constants - single source of truth shared with _shared.py.
# Update HOLDOUT_SEASONS in core/constants.py at the start of each new season.

# ---------------------------------------------------------------------------
# Flat K grid (one K-factor for every week)
# ---------------------------------------------------------------------------

K_VALUES: Final[list[float]] = [15.0, 20.0, 25.0, 32.0, 40.0]
DIVISOR_VALUES: Final[list[float]] = [350.0, 400.0, 450.0, 480.0, 550.0]
REGRESS_VALUES: Final[list[float]] = [0.2, 0.33, 0.4, 0.5]

# ---------------------------------------------------------------------------
# Zone K grid (K varies by week zone) - comprehensive
# ---------------------------------------------------------------------------

# All four K zones use the same symmetric range - no prior assumptions about
# which zones should be higher or lower. The data determines the direction.
# k_week18 extends down to 0 since near-zero K is theoretically sound for
# a structurally noisy week (starter rest, locked seedings).
#
# Grid size: 6^3 * 7 * 7 * 6 = 63,504 combinations ~ 7.9h at 0.45s each.
# Divisor extends down to 280 and up to 520 to ensure the optimum is not
# at a boundary - flat K found 350 winning with the previous floor at 350.
# Regress extends down to 0.1 to test near-zero regression.
K_EARLY_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
K_MID_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
K_WEEK18_VALUES: Final[list[float]] = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
K_POST_VALUES: Final[list[float]] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
DIVISOR_VALUES_ZONE_K: Final[list[float]] = [280.0, 320.0, 360.0, 400.0, 440.0, 480.0, 520.0]
REGRESS_VALUES_ZONE_K: Final[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

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
    """Result for a single flat-K parameter combination."""

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
class TuneResultZoneK:
    """Result for a single zone-K parameter combination."""

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


def _simulate_and_score(
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
    """Tuner-shaped view over :func:`simulate_elo_history`.

    Kept as the public callable consumed by ``models/elo/model.py``
    and the grid-search loops below. The actual Elo simulation is
    delegated to the canonical simulator.
    """
    from gridiron_edge.ratings.elo.simulator import simulate_elo_history

    result = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start,
        k_early=k_early,
        k_mid=k_mid,
        k_week18=k_week18,
        k_post=k_post,
        divisor=divisor,
        regress_frac=regress_frac,
        initial_elo=initial_elo,
        expansion_elo=expansion_elo,
    )
    return (
        result.away_probs,
        result.away_outcomes,
        result.game_seasons,
        result.game_ids,
    )


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


def _prepare_games(
    games: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, set[str]],
]:
    """Clean canonical games and index participating teams by season.

    Args:
        games: Canonical games DataFrame containing completed-game state,
            season, Away Team, and Home Team.

    Returns:
        Completed games, sorted season labels, and the set of canonical
        Away and Home teams participating in each season.

    Raises:
        ValueError: If required canonical game columns are unavailable.
    """
    required: set[str] = {
        "YEAR",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_SCORE",
        "HOME_SCORE",
    }
    missing: list[str] = sorted(required - set(games.columns))
    if missing:
        raise ValueError(
            "Canonical Elo tuning games are missing required columns: " + ", ".join(missing)
        )

    completed_mask = games["AWAY_SCORE"].notna() & games["HOME_SCORE"].notna()
    df = games.loc[completed_mask, :].copy()

    df["AWAY_TEAM"] = df["AWAY_TEAM"].astype(str).str.strip()
    df["HOME_TEAM"] = df["HOME_TEAM"].astype(str).str.strip()

    invalid_identity = df["AWAY_TEAM"].eq("") | df["HOME_TEAM"].eq("")
    if invalid_identity.any():
        raise ValueError("Canonical Elo tuning games contain empty team identities.")

    sorted_years: list[str] = sorted(df["YEAR"].astype(str).unique().tolist())

    teams_by_year: dict[str, set[str]] = {}
    for year in sorted_years:
        year_games = df.loc[
            df["YEAR"].astype(str) == year,
            :,
        ]
        teams_by_year[year] = set(year_games["AWAY_TEAM"].tolist()) | set(
            year_games["HOME_TEAM"].tolist()
        )

    return (
        df,
        sorted_years,
        teams_by_year,
    )


# ---------------------------------------------------------------------------
# Flat K grid search
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
    """Run the flat-K parameter grid search.

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
    logger.info("flat-K grid search: %d combinations  holdout=%s", n_total, sorted(holdout))

    results: list[TuneResult] = []
    best_so_far = float("inf")

    bar = tqdm(grid, desc="flat-K tune", unit="combo", ncols=80)
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
        "Best flat-K: k=%.0f  divisor=%.0f  regress=%.2f  holdout=%.5f  gap=%.5f",
        best["k"],
        best["divisor"],
        best["regress_frac"],
        best["holdout_brier"],
        best["overfit_gap"],
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(save_path, index=False)
        logger.info("flat-K results saved to %s", save_path)
    return df_results


# ---------------------------------------------------------------------------
# Zone K grid search
# ---------------------------------------------------------------------------


def run_grid_search_zone_k(
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
    """Run the zone-K parameter grid search.

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
        divisor_values: Divisor values. Defaults to ``DIVISOR_VALUES_ZONE_K``.
        regress_values: Regression fractions. Defaults to ``REGRESS_VALUES_ZONE_K``.
        holdout_seasons: Seasons held out. Defaults to ``HOLDOUT_SEASONS``.
        save_path: If provided, write the full results DataFrame to this
            path as Parquet after the search completes.

    Returns:
        DataFrame of ``TuneResultZoneK`` rows sorted by ``holdout_brier`` ascending.
    """
    resolved_repo = repo or repo_root()
    games_raw = loaders.load_games(resolved_repo)
    games, sorted_years, teams_by_year = _prepare_games(games_raw)

    ke_vals = k_early_values or K_EARLY_VALUES
    km_vals = k_mid_values or K_MID_VALUES
    kw_vals = k_week18_values or K_WEEK18_VALUES
    kp_vals = k_post_values or K_POST_VALUES
    divs = divisor_values or DIVISOR_VALUES_ZONE_K
    regs = regress_values or REGRESS_VALUES_ZONE_K
    holdout = holdout_seasons or HOLDOUT_SEASONS

    grid = list(itertools.product(ke_vals, km_vals, kw_vals, kp_vals, divs, regs))
    n_total = len(grid)
    logger.info("zone-K grid search: %d combinations  holdout=%s", n_total, sorted(holdout))

    results: list[TuneResultZoneK] = []
    best_so_far = float("inf")

    bar = tqdm(grid, desc="zone-K tune", unit="combo", ncols=80)
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
            TuneResultZoneK(
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
        "Best zone-K: k_early=%.0f k_mid=%.0f k_w18=%.0f k_post=%.0f "
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
        logger.info("zone-K results saved to %s", save_path)
    return df_results


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------


def best_params(results: pd.DataFrame) -> dict[str, float]:
    """Extract best flat-K parameters from grid search results.

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


def best_params_zone_k(results: pd.DataFrame) -> dict[str, float]:
    """Extract best zone-K parameters from grid search results.

    Args:
        results: Output of ``run_grid_search_zone_k()``.

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
