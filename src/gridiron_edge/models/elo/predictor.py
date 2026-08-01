# src/gridiron_edge/models/elo/predictor.py

"""Elo-based predictor implementation.

Implements the ``Predictor`` protocol for the Elo win-probability model.
Registered under the composite model key ``"win_prob_elo"`` with
``model_name="win_prob"`` and ``model_type="elo"``.

Parameters use the production defaults (K=20, divisor=480, regress=1/3)
which match the Elo state table on disk. Tuned hyperparameter sets from
``gridiron evaluate tune`` are intended to inform future fits, not to
diverge from the production Elo state table at predict time.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import ClassVar, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.models.base import ModelSpec
from gridiron_edge.models.game_prediction.post_process import enrich_predictions
from gridiron_edge.models.registry import ModelRegistry

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_elo_predictions(
    away_probs: list[float],
    game_seasons: list[str],
    game_ids: list[str],
    games: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
) -> pd.DataFrame:
    """Convert Elo simulation output into canonical prediction rows.

    The simulation returns game identity and away-team win probability.
    This function restores team orientation and game dates from the
    historical games dataset, then applies pure prediction enrichment.

    Args:
        away_probs: Predicted away-team win probability per game.
        game_seasons: Season label per game.
        game_ids: Canonical GAME_ID per game.
        games: Full games DataFrame (used to look up team names and dates).
        model_name: Model purpose (always ``"win_prob"`` for the Elo predictor).
        model_type: Model algorithm (always ``"elo"`` for the Elo predictor).

    Returns:
        Canonical game-level Elo prediction rows with derived enrichment.
    """
    games_indexed = games.set_index("GAME_ID")
    away_teams: list[str] = []
    home_teams: list[str] = []
    game_dates: list[str] = []

    for gid in game_ids:
        try:
            row = games_indexed.loc[gid]
            away_won = row["GAME_LOCATION"] == _AWAY_WIN_LOCATION
            away_teams.append(str(row["WINNER"]) if away_won else str(row["LOSER"]))
            home_teams.append(str(row["LOSER"]) if away_won else str(row["WINNER"]))
            game_dates.append(str(row.get("GAME_DATE", "")))
        except KeyError:
            away_teams.append("")
            home_teams.append("")
            game_dates.append("")

    weeks: dict = games.set_index("GAME_ID")["WEEK_NUM"].to_dict()

    result = pd.DataFrame(
        {
            "season": game_seasons,
            "week": [int(weeks.get(gid, 0)) for gid in game_ids],
            "game_id": game_ids,
            "game_date": game_dates,
            "away_team": away_teams,
            "home_team": home_teams,
            "away_elo": float("nan"),
            "home_elo": float("nan"),
            "away_win_prob": away_probs,
            "home_win_prob": [1.0 - probability for probability in away_probs],
        }
    )

    result: DataFrame = enrich_predictions(
        result,
        model_name=model_name,
        model_type=model_type,
        recalibrate=False,
    )

    return result


def _run_simulation(
    games: pd.DataFrame,
    *,
    k_early: float,
    k_mid: float,
    k_week18: float,
    k_post: float,
    divisor: float,
    regress_frac: float,
) -> tuple[list[float], list[str], list[str]]:
    """Run the Elo simulation engine and return per-game predictions.

    Thin wrapper around the tuner's ``_simulate_and_score`` that loads
    the shared engine without duplicating the simulation logic.

    Args:
        games: Prepared games DataFrame (output of ``_prepare_games``).
        k_early: K-factor for weeks 1-4.
        k_mid: K-factor for weeks 5-17.
        k_week18: K-factor for week 18.
        k_post: K-factor for weeks 19-22.
        divisor: Win-probability divisor.
        regress_frac: Offseason regression fraction.

    Returns:
        Tuple of (away_probs, game_seasons, game_ids).
    """
    from gridiron_edge.evaluation.tune import (
        _EXPANSION_START,
        _prepare_games,
        _simulate_and_score,
    )

    games_prepared, sorted_years, teams_by_year = _prepare_games(games)

    away_probs, _outcomes, game_seasons, game_ids = _simulate_and_score(
        games_prepared,
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
    return away_probs, game_seasons, game_ids


def _merge_elo_predictions(
    schedule: pd.DataFrame,
    elo: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
) -> pd.DataFrame:
    """Merge Elo ratings onto an upcoming schedule and compute win probs.

    Args:
        schedule: Canonical upcoming schedule DataFrame.
        elo: Elo state table DataFrame.
        model_name: Model purpose (always ``"win_prob"`` for the Elo predictor).
        model_type: Model algorithm (always ``"elo"`` for the Elo predictor).

    Returns:
        DataFrame compatible with ``build_predictions_df()`` output schema.
        Tagged with ``model_name`` and ``model_type`` columns so downstream
        archiving has the pair available.
    """
    from gridiron_edge.ratings.elo.core import elo_win_probability

    df = schedule.copy()

    df = (
        df.merge(
            elo[["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"]],
            how="left",
            left_on=["AWAY_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "AWAY_TEAM_ELO"})
    )

    df = (
        df.merge(
            elo[["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"]],
            how="left",
            left_on=["HOME_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "HOME_TEAM_ELO"})
    )

    df = df.dropna(subset=["AWAY_TEAM_ELO", "HOME_TEAM_ELO"]).copy()

    if df.empty:
        return df

    probs = df.apply(
        lambda row: elo_win_probability(row["AWAY_TEAM_ELO"], row["HOME_TEAM_ELO"]),
        axis=1,
        result_type="expand",
    )
    df["AWAY_WIN_PROB"] = probs[0]
    df["HOME_WIN_PROB"] = probs[1]
    df["AWAY_TEAM_WIN_PROB"] = df["AWAY_WIN_PROB"].map(lambda x: f"{x * 100:.1f} %")
    df["HOME_TEAM_WIN_PROB"] = df["HOME_WIN_PROB"].map(lambda x: f"{x * 100:.1f} %")
    df["model_name"] = model_name
    df["model_type"] = model_type

    return df.drop(columns=["YEAR"])


# ---------------------------------------------------------------------------
# WinProbEloPredictor - composite key "win_prob_elo"
# ---------------------------------------------------------------------------


@ModelRegistry.register
class WinProbEloPredictor:
    """Elo predictor with production-default parameters.

    Composite registry key: ``"win_prob_elo"``.
    Parameters: K=20, divisor=480, offseason regression=1/3.
    These match the production Elo state table built by ``ratings/elo/fit.py``.
    """

    model_name: ClassVar[str] = "win_prob"
    model_type: ClassVar[str] = "elo"

    spec: ClassVar[ModelSpec] = ModelSpec(
        name="win_prob_elo",
        description="Elo ratings - production defaults (K=20, div=480, regress=0.33).",
    )

    # Production parameter constants.
    K: Final[float] = 20.0
    DIVISOR: Final[float] = 480.0
    REGRESS_FRAC: Final[float] = 1 / 3.0

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate Elo predictions for all historical games."""
        away_probs, game_seasons, game_ids = _run_simulation(
            games,
            k_early=self.K,
            k_mid=self.K,
            k_week18=self.K,
            k_post=self.K,
            divisor=self.DIVISOR,
            regress_frac=self.REGRESS_FRAC,
        )
        if not away_probs:
            logger.warning("WinProbEloPredictor: no predictions generated.")
            return pd.DataFrame()

        from gridiron_edge.evaluation.tune import _prepare_games

        games_prepared, _, _ = _prepare_games(games)
        return _build_elo_predictions(
            away_probs,
            game_seasons,
            game_ids,
            games_prepared,
            model_name=self.model_name,
            model_type=self.model_type,
        )

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate Elo predictions for upcoming games."""
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.datasets import loaders

        resolved_repo = repo or get_settings().repo_root
        elo = loaders.load_elo_state(resolved_repo)
        return _merge_elo_predictions(
            schedule,
            elo,
            model_name=self.model_name,
            model_type=self.model_type,
        )
