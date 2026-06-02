# src/gridiron_edge/models/elo/predictor.py

"""Elo-based predictor implementations.

Implements the ``Predictor`` protocol for all Elo model versions.
Each version is registered with ``PredictorRegistry`` under its version
string so the evaluation framework can retrieve it by name.

Adding a new Elo variant (e.g. after tuning produces elo_v4) requires
only adding a new class here — no changes to evaluation, backfill, or CLI.

Model versions:
    elo_v1: Production defaults (K=20, divisor=480, regress=1/3)
    elo_v2: Tuned flat K   (K=40, divisor=350, regress=0.40)
    elo_v3: Tuned zone K   (populated after elo_v3 tune completes)
"""

from __future__ import annotations

import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.models.base import PredictorSpec
from gridiron_edge.models.game_prediction.post_process import enrich_predictions
from gridiron_edge.models.registry import PredictorRegistry

logger: Logger = logging.getLogger(__name__)


def _build_archive_rows(
    away_probs: list[float],
    game_seasons: list[str],
    game_ids: list[str],
    games: pd.DataFrame,
    model_version: str,
) -> pd.DataFrame:
    """Convert simulation output into prediction archive rows.

    Shared by all Elo predictor variants.

    Args:
        away_probs: Predicted away-team win probability per game.
        game_seasons: Season label per game.
        game_ids: Canonical GAME_ID per game.
        games: Full games DataFrame (used to look up team names and dates).
        model_version: Model version string to tag the rows with.

    Returns:
        DataFrame in prediction archive schema.
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

    weeks = games.set_index("GAME_ID")["WEEK_NUM"].to_dict()

    result = pd.DataFrame(
        {
            "predicted_at": datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None),
            "is_backfilled": True,
            "model_version": model_version,
            "season": game_seasons,
            "week": [int(weeks.get(gid, 0)) for gid in game_ids],
            "game_id": game_ids,
            "game_date": game_dates,
            "away_team": away_teams,
            "home_team": home_teams,
            "away_elo": float("nan"),
            "home_elo": float("nan"),
            "away_win_prob": away_probs,
            "home_win_prob": [1.0 - p for p in away_probs],
        }
    )

    result = enrich_predictions(
        result,
        model_version=model_version,
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

    # games is already prepared if called from predict_historical,
    # but _simulate_and_score needs sorted_years and teams_by_year too.
    # We re-prepare here since the predictor receives the raw games df.
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


# ---------------------------------------------------------------------------
# elo_v1 — production defaults
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class EloV1Predictor:
    """Elo predictor with production defaults.

    Parameters match ``EloTableConfig`` defaults and ``core.py`` divisor:
    K=20, divisor=480, offseason regression=1/3.
    """

    spec = PredictorSpec(
        name="elo_v1",
        description="Elo ratings — production defaults (K=20, div=480, regress=0.33)",
    )

    # Production parameter constants
    K: Final[float] = 20.0
    DIVISOR: Final[float] = 480.0
    REGRESS_FRAC: Final[float] = 1 / 3.0

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v1 predictions for all historical games."""
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
            logger.warning("EloV1Predictor: no predictions generated.")
            return pd.DataFrame()

        from gridiron_edge.evaluation.tune import _prepare_games

        games_prepared, _, _ = _prepare_games(games)
        return _build_archive_rows(
            away_probs, game_seasons, game_ids, games_prepared, self.spec.name
        )

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v1 predictions for upcoming games."""
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.datasets import loaders

        resolved_repo = repo or get_settings().repo_root
        elo = loaders.load_elo_state(resolved_repo)
        return _merge_elo_predictions(schedule, elo, self.spec.name)


# ---------------------------------------------------------------------------
# elo_v2 — tuned flat K
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class EloV2Predictor:
    """Elo predictor with tuned flat-K parameters.

    Best parameters from the elo_v2 grid search:
    K=40, divisor=350, regress=0.40.
    """

    spec = PredictorSpec(
        name="elo_v2",
        description="Elo ratings — tuned flat K (K=40, div=350, regress=0.40)",
    )

    K: Final[float] = 40.0
    DIVISOR: Final[float] = 350.0
    REGRESS_FRAC: Final[float] = 0.40

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v2 predictions for all historical games."""
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
            logger.warning("EloV2Predictor: no predictions generated.")
            return pd.DataFrame()

        from gridiron_edge.evaluation.tune import _prepare_games

        games_prepared, _, _ = _prepare_games(games)
        return _build_archive_rows(
            away_probs, game_seasons, game_ids, games_prepared, self.spec.name
        )

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v2 predictions for upcoming games."""
        # elo_v2 uses a different Elo state table built with tuned params.
        # Until that table is built, falls back to the production Elo state.
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.datasets import loaders

        resolved_repo = repo or get_settings().repo_root
        elo = loaders.load_elo_state(resolved_repo)
        return _merge_elo_predictions(schedule, elo, self.spec.name)


# ---------------------------------------------------------------------------
# elo_v3 — tuned zone-based K
# ---------------------------------------------------------------------------


@PredictorRegistry.register
class EloV3Predictor:
    """Elo predictor with tuned zone-based K parameters.

    Best parameters from the elo_v3 grid search (63,504 combinations):
    k_early=40, k_mid=40, k_week18=50, k_post=60,
    divisor=360, regress=0.40.

    The zone-based K reflects structural differences in game informativeness:
    - Weeks 1-4: K=40 (same as mid-season, ratings converging)
    - Weeks 5-17: K=40 (most informative games)
    - Week 18: K=50 (slightly higher -- line movement noise warrants faster update)
    - Weeks 19-22: K=60 (playoff games are high-signal elimination games)
    """

    spec = PredictorSpec(
        name="elo_v3",
        description=(
            "Elo ratings -- tuned zone-based K "
            "(early=40, mid=40, wk18=50, post=60, div=360, regress=0.40)"
        ),
    )

    K_EARLY: Final[float] = 40.0
    K_MID: Final[float] = 40.0
    K_WEEK18: Final[float] = 50.0
    K_POST: Final[float] = 60.0
    DIVISOR: Final[float] = 360.0
    REGRESS_FRAC: Final[float] = 0.40

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v3 predictions for all historical games."""
        away_probs, game_seasons, game_ids = _run_simulation(
            games,
            k_early=self.K_EARLY,
            k_mid=self.K_MID,
            k_week18=self.K_WEEK18,
            k_post=self.K_POST,
            divisor=self.DIVISOR,
            regress_frac=self.REGRESS_FRAC,
        )
        if not away_probs:
            logger.warning("EloV3Predictor: no predictions generated.")
            return pd.DataFrame()

        from gridiron_edge.evaluation.tune import _prepare_games

        games_prepared, _, _ = _prepare_games(games)
        return _build_archive_rows(
            away_probs, game_seasons, game_ids, games_prepared, self.spec.name
        )

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate elo_v3 predictions for upcoming games."""
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.datasets import loaders

        resolved_repo = repo or get_settings().repo_root
        elo = loaders.load_elo_state(resolved_repo)
        return _merge_elo_predictions(schedule, elo, self.spec.name)


# ---------------------------------------------------------------------------
# Shared upcoming prediction helper
# ---------------------------------------------------------------------------


def _merge_elo_predictions(
    schedule: pd.DataFrame,
    elo: pd.DataFrame,
    model_version: str,
) -> pd.DataFrame:
    """Merge Elo ratings onto an upcoming schedule and compute win probs.

    Shared by all Elo-based ``predict_upcoming`` implementations.

    Args:
        schedule: Canonical upcoming schedule DataFrame.
        elo: Elo state table DataFrame.
        model_version: Model version string for labelling.

    Returns:
        DataFrame compatible with ``build_predictions_df()`` output schema.
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

    return df.drop(columns=["YEAR"])
