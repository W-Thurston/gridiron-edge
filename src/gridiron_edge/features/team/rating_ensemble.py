# src/gridiron_edge/features/team/rating_ensemble.py

"""Canonical Away/Home rating-ensemble features.

Three features, each recomputed from the complete games history rather than
read from a persisted artifact (``DECISIONS.md`` D69):

- ``home_away_glicko2``: team Glicko-2 rating and rating deviation (RD).
- ``home_away_linear_ratings``: Massey, Colley and Keener ratings.
- ``home_away_qb_elo``: the resolved starting QB's rating component and prior
  start count.

Every value reflects only completed games with a strictly earlier (season,
week). A game beyond the last completed game gets the last state carried
forward, so these columns are never null for a scheduled game, including a
season's first week. Glicko-2 volatility is computed by the algorithm but not
emitted: it is effectively constant (``DECISIONS.md`` D70).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.qb_profile import resolve_starting_qb_history
from gridiron_edge.models.game_prediction.game_schema import (
    away_feature_name,
    home_feature_name,
)
from gridiron_edge.ratings.glicko2 import Glicko2History
from gridiron_edge.ratings.history import season_start
from gridiron_edge.ratings.linear_ratings import LinearRatings
from gridiron_edge.ratings.qb_elo import QbEloHistory

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gridiron_edge.datasets.accessor import DatasetAccessor

GLICKO2_METRICS: Final[tuple[str, ...]] = ("glicko2_rating", "glicko2_rd")
LINEAR_RATING_METRICS: Final[tuple[str, ...]] = (
    "massey_rating",
    "colley_rating",
    "keener_rating",
)
QB_ELO_METRICS: Final[tuple[str, ...]] = ("qb_elo", "qb_elo_starts")

#: Every rating-ensemble metric, in feature-column order.
RATING_ENSEMBLE_METRICS: Final[tuple[str, ...]] = (
    *GLICKO2_METRICS,
    *LINEAR_RATING_METRICS,
    *QB_ELO_METRICS,
)

_GAME_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)


def _columns(metrics: Sequence[str]) -> list[str]:
    """Away columns followed by Home columns for the given metrics."""
    return [away_feature_name(m.upper()) for m in metrics] + [
        home_feature_name(m.upper()) for m in metrics
    ]


def _prepare(df: DataFrame, metrics: Sequence[str]) -> DataFrame:
    missing = sorted(set(_GAME_INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError("Home/away game frame is missing required columns: " + ", ".join(missing))
    source = df.copy().drop(columns=_columns(metrics), errors="ignore").reset_index(drop=True)
    source["_SEASON"] = source["YEAR"].map(season_start).astype(int)
    return source


def _cutoff_positions(source: DataFrame) -> list[tuple[int, int, np.ndarray]]:
    """Row positions grouped by their (season, week) cutoff."""
    pairs = source[["_SEASON", "WEEK_NUM"]].to_numpy(dtype=int)
    positions: list[tuple[int, int, np.ndarray]] = []
    for season, week in sorted({(int(a), int(b)) for a, b in pairs}):
        rows = np.flatnonzero((pairs[:, 0] == season) & (pairs[:, 1] == week))
        positions.append((season, week, rows))
    return positions


def _attach_team_ratings(
    df: DataFrame,
    metrics: Sequence[str],
    lookup: Callable[[int, int, list[str]], DataFrame],
) -> DataFrame:
    """Attach per-team metrics for both sides, one lookup per cutoff.

    ``lookup(season, week, teams)`` returns a frame indexed by team with one
    column per metric.
    """
    source = _prepare(df, metrics)
    values = {
        side(m.upper()): np.full(len(source), np.nan)
        for m in metrics
        for side in (away_feature_name, home_feature_name)
    }
    for season, week, rows in _cutoff_positions(source):
        block = source.iloc[rows]
        teams = sorted(set(block["AWAY_TEAM"]) | set(block["HOME_TEAM"]))
        table = lookup(season, week, teams)
        for m in metrics:
            values[away_feature_name(m.upper())][rows] = (
                table[m].reindex(block["AWAY_TEAM"]).to_numpy(dtype=float)
            )
            values[home_feature_name(m.upper())][rows] = (
                table[m].reindex(block["HOME_TEAM"]).to_numpy(dtype=float)
            )
    return source.drop(columns=["_SEASON"]).assign(**values)


@FeatureRegistry.register("home_away_glicko2")
class HomeAwayGlicko2Feature:
    """Attach Away and Home team Glicko-2 rating and RD."""

    spec = FeatureSpec(name="home_away_glicko2", produces=_columns(GLICKO2_METRICS))

    def compute(self, *, df: DataFrame, datasets: DatasetAccessor) -> DataFrame:
        """Attach pre-game Glicko-2 values; every input row is kept.

        Raises:
            ValueError: If cleaned game history is incomplete (``D69``).
        """
        history = Glicko2History(datasets.games())

        def lookup(season: int, week: int, teams: list[str]) -> DataFrame:
            states = history.state_as_of(season, week, teams)
            return pd.DataFrame(
                {
                    "glicko2_rating": {t: s.rating for t, s in states.items()},
                    "glicko2_rd": {t: s.rd for t, s in states.items()},
                }
            )

        return _attach_team_ratings(df, GLICKO2_METRICS, lookup)


@FeatureRegistry.register("home_away_linear_ratings")
class HomeAwayLinearRatingsFeature:
    """Attach Away and Home Massey, Colley and Keener ratings."""

    spec = FeatureSpec(name="home_away_linear_ratings", produces=_columns(LINEAR_RATING_METRICS))

    def compute(self, *, df: DataFrame, datasets: DatasetAccessor) -> DataFrame:
        """Attach ratings solved from completed games before each cutoff.

        Raises:
            ValueError: If cleaned game history is incomplete (``D69``).
        """
        ratings = LinearRatings(datasets.games())

        def lookup(season: int, week: int, teams: list[str]) -> DataFrame:
            table = ratings.as_of(season, week, teams)
            return table.rename(columns={c: f"{c.lower()}_rating" for c in table.columns})

        return _attach_team_ratings(df, LINEAR_RATING_METRICS, lookup)


@FeatureRegistry.register("home_away_qb_elo")
class HomeAwayQbEloFeature:
    """Attach the resolved starting QB's rating component and prior starts."""

    spec = FeatureSpec(name="home_away_qb_elo", produces=_columns(QB_ELO_METRICS))

    def compute(self, *, df: DataFrame, datasets: DatasetAccessor) -> DataFrame:
        """Attach QB rating values for the D65-resolved starter of each side.

        Identity is the actual starter for a played game and the team's most
        recent known starter otherwise. A side with no resolvable starter, or
        no sidecar at all, is null.

        Raises:
            ValueError: If cleaned game history is incomplete, or a completed
                game has no recorded starter (``D69``).
        """
        source = _prepare(df, QB_ELO_METRICS)
        values = {
            side(m.upper()): np.full(len(source), np.nan)
            for m in QB_ELO_METRICS
            for side in (away_feature_name, home_feature_name)
        }

        sidecar = datasets.qb_coach_sidecar()
        if not sidecar.empty:
            history = QbEloHistory(datasets.games(), sidecar)
            resolved = resolve_starting_qb_history(sidecar)
            by_side = {
                side: resolved.loc[resolved["SIDE"] == side].set_index("GAME_ID")["RESOLVED_QB_ID"]
                for side in ("AWAY", "HOME")
            }
            ids = {
                side: source["GAME_ID"].map(by_side[side]).astype(object)
                for side in ("AWAY", "HOME")
            }
            for season, week, rows in _cutoff_positions(source):
                wanted = {
                    str(q)
                    for side in ids
                    for q in ids[side].iloc[rows]
                    if isinstance(q, str) or pd.notna(q)
                }
                states = history.qb_state_as_of(season, week, sorted(wanted))
                for side, namer in (("AWAY", away_feature_name), ("HOME", home_feature_name)):
                    for pos, qb_id in zip(rows, ids[side].iloc[rows], strict=True):
                        if pd.isna(qb_id):
                            continue
                        state = states[str(qb_id)]
                        values[namer("QB_ELO")][pos] = state.rating
                        values[namer("QB_ELO_STARTS")][pos] = state.starts

        return source.drop(columns=["_SEASON"]).assign(**values)
