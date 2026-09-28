# src/gridiron_edge/features/team/league_environment.py

"""Canonical league scoring-environment feature.

``LEAGUE_PPG_TO_DATE`` is the league's points per team-game over completed
regular-season games before the target week, blended with the previous
regular season's full-season mean so early-season weeks are not noise and
week 1 has a value.

It is a continuous signal on purpose. Rule-era flags (2024 dynamic kickoff,
2025 rule changes) exist only in holdout seasons, where a model cannot learn
their effect; a season-to-date scoring level extrapolates instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

LEAGUE_PPG_COLUMN: Final[str] = "LEAGUE_PPG_TO_DATE"

# Last regular-season week; playoff games never feed the league level.
_MAX_REG_SEASON_WEEK: Final[int] = 18

# Weight of the previous season's mean, in games. Derived on seasons before
# the holdout window (1999-2022): per-game mean points have within-season
# variance ~49.5; the season means' between-season variance is ~1.20, of
# which ~0.19 is sampling noise, leaving ~1.02 of true variation. The
# precision-weighted prior therefore carries 49.5 / 1.02 ~= 48.5 games, about
# three weeks of a season. Never tuned on holdout seasons.
_PRIOR_SEASON_WEIGHT_GAMES: Final[float] = 48.5

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_SCORE",
    "HOME_SCORE",
)


def _season_weekly_totals(games: DataFrame) -> DataFrame:
    """Return per-(season, week) point sums and game counts for regular-season games."""
    missing = sorted(set(_REQUIRED_COLUMNS) - set(games.columns))
    if missing:
        raise ValueError("Historical games is missing required columns: " + ", ".join(missing))

    frame = games.loc[:, list(_REQUIRED_COLUMNS)].copy()
    frame["WEEK_NUM"] = pd.to_numeric(frame["WEEK_NUM"], errors="raise").astype(int)
    frame["_POINTS_PER_TEAM"] = (
        pd.to_numeric(frame["AWAY_SCORE"], errors="coerce")
        + pd.to_numeric(frame["HOME_SCORE"], errors="coerce")
    ) / 2.0
    frame = frame.loc[
        frame["_POINTS_PER_TEAM"].notna() & (frame["WEEK_NUM"] <= _MAX_REG_SEASON_WEEK), :
    ]
    return (
        frame.groupby([frame["YEAR"].astype(str), "WEEK_NUM"])
        .agg(points=("_POINTS_PER_TEAM", "sum"), games=("_POINTS_PER_TEAM", "size"))
        .reset_index()
        .rename(columns={"YEAR": "SEASON"})
    )


def _level_entering_week(
    weekly: DataFrame,
    season_means: pd.Series,
    *,
    season: str,
    week: int,
) -> float:
    """Return the blended league level entering ``week`` of ``season``."""
    earlier = weekly.loc[(weekly["SEASON"] == season) & (weekly["WEEK_NUM"] < week), :]
    games = float(earlier["games"].sum())
    points = float(earlier["points"].sum())
    prior = _previous_season_mean(season_means, season)

    if prior is None:
        return points / games if games > 0 else float("nan")
    return (_PRIOR_SEASON_WEIGHT_GAMES * prior + points) / (_PRIOR_SEASON_WEIGHT_GAMES + games)


def _season_means(weekly: DataFrame) -> pd.Series:
    """Return each season's full regular-season points per team-game, in season order."""
    totals = weekly.groupby("SEASON")[["points", "games"]].sum()
    return (totals["points"] / totals["games"]).sort_index()


def _previous_season_mean(season_means: pd.Series, season: str) -> float | None:
    """Return the mean of the latest season strictly before ``season``.

    Looked up by label rather than by adjacency among seasons with completed
    games, so a season with no completed games yet (the first week of a new
    season) still resolves its prior.
    """
    earlier = season_means.loc[season_means.index < season]
    return float(earlier.iloc[-1]) if not earlier.empty else None


@FeatureRegistry.register("league_environment")
class LeagueEnvironmentFeature:
    """Attach the season-to-date league scoring level to each game."""

    spec = FeatureSpec(
        name="league_environment",
        produces=[LEAGUE_PPG_COLUMN],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach the blended league points-per-team-game level as of each week."""
        missing = sorted({"YEAR", "WEEK_NUM"} - set(df.columns))
        if missing:
            raise ValueError(
                "Home/away game frame is missing required columns: " + ", ".join(missing)
            )

        source = df.copy().drop(columns=[LEAGUE_PPG_COLUMN], errors="ignore")
        weekly = _season_weekly_totals(datasets.games())
        season_means = _season_means(weekly)

        weeks = pd.to_numeric(source["WEEK_NUM"], errors="raise").astype(int)
        cache: dict[tuple[str, int], float] = {}
        levels: list[float] = []
        for season, week in zip(source["YEAR"].astype(str), weeks, strict=True):
            key = (season, int(week))
            if key not in cache:
                cache[key] = _level_entering_week(
                    weekly, season_means, season=season, week=int(week)
                )
            levels.append(cache[key])

        source[LEAGUE_PPG_COLUMN] = levels
        return source.reset_index(drop=True)
