# src/gridiron_edge/features/team/venue_hfa.py

"""Canonical Home franchise advantage feature.

Computes a leakage-free Home franchise home-advantage coefficient from
completed, non-neutral games before the target week.

Neutral-site games receive zero Home franchise advantage. Franchises
without the configured minimum historical Home sample also receive zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


# Minimum home games required to use a franchise's own win rate.
# Below this threshold we fall back to 0.0 (league average differential).
_MIN_HOME_GAMES: Final[int] = 20

_HOME_AWAY_HFA_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "HOME_TEAM",
    "IS_NEUTRAL_SITE",
)

_HOME_AWAY_HFA_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
    "IS_NEUTRAL_SITE",
)

_HOME_AWAY_HFA_OUTPUT: Final[str] = "HOME_FRANCHISE_HFA"


def _require_home_away_hfa_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require canonical franchise-HFA columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _canonical_season_numbers(values: Series) -> Series:
    """Convert canonical season labels to starting-season integers."""
    if values.isna().any():
        raise ValueError("YEAR must not contain nulls.")

    season_text = values.astype(str).str.strip().str.split("-", n=1).str[0]
    if season_text.eq("").any():
        raise ValueError("YEAR must not contain empty values.")

    try:
        return season_text.astype(int)
    except ValueError as exc:
        raise ValueError("YEAR must begin with a numeric season.") from exc


def _coerce_neutral_state(values: Series, *, label: str) -> Series:
    """Validate and normalize canonical neutral-site state."""
    # pyrefly: ignore [bad-assignment]
    neutral: Series = pd.to_numeric(values, errors="raise")
    if neutral.isna().any() or not neutral.isin([0, 1]).all():
        raise ValueError(f"{label} IS_NEUTRAL_SITE must contain only 0 or 1.")
    return neutral.astype(int)


def _build_home_advantage_history(games: DataFrame) -> DataFrame:
    """Build completed non-neutral home results for as-of HFA estimates."""
    columns = [
        "HOME_TEAM",
        "_SEASON_START",
        "WEEK_NUM",
        "_HOME_RESULT",
    ]
    if games.empty:
        return DataFrame(columns=columns)

    _require_home_away_hfa_columns(
        games,
        _HOME_AWAY_HFA_HISTORY_COLUMNS,
        label="Historical games",
    )

    history = games.loc[
        :,
        list(_HOME_AWAY_HFA_HISTORY_COLUMNS),
    ].copy()

    if history["GAME_ID"].duplicated().any():
        raise ValueError("Historical games contain duplicate game IDs.")

    history["_SEASON_START"] = _canonical_season_numbers(history["YEAR"])
    history["WEEK_NUM"] = history["WEEK_NUM"].astype(int)
    history["IS_NEUTRAL_SITE"] = _coerce_neutral_state(
        history["IS_NEUTRAL_SITE"],
        label="Historical games",
    )
    history["AWAY_SCORE"] = pd.to_numeric(
        history["AWAY_SCORE"],
        errors="coerce",
    )
    history["HOME_SCORE"] = pd.to_numeric(
        history["HOME_SCORE"],
        errors="coerce",
    )

    completed = history.loc[
        history["AWAY_SCORE"].notna()
        & history["HOME_SCORE"].notna()
        & history["IS_NEUTRAL_SITE"].eq(0),
        :,
    ].copy()

    if completed.empty:
        return DataFrame(columns=columns)

    home_won = completed["HOME_SCORE"] > completed["AWAY_SCORE"]
    away_won = completed["AWAY_SCORE"] > completed["HOME_SCORE"]
    completed["_HOME_RESULT"] = np.select(
        [home_won, away_won],
        [1.0, 0.0],
        default=0.5,
    )

    return completed.loc[:, columns].sort_values(
        ["_SEASON_START", "WEEK_NUM", "HOME_TEAM"],
        kind="stable",
        ignore_index=True,
    )


def _home_hfa_entering_week(
    history: DataFrame,
    *,
    home_team: str,
    season_start: int,
    week: int,
) -> float:
    """Return a leakage-free franchise HFA entering the target week."""
    prior = history.loc[
        (history["_SEASON_START"] < season_start)
        | (history["_SEASON_START"].eq(season_start) & history["WEEK_NUM"].lt(week)),
        :,
    ]

    if prior.empty:
        return 0.0

    team_results = prior.loc[
        prior["HOME_TEAM"].astype(str).eq(home_team),
        "_HOME_RESULT",
    ]
    if len(team_results) < _MIN_HOME_GAMES:
        return 0.0

    league_average = float(prior["_HOME_RESULT"].mean())
    team_average = float(team_results.mean())
    return team_average - league_average


@FeatureRegistry.register("home_away_venue_hfa")
class HomeAwayVenueHFAFeature:
    """Compute the Home franchise's pregame home-advantage coefficient."""

    spec = FeatureSpec(
        name="home_away_venue_hfa",
        produces=[_HOME_AWAY_HFA_OUTPUT],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach leakage-free Home franchise HFA to canonical game rows."""
        _require_home_away_hfa_columns(
            df,
            _HOME_AWAY_HFA_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=[_HOME_AWAY_HFA_OUTPUT],
            errors="ignore",
        )
        source["_SEASON_START"] = _canonical_season_numbers(source["YEAR"])
        source["WEEK_NUM"] = source["WEEK_NUM"].astype(int)
        source["IS_NEUTRAL_SITE"] = _coerce_neutral_state(
            source["IS_NEUTRAL_SITE"],
            label="Home/away game frame",
        )

        history = _build_home_advantage_history(datasets.games())

        values = [
            0.0
            if int(neutral) == 1
            else _home_hfa_entering_week(
                history,
                home_team=str(home_team),
                season_start=int(season_start),
                week=int(week),
            )
            for home_team, season_start, week, neutral in zip(
                source["HOME_TEAM"],
                source["_SEASON_START"],
                source["WEEK_NUM"],
                source["IS_NEUTRAL_SITE"],
                strict=True,
            )
        ]

        source[_HOME_AWAY_HFA_OUTPUT] = values
        return source.drop(
            columns=["_SEASON_START"],
        ).reset_index(drop=True)
