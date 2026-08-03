# src/gridiron_edge/features/team/record.py

"""Canonical Away/Home pregame record and streak features.

Builds one completed result history per team and game from canonical
Away and Home scores, then derives each designated team's same-season
record and active streak entering the target week.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

_HOME_AWAY_RECORD_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_HOME_AWAY_RECORD_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "GAME_DATE",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
)

_HOME_AWAY_RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_WINS",
    "AWAY_LOSSES",
    "AWAY_WIN_PCT",
    "AWAY_WIN_STREAK",
    "AWAY_LOSS_STREAK",
    "HOME_WINS",
    "HOME_LOSSES",
    "HOME_WIN_PCT",
    "HOME_WIN_STREAK",
    "HOME_LOSS_STREAK",
)


def _require_home_away_record_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require canonical record-feature columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _build_home_away_result_history(
    games: DataFrame,
) -> DataFrame:
    """Build one completed result row per team and game."""
    _require_home_away_record_columns(
        games,
        _HOME_AWAY_RECORD_HISTORY_COLUMNS,
        label="Historical games",
    )

    history = games.loc[
        :,
        list(_HOME_AWAY_RECORD_HISTORY_COLUMNS),
    ].copy()

    duplicated_games = history["GAME_ID"].duplicated(
        keep=False,
    )
    if duplicated_games.any():
        raise ValueError("Historical games contain duplicate game IDs.")

    # pyrefly: ignore [bad-assignment]
    weeks: pd.Series = pd.to_numeric(
        history["WEEK_NUM"],
        errors="raise",
    )
    history["WEEK_NUM"] = weeks.astype(int)

    # pyrefly: ignore [bad-assignment]
    away_scores: pd.Series = pd.to_numeric(
        history["AWAY_SCORE"],
        errors="coerce",
    )
    # pyrefly: ignore [bad-assignment]
    home_scores: pd.Series = pd.to_numeric(
        history["HOME_SCORE"],
        errors="coerce",
    )

    history["AWAY_SCORE"] = away_scores
    history["HOME_SCORE"] = home_scores

    completed = history.loc[
        away_scores.notna() & home_scores.notna(),
        :,
    ].copy()

    away_won = completed["AWAY_SCORE"] > completed["HOME_SCORE"]
    home_won = completed["HOME_SCORE"] > completed["AWAY_SCORE"]

    away_results = completed.loc[
        :,
        [
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "AWAY_TEAM",
        ],
    ].rename(
        columns={
            "AWAY_TEAM": "TEAM",
        }
    )
    away_results["RESULT"] = np.select(
        [
            away_won,
            home_won,
        ],
        [
            1.0,
            0.0,
        ],
        default=0.5,
    )

    home_results = completed.loc[
        :,
        [
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "HOME_TEAM",
        ],
    ].rename(
        columns={
            "HOME_TEAM": "TEAM",
        }
    )
    home_results["RESULT"] = np.select(
        [
            home_won,
            away_won,
        ],
        [
            1.0,
            0.0,
        ],
        default=0.5,
    )

    results = pd.concat(
        [
            away_results,
            home_results,
        ],
        ignore_index=True,
    )

    duplicated_teams = results.duplicated(
        subset=[
            "GAME_ID",
            "TEAM",
        ],
        keep=False,
    )
    if duplicated_teams.any():
        raise ValueError("Historical games contain duplicate team-game identities.")

    return results.sort_values(
        [
            "TEAM",
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
        ignore_index=True,
    )


def _record_entering_week(
    results: DataFrame,
    *,
    team: str,
    year: str,
    week: int,
) -> tuple[float, float, float, int, int]:
    """Return one team's same-season record entering a week."""
    prior = results.loc[
        (results["TEAM"].astype(str) == team)
        & (results["YEAR"].astype(str) == year)
        & (results["WEEK_NUM"] < week),
        "RESULT",
    ]

    if prior.empty:
        return (
            0.0,
            0.0,
            0.0,
            0,
            0,
        )

    wins = float((prior == 1.0).sum())
    losses = float((prior == 0.0).sum())
    ties = float((prior == 0.5).sum())

    wins += 0.5 * ties
    losses += 0.5 * ties
    win_pct = wins / float(len(prior))

    win_streak = 0
    loss_streak = 0

    latest = float(prior.iloc[-1])

    if latest == 1.0:
        for result in reversed(prior.tolist()):
            if float(result) != 1.0:
                break
            win_streak += 1

    if latest == 0.0:
        for result in reversed(prior.tolist()):
            if float(result) != 0.0:
                break
            loss_streak += 1

    return (
        wins,
        losses,
        win_pct,
        win_streak,
        loss_streak,
    )


@FeatureRegistry.register("home_away_record")
class HomeAwayRecordFeature:
    """Compute canonical Away and Home records entering each week."""

    spec = FeatureSpec(
        name="home_away_record",
        produces=list(_HOME_AWAY_RECORD_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach same-season pregame record and streak features."""
        _require_home_away_record_columns(
            df,
            _HOME_AWAY_RECORD_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=list(_HOME_AWAY_RECORD_COLUMNS),
            errors="ignore",
        )

        # pyrefly: ignore [bad-assignment]
        weeks: pd.Series = pd.to_numeric(
            source["WEEK_NUM"],
            errors="raise",
        )
        source["WEEK_NUM"] = weeks.astype(int)

        results = _build_home_away_result_history(datasets.games())

        away_records = [
            _record_entering_week(
                results,
                team=str(team),
                year=str(year),
                week=int(week),
            )
            for team, year, week in zip(
                source["AWAY_TEAM"],
                source["YEAR"],
                source["WEEK_NUM"],
                strict=True,
            )
        ]

        home_records = [
            _record_entering_week(
                results,
                team=str(team),
                year=str(year),
                week=int(week),
            )
            for team, year, week in zip(
                source["HOME_TEAM"],
                source["YEAR"],
                source["WEEK_NUM"],
                strict=True,
            )
        ]

        away_frame = DataFrame(
            away_records,
            index=source.index,
            columns=[
                "AWAY_WINS",
                "AWAY_LOSSES",
                "AWAY_WIN_PCT",
                "AWAY_WIN_STREAK",
                "AWAY_LOSS_STREAK",
            ],
        )

        home_frame = DataFrame(
            home_records,
            index=source.index,
            columns=[
                "HOME_WINS",
                "HOME_LOSSES",
                "HOME_WIN_PCT",
                "HOME_WIN_STREAK",
                "HOME_LOSS_STREAK",
            ],
        )

        return pd.concat(
            [
                source,
                away_frame,
                home_frame,
            ],
            axis=1,
        ).reset_index(drop=True)
