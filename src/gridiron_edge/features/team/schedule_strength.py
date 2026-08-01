# src/gridiron_edge/features/team/schedule_strength.py

"""Canonical Away/Home schedule-strength features.

Computes each designated team's strength of schedule and strength of
victory entering the target week from prior same-season opponents and
their historical pregame Elo ratings.

All outputs use stable Away/Home identity and canonical score-derived
game results.
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


_HOME_AWAY_STRENGTH_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_HOME_AWAY_STRENGTH_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
)

_ELO_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "NFL_TEAM",
    "NFL_YEAR",
    "NFL_WEEK",
    "ELO",
)

_HOME_AWAY_STRENGTH_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_SOS",
    "AWAY_SOV",
    "HOME_SOS",
    "HOME_SOV",
)


def _require_home_away_strength_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require canonical schedule-strength input columns."""
    missing: list[str] = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _validate_home_away_strength_elo_identity(
    elo: DataFrame,
) -> None:
    """Reject duplicate team, season, and week Elo identities."""
    duplicated: Series = elo.duplicated(
        subset=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        keep=False,
    )

    if duplicated.any():
        raise ValueError("Elo state contains duplicate team-season-week identities.")


def _build_home_away_opponent_history(
    games: DataFrame,
    elo: DataFrame,
) -> DataFrame:
    """Build team-opponent result history with historical opponent Elo."""
    output_columns = [
        "TEAM",
        "OPPONENT",
        "YEAR",
        "WEEK_NUM",
        "RESULT",
        "OPP_ELO",
    ]

    if games.empty or elo.empty:
        return DataFrame(columns=output_columns)

    _require_home_away_strength_columns(
        games,
        _HOME_AWAY_STRENGTH_HISTORY_COLUMNS,
        label="Historical games",
    )
    _require_home_away_strength_columns(
        elo,
        _ELO_IDENTITY_COLUMNS,
        label="Elo state",
    )
    _validate_home_away_strength_elo_identity(elo)

    history = games.loc[
        :,
        list(_HOME_AWAY_STRENGTH_HISTORY_COLUMNS),
    ].copy()

    if history["GAME_ID"].duplicated().any():
        raise ValueError("Historical games contain duplicate game IDs.")

    history["WEEK_NUM"] = history["WEEK_NUM"].astype(int)
    history["AWAY_SCORE"] = pd.to_numeric(
        history["AWAY_SCORE"],
        errors="coerce",
    )
    history["HOME_SCORE"] = pd.to_numeric(
        history["HOME_SCORE"],
        errors="coerce",
    )

    completed = history.loc[
        history["AWAY_SCORE"].notna() & history["HOME_SCORE"].notna(),
        :,
    ].copy()

    away_won = completed["AWAY_SCORE"] > completed["HOME_SCORE"]
    home_won = completed["HOME_SCORE"] > completed["AWAY_SCORE"]

    away_rows = completed.loc[
        :,
        [
            "YEAR",
            "WEEK_NUM",
            "AWAY_TEAM",
            "HOME_TEAM",
        ],
    ].rename(
        columns={
            "AWAY_TEAM": "TEAM",
            "HOME_TEAM": "OPPONENT",
        }
    )
    away_rows["RESULT"] = np.select(
        [away_won, home_won],
        [1.0, 0.0],
        default=0.5,
    )

    home_rows = completed.loc[
        :,
        [
            "YEAR",
            "WEEK_NUM",
            "HOME_TEAM",
            "AWAY_TEAM",
        ],
    ].rename(
        columns={
            "HOME_TEAM": "TEAM",
            "AWAY_TEAM": "OPPONENT",
        }
    )
    home_rows["RESULT"] = np.select(
        [home_won, away_won],
        [1.0, 0.0],
        default=0.5,
    )

    opponent_history = pd.concat(
        [away_rows, home_rows],
        ignore_index=True,
    )

    opponent_history["TEAM"] = opponent_history["TEAM"].astype(str)
    opponent_history["OPPONENT"] = opponent_history["OPPONENT"].astype(str)

    elo_lookup = elo.loc[
        :,
        list(_ELO_IDENTITY_COLUMNS),
    ].rename(
        columns={
            "NFL_TEAM": "OPPONENT",
            "NFL_YEAR": "YEAR",
            "NFL_WEEK": "WEEK_NUM",
            "ELO": "OPP_ELO",
        }
    )
    elo_lookup["WEEK_NUM"] = elo_lookup["WEEK_NUM"].astype(int)

    return opponent_history.merge(
        elo_lookup,
        how="left",
        on=[
            "OPPONENT",
            "YEAR",
            "WEEK_NUM",
        ],
        sort=False,
        validate="many_to_one",
    )


def _strength_entering_week(
    history: DataFrame,
    *,
    team: str,
    year: str,
    week: int,
) -> tuple[float, float]:
    """Return one team's SOS and SOV entering the target week."""
    prior = history.loc[
        (history["TEAM"] == team)
        & (history["YEAR"].astype(str) == year)
        & (history["WEEK_NUM"] < week),
        :,
    ]

    opponent_elos = prior["OPP_ELO"].dropna()
    sos = float(opponent_elos.mean()) if not opponent_elos.empty else float("nan")

    defeated_opponent_elos = prior.loc[
        prior["RESULT"] == 1.0,
        "OPP_ELO",
    ].dropna()
    sov = float(defeated_opponent_elos.mean()) if not defeated_opponent_elos.empty else float("nan")

    return sos, sov


@FeatureRegistry.register("home_away_schedule_strength")
class HomeAwayScheduleStrengthFeature:
    """Compute canonical Away and Home pregame schedule strength."""

    spec = FeatureSpec(
        name="home_away_schedule_strength",
        produces=list(_HOME_AWAY_STRENGTH_COLUMNS),
        depends_on=("home_away_elo",),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach same-season pregame SOS and SOV values."""
        _require_home_away_strength_columns(
            df,
            _HOME_AWAY_STRENGTH_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=list(_HOME_AWAY_STRENGTH_COLUMNS),
            errors="ignore",
        )

        # pyrefly: ignore [bad-assignment]
        weeks: Series = pd.to_numeric(
            source["WEEK_NUM"],
            errors="raise",
        )
        source["WEEK_NUM"] = weeks.astype(int)

        history = _build_home_away_opponent_history(
            datasets.games(),
            datasets.elo_state(),
        )

        away_strength = [
            _strength_entering_week(
                history,
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

        home_strength = [
            _strength_entering_week(
                history,
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
            away_strength,
            index=source.index,
            columns=[
                "AWAY_SOS",
                "AWAY_SOV",
            ],
        )
        home_frame = DataFrame(
            home_strength,
            index=source.index,
            columns=[
                "HOME_SOS",
                "HOME_SOV",
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
