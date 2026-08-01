# src/gridiron_edge/features/team/rest.py

"""Canonical Away/Home pregame rest features.

Computes days since each designated team's latest completed game before
the target game date, along with nullable short-week and post-bye flags.

All outputs use stable Away/Home identity, and the rest differential is:

    Home Days Rest - Away Days Rest
"""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Days-rest thresholds
_SHORT_WEEK_THRESHOLD: Final[int] = 6  # fewer days → short week flag
_BYE_WEEK_THRESHOLD: Final[int] = 13  # 13+ days → post-bye flag

_HOME_AWAY_REST_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "GAME_DATE",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_HOME_AWAY_REST_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "GAME_DATE",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_HOME_AWAY_REST_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_DAYS_REST",
    "HOME_DAYS_REST",
    "AWAY_SHORT_WEEK",
    "HOME_SHORT_WEEK",
    "AWAY_POST_BYE",
    "HOME_POST_BYE",
    "DAYS_REST_DIFF",
)

type _RestScalar = str | float | int | pd.Timestamp | None


def _require_home_away_rest_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require canonical rest-feature input columns."""
    missing: list[str] = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _completed_team_dates(
    games: DataFrame,
) -> dict[str, list[pd.Timestamp]]:
    """Build sorted completed-game dates for every team."""
    _require_home_away_rest_columns(
        games,
        _HOME_AWAY_REST_HISTORY_COLUMNS,
        label="Historical games",
    )

    away = games.loc[
        :,
        [
            "GAME_ID",
            "GAME_DATE",
            "AWAY_TEAM",
        ],
    ].rename(
        columns={
            "AWAY_TEAM": "TEAM",
        }
    )

    home = games.loc[
        :,
        [
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM",
        ],
    ].rename(
        columns={
            "HOME_TEAM": "TEAM",
        }
    )

    team_games = pd.concat(
        [
            away,
            home,
        ],
        ignore_index=True,
    )

    duplicated = team_games.duplicated(
        subset=[
            "GAME_ID",
            "TEAM",
        ],
        keep=False,
    )
    if duplicated.any():
        raise ValueError("Historical games contain duplicate team-game identities.")

    team_games["_DATE"] = pd.to_datetime(
        team_games["GAME_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    valid = (
        team_games["TEAM"].notna()
        & team_games["TEAM"].astype(str).str.strip().ne("")
        # pyrefly: ignore [missing-attribute]
        & team_games["_DATE"].notna()
    )
    team_games = team_games.loc[
        valid,
        :,
    ].copy()

    return {
        str(team): sorted(group["_DATE"].tolist())
        for team, group in team_games.groupby(
            "TEAM",
            sort=False,
        )
    }


def _days_since_previous_game(
    *,
    team: str | None,
    game_date: pd.Timestamp | None,
    history: dict[str, list[pd.Timestamp]],
) -> float:
    """Return days since the latest completed game before the target."""
    if team is None or game_date is None:
        return float("nan")

    team_name = team.strip()
    if not team_name:
        return float("nan")

    dates = history.get(
        team_name,
        [],
    )
    index = bisect_left(
        dates,
        game_date,
    )
    if index == 0:
        return float("nan")

    previous_date = dates[index - 1]
    return float((game_date - previous_date).days)


def _rest_flag(
    rest: pd.Series,
    *,
    threshold: int,
    comparison: str,
) -> pd.Series:
    """Build a nullable binary rest flag."""
    present = rest.notna()

    if comparison == "lt":
        flag = rest < threshold
    elif comparison == "ge":
        flag = rest >= threshold
    else:
        raise ValueError(f"Unsupported rest comparison: {comparison}")

    return flag.where(
        present,
        other=float("nan"),
    ).astype("float64")


@FeatureRegistry.register("home_away_rest")
class HomeAwayRestFeature:
    """Compute canonical Away and Home pregame rest features."""

    spec = FeatureSpec(
        name="home_away_rest",
        produces=list(_HOME_AWAY_REST_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach schedule-complete Away and Home rest features.

        Rest is measured from the target game date to the latest
        completed historical game date strictly before it.

        Missing history or an invalid target date remains null.
        """
        _require_home_away_rest_columns(
            df,
            _HOME_AWAY_REST_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=list(_HOME_AWAY_REST_COLUMNS),
            errors="ignore",
        )

        history = _completed_team_dates(datasets.games())

        target_dates = pd.to_datetime(
            source["GAME_DATE"],
            format="%Y-%m-%d",
            errors="coerce",
        )

        source["AWAY_DAYS_REST"] = [
            _days_since_previous_game(
                team=(None if pd.isna(team) else str(team)),
                game_date=(None if pd.isna(game_date) else game_date),
                history=history,
            )
            for team, game_date in zip(
                source["AWAY_TEAM"],
                # pyrefly: ignore [bad-argument-type]
                target_dates,
                strict=True,
            )
        ]

        source["HOME_DAYS_REST"] = [
            _days_since_previous_game(
                team=(None if pd.isna(team) else str(team)),
                game_date=(None if pd.isna(game_date) else game_date),
                history=history,
            )
            for team, game_date in zip(
                source["HOME_TEAM"],
                # pyrefly: ignore [bad-argument-type]
                target_dates,
                strict=True,
            )
        ]

        source["AWAY_SHORT_WEEK"] = _rest_flag(
            source["AWAY_DAYS_REST"],
            threshold=_SHORT_WEEK_THRESHOLD,
            comparison="lt",
        )
        source["HOME_SHORT_WEEK"] = _rest_flag(
            source["HOME_DAYS_REST"],
            threshold=_SHORT_WEEK_THRESHOLD,
            comparison="lt",
        )

        source["AWAY_POST_BYE"] = _rest_flag(
            source["AWAY_DAYS_REST"],
            threshold=_BYE_WEEK_THRESHOLD,
            comparison="ge",
        )
        source["HOME_POST_BYE"] = _rest_flag(
            source["HOME_DAYS_REST"],
            threshold=_BYE_WEEK_THRESHOLD,
            comparison="ge",
        )

        source["DAYS_REST_DIFF"] = source["HOME_DAYS_REST"] - source["AWAY_DAYS_REST"]

        return source.reset_index(
            drop=True,
        )
