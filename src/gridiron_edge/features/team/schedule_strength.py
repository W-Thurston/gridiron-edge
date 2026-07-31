"""Strength of schedule and strength of victory features.

Measures opponent quality to contextualise a team's record. A team's raw
win-loss record does not distinguish between a schedule of weak opponents
and one of strong opponents - these features provide that context.

Both features use opponent Elo as the quality proxy. Elo is already
computed, trusted, and captures opponent strength in a single number
without introducing additional multicollinearity with the EPA features.
EPA-based opponent quality measures are a potential future extension
tracked in the backlog.

Produces (per team, computed from completed games prior to each matchup):

    TEAM_A_SOS    float   Average pre-game Elo of all opponents faced to
                          date this season. NaN if no games played yet (week 1).

    TEAM_A_SOV    float   Average pre-game Elo of opponents that the team
                          has beaten to date this season. NaN if no wins yet.

    TEAM_B_SOS    float   Same for TEAM_B.
    TEAM_B_SOV    float   Same for TEAM_B.

Design notes:
    - "Pre-game Elo" means the opponent's Elo *entering* the game they
      played against this team - not the opponent's current Elo. This
      avoids future leakage: an opponent that improved dramatically after
      the game does not retroactively make that game look harder.
    - The Elo state table (datasets.elo_state()) provides pre-game Elo
      keyed by (NFL_TEAM, NFL_YEAR, NFL_WEEK). This is the same table
      used by the team_elo feature.
    - Week 1: no prior opponents → SOS = NaN, SOV = NaN. These rows are
      excluded from training by _prepare_data's NaN filter.
    - Teams with wins but no losses (e.g. 4-0) still get a valid SOS
      (average of all 4 opponents' Elos) and SOV (average of those same
      4 opponents, since all games were wins).
    - Ties (WIN_OR_TIE == 0.5) are excluded from SOV - only outright
      wins count toward strength of victory, consistent with NFL
      tiebreaker convention.
    - Postseason games are included in opponent Elo lookups but the
      features reset each season (YEAR grouping).

Implementation note (schedule_strength/M1):
    Cumulative averages computed via cumsum + cumcount + shift, vectorized
    over all team-seasons. Opponent Elo is merged onto the long-format
    team-game table in a single join, replacing the per-row dict lookup
    plus four ``df.apply(axis=1)`` calls.
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

logger: Logger = logging.getLogger(__name__)

_PRODUCES: Final[list[str]] = [
    "TEAM_A_SOS",
    "TEAM_A_SOV",
    "TEAM_B_SOS",
    "TEAM_B_SOV",
]

_STAT_COLS: Final[list[str]] = ["SOS", "SOV"]

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


@FeatureRegistry.register("schedule_strength")
class ScheduleStrengthFeature:
    """Strength of schedule and strength of victory via opponent Elo.

    Computes the average pre-game Elo of all opponents faced (SOS) and
    of opponents beaten (SOV) for each team entering each game, using
    only results from prior games in the same season.
    """

    spec = FeatureSpec(
        name="schedule_strength",
        produces=_PRODUCES,
        depends_on=("team_elo",),
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute SOS and SOV features and join onto df.

        Args:
            df: Modeling DataFrame with GAME_ID, TEAM_A, TEAM_B, YEAR,
                WEEK_NUM columns.
            datasets: Provides games() and elo_state() datasets.

        Returns:
            Input DataFrame with TEAM_A_SOS, TEAM_A_SOV, TEAM_B_SOS,
            TEAM_B_SOV columns appended.
        """
        games: pd.DataFrame = datasets.games()
        elo: pd.DataFrame = datasets.elo_state()
        sos_sov_table: DataFrame = _build_sos_sov_table(games, elo)

        df = df.copy()
        df["WEEK_NUM"] = df["WEEK_NUM"].astype(int)

        if sos_sov_table.empty:
            for col in _PRODUCES:
                df[col] = float("nan")
            return df

        # Two vectorized merges replace four df.apply() calls
        # (schedule_strength/M1).
        for prefix, team_col in (("TEAM_A", "TEAM_A"), ("TEAM_B", "TEAM_B")):
            renamed: DataFrame = sos_sov_table.rename(
                columns={
                    "TEAM": team_col,
                    **{c: f"{prefix}_{c}" for c in _STAT_COLS},
                }
            )
            df = df.merge(
                renamed,
                how="left",
                on=[team_col, "YEAR", "WEEK_NUM"],
            )

        return df


# ---------------------------------------------------------------------------
# SOS / SOV computation (vectorized)
# ---------------------------------------------------------------------------


def _build_sos_sov_table(
    games: pd.DataFrame,
    elo: pd.DataFrame,
) -> pd.DataFrame:
    """Build a vectorized SOS/SOV lookup table.

    Returns a DataFrame with one row per (TEAM, YEAR, WEEK_NUM) capturing
    the team's strength of schedule and strength of victory *entering*
    that week. Empty DataFrame when input is empty or missing required
    columns.

    Columns:
        TEAM, YEAR, WEEK_NUM, SOS, SOV
    """
    required_games: set[str] = {"WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"}
    required_elo: set[str] = {"NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"}
    if (
        games.empty
        or elo.empty
        or not required_games.issubset(games.columns)
        or not required_elo.issubset(elo.columns)
    ):
        return pd.DataFrame(columns=["TEAM", "YEAR", "WEEK_NUM", *_STAT_COLS])

    completed: DataFrame = games.loc[games["WIN_OR_TIE"].notna(), :].copy()
    completed["WEEK_NUM"] = completed["WEEK_NUM"].astype(int)

    # Long format: one row per team per game with opponent and result.
    winner_rows = completed.loc[:, ["WINNER", "LOSER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].rename(
        columns={"WINNER": "TEAM", "LOSER": "OPPONENT", "WIN_OR_TIE": "RESULT"}
    )
    loser_rows = completed.loc[:, ["LOSER", "WINNER", "YEAR", "WEEK_NUM", "WIN_OR_TIE"]].copy()
    loser_rows["RESULT"] = 1.0 - loser_rows["WIN_OR_TIE"]
    loser_rows = loser_rows.rename(columns={"LOSER": "TEAM", "WINNER": "OPPONENT"}).drop(
        columns=["WIN_OR_TIE"]
    )

    long: DataFrame = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # Merge opponent's pre-game Elo onto the long table.
    # Replaces the dict-based row-by-row lookup (schedule_strength/M1).
    elo_for_join: DataFrame = elo.loc[:, ["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"]].rename(
        columns={
            "NFL_TEAM": "OPPONENT",
            "NFL_YEAR": "YEAR",
            "NFL_WEEK": "WEEK_NUM",
            "ELO": "OPP_ELO",
        }
    )
    elo_for_join["WEEK_NUM"] = elo_for_join["WEEK_NUM"].astype(int)
    elo_for_join["OPPONENT"] = elo_for_join["OPPONENT"].astype(str)

    long["OPPONENT"] = long["OPPONENT"].astype(str)
    long = long.merge(
        elo_for_join,
        how="left",
        on=["OPPONENT", "YEAR", "WEEK_NUM"],
    )

    # Sort by team-season-week so cumsum/shift work chronologically.
    long = long.sort_values(["TEAM", "YEAR", "WEEK_NUM"], ignore_index=True)

    # Build numerator/denominator components for SOS and SOV.
    # OPP_ELO is NaN when the opponent's Elo for that week isn't in the
    # state table; those games are excluded from both averages.
    has_opp_elo: pd.Series = long["OPP_ELO"].notna()
    is_win: pd.Series = long["RESULT"] >= 1.0

    long["_OPP_ELO_FOR_SOS"] = long["OPP_ELO"].where(has_opp_elo, 0.0)
    long["_SOS_DENOM_INC"] = has_opp_elo.astype(int)

    win_with_elo: pd.Series = has_opp_elo & is_win
    long["_OPP_ELO_FOR_SOV"] = long["OPP_ELO"].where(win_with_elo, 0.0)
    long["_SOV_DENOM_INC"] = win_with_elo.astype(int)

    grouped = long.groupby(["TEAM", "YEAR"], sort=False)

    # Cumulative sums then shift(1) - "entering this game" semantics.
    sos_num = (
        grouped["_OPP_ELO_FOR_SOS"]
        .cumsum()
        .groupby([long["TEAM"], long["YEAR"]])
        .shift(1)
        .fillna(0.0)
    )
    sos_den = (
        grouped["_SOS_DENOM_INC"].cumsum().groupby([long["TEAM"], long["YEAR"]]).shift(1).fillna(0)
    )
    sov_num = (
        grouped["_OPP_ELO_FOR_SOV"]
        .cumsum()
        .groupby([long["TEAM"], long["YEAR"]])
        .shift(1)
        .fillna(0.0)
    )
    sov_den = (
        grouped["_SOV_DENOM_INC"].cumsum().groupby([long["TEAM"], long["YEAR"]]).shift(1).fillna(0)
    )

    long["SOS"] = np.where(sos_den > 0, sos_num / sos_den.replace(0, np.nan), np.nan)
    long["SOV"] = np.where(sov_den > 0, sov_num / sov_den.replace(0, np.nan), np.nan)

    return long.loc[:, ["TEAM", "YEAR", "WEEK_NUM", *_STAT_COLS]].reset_index(drop=True)


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
