# src/gridiron_edge/features/team/rest.py

"""Rest and schedule stress features.

Computes preparation-time features for each team in a matchup.  These
capture fatigue and scheduling effects that EPA rolling averages cannot
reflect — EPA measures what happened in past games, not the physical and
logistical state each team enters the next game with.

Produces (symmetric — computed for TEAM_A perspective in each row):

    TEAM_A_DAYS_REST    int     Days since TEAM_A's last game.
                                Typical: 6-7 (standard week).
                                Short:   3-4 (Thursday or international game).
                                Bye:     13-14.
                                Week 1:  NaN (no prior game).

    TEAM_B_DAYS_REST    int     Same for TEAM_B.

    TEAM_A_SHORT_WEEK   int     1 if TEAM_A_DAYS_REST < 6, else 0.
                                Flags Thursday games, international travel
                                weeks, and any other compressed schedule.

    TEAM_B_SHORT_WEEK   int     Same for TEAM_B.

    TEAM_A_POST_BYE     int     1 if TEAM_A_DAYS_REST >= 13, else 0.
                                Post-bye teams are the inverse of short-week:
                                extra preparation time, typically entering
                                the game fresher.

    TEAM_B_POST_BYE     int     Same for TEAM_B.

    TEAM_A_REST_DIFF    int     TEAM_A_DAYS_REST - TEAM_B_DAYS_REST.
                                Positive means TEAM_A is more rested.
                                Captures the relative rest advantage
                                in a single feature rather than requiring
                                the model to learn the subtraction.
    TEAM_B_REST_DIFF    int     Same subtraction from TEAM_B perspective.
                                Always equals -TEAM_A_REST_DIFF.

Design notes:
    - Week 1 produces NaN for DAYS_REST (no prior game in the dataset).
      NaN rows are excluded by _prepare_data in all model training paths,
      so this is handled correctly downstream without special-casing here.
    - The two-row-per-game design means this feature is computed twice
      (once for each team's perspective), which is correct: in row A,
      TEAM_A is the winner; in row B, TEAM_A is the loser.  Rest days
      are a property of the team, not the game, so both rows get the
      correct team-specific value.
    - Games are ordered by (YEAR, WEEK_NUM) within each team's history.
      This is correct because the canonical games CSV stores week numbers
      as integers within the season, and seasons are string-labelled
      (e.g. "2025-2026") — sorting on YEAR then WEEK_NUM gives the right
      chronological order.
"""

from __future__ import annotations

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


@FeatureRegistry.register("rest")
class RestFeature:
    """Schedule stress features: days rest, short-week flag, post-bye flag.

    Computes per-team rest days by computing the gap between each game's
    date and the team's previous game date within the same dataset.
    Binary flags for short weeks (<6 days) and post-bye weeks (13+ days)
    are derived directly from the rest-day count.

    Requires ``GAME_DATE`` in the games dataset (stored as "YYYY-MM-DD").
    """

    spec = FeatureSpec(
        name="rest",
        produces=[
            "TEAM_A_DAYS_REST",
            "TEAM_B_DAYS_REST",
            "TEAM_A_SHORT_WEEK",
            "TEAM_B_SHORT_WEEK",
            "TEAM_A_POST_BYE",
            "TEAM_B_POST_BYE",
            "TEAM_A_REST_DIFF",
            "TEAM_B_REST_DIFF",
        ],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute rest features and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with GAME_ID, TEAM_A, TEAM_B, YEAR,
                WEEK_NUM columns.
            datasets: Provides ``games()`` with GAME_DATE, WINNER, LOSER,
                YEAR, WEEK_NUM, GAME_ID columns.

        Returns:
            Input DataFrame with six rest/schedule columns appended.
            Rows where GAME_DATE is missing or unparseable receive NaN
            rest columns (excluded from model training by _prepare_data).
        """
        games: DataFrame = datasets.games()

        # Build a long-format team-game table: one row per (team, game)
        # using both WINNER and LOSER so every team appears once per game.
        # We join GAME_DATE from the games table.
        needed: list[str] = ["GAME_ID", "WINNER", "LOSER", "YEAR", "WEEK_NUM", "GAME_DATE"]
        g = games.loc[:, needed].copy()

        # Parse date — store as date (not datetime) for day-diff arithmetic
        # pyrefly: ignore [missing-attribute]
        g["_DATE"] = pd.to_datetime(g["GAME_DATE"], format="%Y-%m-%d", errors="coerce").dt.date

        # Winner perspective
        w = g[["GAME_ID", "WINNER", "YEAR", "WEEK_NUM", "_DATE"]].rename(columns={"WINNER": "TEAM"})
        # Loser perspective
        lo = g[["GAME_ID", "LOSER", "YEAR", "WEEK_NUM", "_DATE"]].rename(columns={"LOSER": "TEAM"})

        team_games = (
            pd.concat([w, lo], ignore_index=True)
            .drop_duplicates(subset=["GAME_ID", "TEAM"])
            .sort_values(["TEAM", "YEAR", "WEEK_NUM"])
            .reset_index(drop=True)
        )

        # Compute days since previous game per team (shift within team group)
        team_games["_PREV_DATE"] = team_games.groupby("TEAM")["_DATE"].shift(1)
        team_games["_DAYS_REST"] = team_games.apply(
            lambda r: (
                (r["_DATE"] - r["_PREV_DATE"]).days
                if pd.notna(r["_PREV_DATE"]) and pd.notna(r["_DATE"])
                else float("nan")
            ),
            axis=1,
        )

        # Derive binary flags
        team_games["_SHORT_WEEK"] = team_games["_DAYS_REST"].apply(
            lambda x: int(x < _SHORT_WEEK_THRESHOLD) if pd.notna(x) else float("nan")
        )
        team_games["_POST_BYE"] = team_games["_DAYS_REST"].apply(
            lambda x: int(x >= _BYE_WEEK_THRESHOLD) if pd.notna(x) else float("nan")
        )

        rest_lookup = team_games[
            ["GAME_ID", "TEAM", "_DAYS_REST", "_SHORT_WEEK", "_POST_BYE"]
        ].copy()

        # --- Merge for TEAM_A ---
        df = df.merge(
            rest_lookup.rename(
                columns={
                    "TEAM": "TEAM_A",
                    "_DAYS_REST": "TEAM_A_DAYS_REST",
                    "_SHORT_WEEK": "TEAM_A_SHORT_WEEK",
                    "_POST_BYE": "TEAM_A_POST_BYE",
                }
            ),
            how="left",
            on=["GAME_ID", "TEAM_A"],
        )

        # --- Merge for TEAM_B ---
        df = df.merge(
            rest_lookup.rename(
                columns={
                    "TEAM": "TEAM_B",
                    "_DAYS_REST": "TEAM_B_DAYS_REST",
                    "_SHORT_WEEK": "TEAM_B_SHORT_WEEK",
                    "_POST_BYE": "TEAM_B_POST_BYE",
                }
            ),
            how="left",
            on=["GAME_ID", "TEAM_B"],
        )

        # -- Rest differential (continuous, no binning) --
        df["TEAM_A_REST_DIFF"] = df["TEAM_A_DAYS_REST"] - df["TEAM_B_DAYS_REST"]
        df["TEAM_B_REST_DIFF"] = df["TEAM_B_DAYS_REST"] - df["TEAM_A_DAYS_REST"]

        return df
