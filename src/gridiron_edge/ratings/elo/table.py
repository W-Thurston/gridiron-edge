# src/gridiron_edge/ratings/elo/table.py

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from gridiron_edge.ratings.elo.core import update_elo


@dataclass(frozen=True)
class EloTableConfig:
    """Configuration parameters for Elo table construction.

    Attributes:
        k: Elo K-factor controlling rating update magnitude per game.
        initial_elo: Starting Elo assigned to all teams at season zero.
        expansion_elo: Starting Elo assigned to expansion franchises in
            their inaugural season.
        offseason_regress_frac: Fraction of the gap between a team's
            end-of-season Elo and the league mean to revert each offseason.
    """

    k: float = 20.0
    initial_elo: float = 1500.0
    expansion_elo: float = 1300.0
    offseason_regress_frac: float = 1 / 3.0


EXPANSION_START_YEAR: dict[str, str] = {
    "Carolina Panthers": "1995-1996",
    "Jacksonville Jaguars": "1995-1996",
    "Baltimore Ravens": "1996-1997",
    "Houston Texans": "2002-2003",
}


def _build_years(df: pd.DataFrame) -> list[str]:
    max_year = df["YEAR"].max()
    if df.loc[df["YEAR"] == max_year, "WEEK_NUM"].max() == 22:
        now: datetime = datetime.now(tz=UTC)
        return [*sorted(df["YEAR"].unique().tolist()), f"{now.year}-{now.year + 1}"]
    return sorted(df["YEAR"].unique().tolist())


def _season_weeks(df: pd.DataFrame) -> list[int]:
    return sorted(df["WEEK_NUM"].unique().tolist())


def build_elo_state_table_all_years(
    games: pd.DataFrame,
    *,
    cfg: EloTableConfig | None = None,
) -> pd.DataFrame:
    """Legacy-identical Elo table construction for all years.

    Input `games` must contain:
      YEAR, WEEK_NUM, WINNER, LOSER, WIN_OR_TIE, GAME_DATE

    Output columns:
      NFL_TEAM, NFL_YEAR, NFL_WEEK, ELO
    """
    cfg = cfg or EloTableConfig()

    df: pd.DataFrame = games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    nfl_teams: list[str] = sorted(df["WINNER"].unique().tolist())
    nfl_years: list[str] = _build_years(df)
    nfl_weeks: list[int] = _season_weeks(df)

    sorted_years: list[str] = nfl_years.copy()
    increase_weeks_idx: int = (
        sorted_years.index("2021-2022") if "2021-2022" in sorted_years else len(sorted_years)
    )

    # Build the cartesian table of team x year x week
    records: list[tuple[str, str, int]] = []
    for team in nfl_teams:
        for year in nfl_years:
            for week in nfl_weeks:
                records.append((team, year, week))

    df_team_elo: pd.DataFrame = pd.DataFrame.from_records(
        records,
        columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
    ).drop_duplicates()
    df_team_elo["ELO"] = np.nan

    # Initialize: week 1 of first year = 1500
    df_team_elo.loc[
        (df_team_elo["NFL_YEAR"] == sorted_years[0]) & (df_team_elo["NFL_WEEK"] == 1),
        "ELO",
    ] = cfg.initial_elo

    # Make year categorical with explicit ordering
    df_team_elo["NFL_YEAR"] = pd.Categorical(df_team_elo["NFL_YEAR"], sorted_years)

    # Remove week 22 prior to 2021-2022 (legacy: seasons_with_21_weeks excludes 1993-1994 special case)
    seasons_with_21_weeks: list[str] = sorted_years[:increase_weeks_idx]
    if "1993-1994" in seasons_with_21_weeks:
        seasons_with_21_weeks.remove("1993-1994")

    df_team_elo = df_team_elo.loc[
        ~((df_team_elo["NFL_YEAR"].isin(seasons_with_21_weeks)) & (df_team_elo["NFL_WEEK"] == 22))
    ]

    # Sort
    df_team_elo = df_team_elo.sort_values(
        ["NFL_YEAR", "NFL_WEEK", "NFL_TEAM"],
        ignore_index=True,
    )

    # Drop rows for teams before they existed (legacy hardcoded)
    idx_to_drop: list[int] = []
    for team, start_year in EXPANSION_START_YEAR.items():
        if start_year in sorted_years:
            drop_years: list[str] = sorted_years[: sorted_years.index(start_year)]
            idx = df_team_elo[
                (df_team_elo["NFL_TEAM"] == team) & (df_team_elo["NFL_YEAR"].isin(drop_years))
            ].index
            idx_to_drop.extend(idx.tolist())
    if idx_to_drop:
        df_team_elo = df_team_elo.drop(idx_to_drop)

    # Fill Elo week-to-week
    k: float = cfg.k

    print("> Filling out ELO column")
    for idx, curr_year in enumerate(tqdm(sorted_years)):
        next_year: str | None = sorted_years[idx + 1] if idx < len(sorted_years) - 1 else None

        number_of_weeks_in_curr_year: int = len(
            df.loc[df["YEAR"] == curr_year, "WEEK_NUM"].unique(),
        )
        teams_this_season = df_team_elo.loc[
            df_team_elo["NFL_YEAR"] == curr_year,
            "NFL_TEAM",
        ].unique()

        for wk in range(1, number_of_weeks_in_curr_year + 1):
            # iterate each game this week
            week_games: pd.DataFrame = df.loc[
                (df["WEEK_NUM"] == wk) & (df["YEAR"] == curr_year),
                ["WINNER", "LOSER", "WIN_OR_TIE"],
            ]
            for _, row in week_games.iterrows():
                winning_team_name = row["WINNER"]
                losing_team_name = row["LOSER"]

                winner_prev = df_team_elo.loc[
                    (df_team_elo["NFL_TEAM"] == winning_team_name)
                    & (df_team_elo["NFL_YEAR"] == curr_year)
                    & (df_team_elo["NFL_WEEK"] == wk),
                    "ELO",
                ].values[0]

                loser_prev = df_team_elo.loc[
                    (df_team_elo["NFL_TEAM"] == losing_team_name)
                    & (df_team_elo["NFL_YEAR"] == curr_year)
                    & (df_team_elo["NFL_WEEK"] == wk),
                    "ELO",
                ].values[0]

                winner_elo, loser_elo = update_elo(
                    winner_prev,
                    loser_prev,
                    win_or_tie=float(row["WIN_OR_TIE"]),
                    k=k,
                )

                # Update next year's wk1 if end of year, else next week in same year
                if (wk == number_of_weeks_in_curr_year) and (next_year is not None):
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == winning_team_name)
                        & (df_team_elo["NFL_YEAR"] == next_year)
                        & (df_team_elo["NFL_WEEK"] == 1),
                        "ELO",
                    ] = winner_elo
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == losing_team_name)
                        & (df_team_elo["NFL_YEAR"] == next_year)
                        & (df_team_elo["NFL_WEEK"] == 1),
                        "ELO",
                    ] = loser_elo
                else:
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == winning_team_name)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                        "ELO",
                    ] = winner_elo
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == losing_team_name)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                        "ELO",
                    ] = loser_elo

            # Handle byes / no game / playoff missing (ffill logic identical)
            if (wk == number_of_weeks_in_curr_year) and (next_year is not None):
                null_idx = df_team_elo.loc[
                    (df_team_elo["NFL_YEAR"] == next_year) & (df_team_elo["NFL_WEEK"] == 1),
                    "ELO",
                ].isnull()

                teams_missing = (
                    df_team_elo.loc[
                        (df_team_elo["NFL_YEAR"] == next_year) & (df_team_elo["NFL_WEEK"] == 1),
                    ]
                    .loc[null_idx, "NFL_TEAM"]
                    .unique()
                )

                for team in teams_missing:
                    if team not in teams_this_season:
                        df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == next_year)
                            & (df_team_elo["NFL_WEEK"] == 1),
                            "ELO",
                        ] = cfg.expansion_elo
                    else:
                        df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == next_year)
                            & (df_team_elo["NFL_WEEK"] == 1),
                            "ELO",
                        ] = df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == curr_year)
                            & (df_team_elo["NFL_WEEK"] == number_of_weeks_in_curr_year),
                            "ELO",
                        ].values[0]
            else:
                null_idx = df_team_elo.loc[
                    (df_team_elo["NFL_YEAR"] == curr_year) & (df_team_elo["NFL_WEEK"] == wk + 1),
                    "ELO",
                ].isna()

                teams_missing = (
                    df_team_elo.loc[
                        (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                    ]
                    .loc[null_idx, "NFL_TEAM"]
                    .unique()
                )

                for team in teams_missing:
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == team)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"].isin([wk, wk + 1])),
                        "ELO",
                    ] = df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == team)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"].isin([wk, wk + 1])),
                        "ELO",
                    ].ffill()

        # Offseason regression step (identical)
        if next_year is not None:
            curr_season_mean_elo = df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ].mean()

            frac: float = cfg.offseason_regress_frac
            df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ] = curr_season_mean_elo * frac + df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ] * (1 - frac)

    return df_team_elo.reset_index(drop=True)


def _last_fully_filled_week(df_team_elo: pd.DataFrame) -> tuple[str, int]:
    """Find the last (NFL_YEAR, NFL_WEEK) where ALL teams in that yaer have an Elo."""
    # Ensure ordering
    df: pd.DataFrame = df_team_elo.sort_values(["NFL_YEAR", "NFL_WEEK", "NFL_TEAM"]).copy()

    # Determine "teams per year" from the table itself (handles expansion eras)
    teams_per_year = df.groupby("NFL_YEAR")["NFL_TEAM"].nunique()

    # Count non-null Elo entries per (year, week)
    filled = (
        df.dropna(subset=["ELO"])
        .groupby(["NFL_YEAR", "NFL_WEEK"])["NFL_TEAM"]
        .nunique()
        .rename("n_filled")
        .reset_index()
    )

    filled["n_teams_in_year"] = filled["NFL_YEAR"].map(teams_per_year)
    full_weeks = filled[filled["n_filled"] == filled["n_teams_in_year"]]

    if full_weeks.empty:
        # If nothing is fully filled, default to the very first year/week.
        first = df.iloc[0]
        return str(first["NFL_YEAR"]), int(first["NFL_WEEK"])

    last_row = full_weeks.sort_values(["NFL_YEAR", "NFL_WEEK"]).iloc[-1]
    return str(last_row["NFL_YEAR"]), int(last_row["NFL_WEEK"])


def update_elo_state_table_incremental(
    *,
    games: pd.DataFrame,
    elo_state_existing: pd.DataFrame,
    cfg: EloTableConfig | None = None,
) -> pd.DataFrame:
    """Incrementally update an existing Elo state table forward from the last fully-filled week.

    Assumptions:
      - elo_state_existing has columns: NFL_TEAM, NFL_YEAR, NFL_WEEK, ELO
      - games has columns: YEAR, WEEK_NUM, WINNER, LOSER, WIN_OR_TIE, GAME_DATE

    Returns:
      Updated elo_state table (same shape, with new weeks filled).

    """
    cfg = cfg or EloTableConfig()

    df_games: pd.DataFrame = games.copy()
    df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"])

    df_team_elo: pd.DataFrame = elo_state_existing.copy()

    # 1) Find where to start
    start_year, start_week = _last_fully_filled_week(df_team_elo)

    # 2) Determine chronological ordering of years we actually need to process
    # Use ordering from the existing table categories if present, else fall back to sorted unique.
    if isinstance(df_team_elo["NFL_YEAR"].dtype, pd.CategoricalDtype):
        years_order: list = list(df_team_elo["NFL_YEAR"].dtype.categories)
    else:
        years_order = sorted(df_team_elo["NFL_YEAR"].astype(str).unique().tolist())

    # Safety: make sure start_year is in order
    if start_year not in years_order:
        years_order = sorted({*years_order, start_year})

    start_year_idx: int = years_order.index(start_year)

    # 3) Iterate from (start_year, start_week) forward
    k: float = cfg.k

    for y_idx in range(start_year_idx, len(years_order)):
        curr_year = years_order[y_idx]
        next_year = years_order[y_idx + 1] if y_idx < len(years_order) - 1 else None

        # If we're past the range of game data, nothing to do
        if curr_year not in df_games["YEAR"].unique():
            continue

        weeks_in_year = sorted(
            df_games.loc[df_games["YEAR"] == curr_year, "WEEK_NUM"].unique().tolist(),
        )
        if not weeks_in_year:
            continue

        # If we are in the first year being processed, start after start_week;
        # otherwise start at week 1.
        first_week_to_process: int = (start_week if curr_year == start_year else 0) + 1

        # Teams “in this season” (for offseason regression and expansion handling)
        teams_this_season = df_team_elo.loc[
            df_team_elo["NFL_YEAR"] == curr_year,
            "NFL_TEAM",
        ].unique()

        for wk in weeks_in_year:
            if wk < first_week_to_process:
                continue

            # Apply updates for all games played in (curr_year, wk)
            week_games = df_games.loc[
                (df_games["YEAR"] == curr_year) & (df_games["WEEK_NUM"] == wk),
                ["WINNER", "LOSER", "WIN_OR_TIE"],
            ]

            for _, row in week_games.iterrows():
                winner = row["WINNER"]
                loser = row["LOSER"]

                winner_prev = df_team_elo.loc[
                    (df_team_elo["NFL_TEAM"] == winner)
                    & (df_team_elo["NFL_YEAR"] == curr_year)
                    & (df_team_elo["NFL_WEEK"] == wk),
                    "ELO",
                ].values[0]

                loser_prev = df_team_elo.loc[
                    (df_team_elo["NFL_TEAM"] == loser)
                    & (df_team_elo["NFL_YEAR"] == curr_year)
                    & (df_team_elo["NFL_WEEK"] == wk),
                    "ELO",
                ].values[0]

                winner_elo, loser_elo = update_elo(
                    winner_prev,
                    loser_prev,
                    win_or_tie=float(row["WIN_OR_TIE"]),
                    k=k,
                )

                # Write to next week or next season week 1
                last_week_of_year = max(weeks_in_year)
                if (wk == last_week_of_year) and (next_year is not None):
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == winner)
                        & (df_team_elo["NFL_YEAR"] == next_year)
                        & (df_team_elo["NFL_WEEK"] == 1),
                        "ELO",
                    ] = winner_elo
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == loser)
                        & (df_team_elo["NFL_YEAR"] == next_year)
                        & (df_team_elo["NFL_WEEK"] == 1),
                        "ELO",
                    ] = loser_elo
                else:
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == winner)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                        "ELO",
                    ] = winner_elo
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == loser)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                        "ELO",
                    ] = loser_elo

            # Bye / missing game ffill, matching your rebuild logic
            last_week_of_year = max(weeks_in_year)
            if (wk == last_week_of_year) and (next_year is not None):
                mask_null = df_team_elo.loc[
                    (df_team_elo["NFL_YEAR"] == next_year) & (df_team_elo["NFL_WEEK"] == 1),
                    "ELO",
                ].isna()

                teams_missing = (
                    df_team_elo.loc[
                        (df_team_elo["NFL_YEAR"] == next_year) & (df_team_elo["NFL_WEEK"] == 1),
                        ["NFL_TEAM"],
                    ]
                    .loc[mask_null, "NFL_TEAM"]
                    .unique()
                )

                for team in teams_missing:
                    if team not in teams_this_season:
                        df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == next_year)
                            & (df_team_elo["NFL_WEEK"] == 1),
                            "ELO",
                        ] = cfg.expansion_elo
                    else:
                        df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == next_year)
                            & (df_team_elo["NFL_WEEK"] == 1),
                            "ELO",
                        ] = df_team_elo.loc[
                            (df_team_elo["NFL_TEAM"] == team)
                            & (df_team_elo["NFL_YEAR"] == curr_year)
                            & (df_team_elo["NFL_WEEK"] == last_week_of_year),
                            "ELO",
                        ].values[0]
            else:
                mask_null = df_team_elo.loc[
                    (df_team_elo["NFL_YEAR"] == curr_year) & (df_team_elo["NFL_WEEK"] == wk + 1),
                    "ELO",
                ].isna()

                teams_missing = (
                    df_team_elo.loc[
                        (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"] == wk + 1),
                        ["NFL_TEAM"],
                    ]
                    .loc[mask_null, "NFL_TEAM"]
                    .unique()
                )

                for team in teams_missing:
                    df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == team)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"].isin([wk, wk + 1])),
                        "ELO",
                    ] = df_team_elo.loc[
                        (df_team_elo["NFL_TEAM"] == team)
                        & (df_team_elo["NFL_YEAR"] == curr_year)
                        & (df_team_elo["NFL_WEEK"].isin([wk, wk + 1])),
                        "ELO",
                    ].ffill()

        # Offseason regression at transition to next_year (if next_year exists in table)
        if next_year is not None and next_year in df_team_elo["NFL_YEAR"].astype(str).unique():
            curr_season_mean_elo = df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ].mean()

            frac = cfg.offseason_regress_frac
            df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ] = curr_season_mean_elo * frac + df_team_elo.loc[
                (df_team_elo["NFL_YEAR"] == next_year)
                & (df_team_elo["NFL_WEEK"] == 1)
                & (df_team_elo["NFL_TEAM"].isin(teams_this_season)),
                "ELO",
            ] * (1 - frac)

    return df_team_elo.sort_values(["NFL_YEAR", "NFL_WEEK", "NFL_TEAM"]).reset_index(
        drop=True,
    )
