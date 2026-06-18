# tests/fixtures/dataframes.py
"""Centralized DataFrame factory functions for Gridiron Edge tests.

Consolidates the repeated _make_games(), _make_eval_df(), _make_predictions()
patterns from individual test files into a single importable module.

Usage::

    from tests.fixtures.dataframes import make_games, make_modeling_rows


    def test_something():
        games = make_games([{"WINNER": "KC", "LOSER": "LV"}])
        modeling = make_modeling_rows([{"TEAM_A": "KC", "TEAM_B": "LV"}])
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

_GAME_DEFAULTS: dict[str, Any] = {
    "GAME_ID": "2024_01_KC_LV",
    "WINNER": "Kansas City Chiefs",
    "LOSER": "Las Vegas Raiders",
    "WIN_OR_TIE": 1,
    "YEAR": "2024-2025",
    "WEEK_NUM": 1,
    "GAME_DATE": "2024-09-05",
    "GAME_LOCATION": "NULL_VALUE",
    "STADIUM": "Arrowhead Stadium",
    "ROOF": "outdoors",
    "SURFACE": "grass",
    "GAMETIME": "20:20",
    "GAME_DAY_OF_WEEK": "Thursday",
    "PTS_WINNER": 27,
    "PTS_LOSER": 20,
    "VEGAS_LINE": -3.5,
    "OVER_UNDER": 47.5,
    "FAVORITED": "Kansas City Chiefs",
}


def make_games(
    overrides: list[dict[str, Any]] | None = None,
    *,
    n: int = 2,
) -> pd.DataFrame:
    """Build a minimal games DataFrame.

    Args:
        overrides: Per-row field overrides.  If *None*, creates *n* rows
            using defaults with auto-incremented GAME_IDs.
        n: Number of rows when *overrides* is not provided.

    Returns:
        DataFrame matching the canonical games schema.
    """
    if overrides is not None:
        rows: list[dict[str, Any]] = [{**_GAME_DEFAULTS, **o} for o in overrides]
    else:
        rows = [{**_GAME_DEFAULTS, "GAME_ID": f"2024_0{i + 1}_KC_LV"} for i in range(n)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modeling rows (two-row-per-game structure)
# ---------------------------------------------------------------------------

_MODELING_DEFAULTS: dict[str, Any] = {
    "GAME_ID": "2024_01_KC_LV",
    "TEAM_A": "Kansas City Chiefs",
    "TEAM_B": "Las Vegas Raiders",
    "YEAR": "2024-2025",
    "WEEK_NUM": 1,
    "RESULT": 1,
    "HOME_FIELD": 1,
}


def make_modeling_rows(
    overrides: list[dict[str, Any]] | None = None,
    *,
    n: int = 2,
) -> pd.DataFrame:
    """Build a minimal modeling DataFrame (TEAM_A / TEAM_B structure).

    Args:
        overrides: Per-row field overrides.
        n: Number of rows when *overrides* is not provided.

    Returns:
        DataFrame matching the modeling schema.
    """
    if overrides is not None:
        rows: list[dict[str, Any]] = [{**_MODELING_DEFAULTS, **o} for o in overrides]
    else:
        rows = [
            {
                **_MODELING_DEFAULTS,
                "GAME_ID": f"2024_0{i + 1}_KC_LV",
                "TEAM_A": "Kansas City Chiefs" if i % 2 == 0 else "Las Vegas Raiders",
                "TEAM_B": "Las Vegas Raiders" if i % 2 == 0 else "Kansas City Chiefs",
                "HOME_FIELD": 1 if i % 2 == 0 else 0,
            }
            for i in range(n)
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stadiums
# ---------------------------------------------------------------------------

_STADIUM_DEFAULTS: list[dict[str, Any]] = [
    {
        "HOME_TEAM": "Team A",
        "YEAR": "2025-2026",
        "STADIUM": "Stadium A",
        "LATITUDE": 40.0,
        "LONGITUDE": -75.0,
        "ALTITUDE": 10,
    },
    {
        "HOME_TEAM": "Team B",
        "YEAR": "2025-2026",
        "STADIUM": "Stadium B",
        "LATITUDE": 34.0,
        "LONGITUDE": -118.0,
        "ALTITUDE": 50,
    },
]


def make_stadiums(
    overrides: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a minimal stadiums reference DataFrame.

    Args:
        overrides: Complete row list override.  If *None*, uses two-team
            defaults (Team A in the east, Team B in the west).

    Returns:
        DataFrame matching the stadium reference schema.
    """
    rows: list[dict[str, Any]] = overrides if overrides is not None else _STADIUM_DEFAULTS
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Elo state
# ---------------------------------------------------------------------------


def make_elo_state(
    teams: list[str] | None = None,
    *,
    year: str = "2025-2026",
    weeks: int = 3,
    base_elo: float = 1500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic Elo state table.

    Args:
        teams: Team names.  Defaults to ``["Team A", "Team B"]``.
        year: Season identifier.
        weeks: Number of weeks to generate.
        base_elo: Starting Elo.
        seed: RNG seed for small perturbations.

    Returns:
        DataFrame with columns NFL_TEAM, NFL_YEAR, NFL_WEEK, ELO.
    """
    teams = teams or ["Team A", "Team B"]
    rng: Generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for team in teams:
        elo: float = base_elo
        for week in range(1, weeks + 1):
            elo += rng.uniform(-20, 20)
            rows.append(
                {
                    "NFL_TEAM": team,
                    "NFL_YEAR": year,
                    "NFL_WEEK": week,
                    "ELO": round(elo, 1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EPA by game
# ---------------------------------------------------------------------------


def make_epa_by_game(
    *,
    teams: list[str] | None = None,
    seasons: list[int] | None = None,
    weeks_per_season: int = 18,
    seed: int = 1,
) -> pd.DataFrame:
    """Build a synthetic game-level EPA DataFrame.

    Args:
        teams: Team abbreviations.  Defaults to 8 common teams.
        seasons: Season years.  Defaults to ``[2006, 2007, 2008, 2023, 2024]``.
        weeks_per_season: Games per team per season.
        seed: RNG seed.

    Returns:
        DataFrame matching the epa_by_game schema.
    """
    teams = teams or ["KC", "SF", "BUF", "PHI", "DAL", "NYG", "MIA", "LAR"]
    seasons = seasons or [2006, 2007, 2008, 2023, 2024]
    rng: Generator = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            for team in teams:
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{team}_OPP",
                        "season": season,
                        "week": week,
                        "team": team,
                        "off_epa_per_play": rng.uniform(-0.2, 0.3),
                        "off_pass_epa": rng.uniform(-0.3, 0.4),
                        "off_rush_epa": rng.uniform(-0.2, 0.2),
                        "off_success_rate": rng.uniform(0.3, 0.6),
                        "off_explosive_rate": rng.uniform(0.03, 0.15),
                        # After off_success_rate:
                        "off_pass_success_rate": rng.uniform(0.30, 0.60),
                        "off_rush_success_rate": rng.uniform(0.30, 0.50),
                        # After off_explosive_rate:
                        "off_third_down_pct": rng.uniform(0.25, 0.50),
                        "off_redzone_td_pct": rng.uniform(0.40, 0.70),
                        "off_turnover_rate": rng.uniform(0.01, 0.06),
                        "off_sack_rate": rng.uniform(0.04, 0.10),
                        "def_explosive_rate": rng.uniform(0.03, 0.15),
                        "def_epa_per_play": rng.uniform(-0.3, 0.2),
                        "def_pass_epa": rng.uniform(-0.4, 0.3),
                        "def_rush_epa": rng.uniform(-0.2, 0.2),
                        "def_success_rate": rng.uniform(0.3, 0.6),
                        # After def_success_rate:
                        "def_pass_success_rate": rng.uniform(0.30, 0.60),
                        "def_rush_success_rate": rng.uniform(0.30, 0.50),
                        # After def_explosive_rate:
                        "def_third_down_pct": rng.uniform(0.25, 0.50),
                        "def_redzone_td_pct": rng.uniform(0.40, 0.70),
                        "def_turnover_rate": rng.uniform(0.01, 0.06),
                        "def_sack_rate": rng.uniform(0.04, 0.10),
                        "off_plays": rng.integers(50, 80),
                        "off_yards_per_play": rng.uniform(4.0, 7.0),
                        "off_redzone_attempts": rng.integers(2, 10),
                        "off_int_rate": rng.uniform(0.01, 0.05),
                        "off_penalty_rate": rng.uniform(0.02, 0.08),
                        "off_avg_score_diff": rng.uniform(-10.0, 10.0),
                        "off_close_game_pct": rng.uniform(0.3, 0.8),
                        "def_plays": rng.integers(50, 80),
                        "def_yards_per_play": rng.uniform(4.0, 7.0),
                        "def_redzone_attempts": rng.integers(2, 10),
                        "def_int_rate": rng.uniform(0.01, 0.05),
                        "def_penalty_rate": rng.uniform(0.02, 0.08),
                        "def_avg_score_diff": rng.uniform(-10.0, 10.0),
                        "def_close_game_pct": rng.uniform(0.3, 0.8),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weather enriched
# ---------------------------------------------------------------------------


def make_weather_enriched(
    overrides: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a minimal weather-enriched DataFrame.

    Args:
        overrides: Per-row field overrides.  If *None*, creates a single
            row with mild outdoor conditions.

    Returns:
        DataFrame matching the weather_enriched schema.
    """
    defaults: dict[str, Any] = {
        "GAME_ID": "2024_01_KC_LV",
        "TEMP_F": 72.0,
        "WIND_MPH": 8.0,
        "WEATHER_MAIN": "Clear",
        "PRECIP_FLAG": 0,
    }
    rows: list[dict[str, Any]] = (
        [{**defaults, **o} for o in overrides] if overrides is not None else [defaults]
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation / metrics
# ---------------------------------------------------------------------------


def make_eval_df(
    overrides: list[dict[str, Any]] | None = None,
    *,
    n: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal evaluation DataFrame (predictions with actuals).

    Args:
        overrides: Per-row field overrides.
        n: Number of rows when *overrides* is not provided.
        seed: RNG seed for default win probabilities and outcomes.

    Returns:
        DataFrame matching the evaluation metrics schema.
    """
    rng: Generator = np.random.default_rng(seed)
    defaults: list[dict[str, Any]] = [
        {
            "game_id": f"2024_01_AWAY_HOME_{i}",
            "season": "2024-2025",
            "week": 1,
            "away_team": "NYJ",
            "home_team": "MIA",
            "away_win_prob": round(rng.uniform(0.3, 0.7), 3),
            "away_team_won": int(rng.random() > 0.5),
            "model_version": "test_model",
        }
        for i in range(n)
    ]

    if overrides is not None:
        rows: list[dict[str, Any]] = defaults[: len(overrides)]
        for i, override in enumerate(overrides):
            if i < len(rows):
                rows[i].update(override)
            else:
                rows.append({**defaults[0], **override, "game_id": f"2024_01_AWAY_HOME_{i}"})
    else:
        rows = defaults

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Predictions (archive schema)
# ---------------------------------------------------------------------------


def make_predictions(
    *,
    n: int = 3,
    season: str = "2025-2026",
    week: int = 1,
    game_id_prefix: str = "2025",
) -> pd.DataFrame:
    """Build a minimal predictions DataFrame matching build_predictions_df output.

    Args:
        n: Number of predictions.
        season: Season identifier.
        week: Week number.
        game_id_prefix: Prefix for GAME_ID construction.

    Returns:
        DataFrame matching the predictions schema.
    """
    return pd.DataFrame(
        {
            "GAME_ID": [f"{game_id_prefix}_0{i}_KC_LAC" for i in range(1, n + 1)],
            "GAME_DATE": ["2025-09-05"] * n,
            "AWAY_TEAM": ["Kansas City Chiefs"] * n,
            "HOME_TEAM": ["Los Angeles Chargers"] * n,
            "AWAY_TEAM_ELO": [1520.0] * n,
            "HOME_TEAM_ELO": [1480.0] * n,
            "AWAY_WIN_PROB": [0.55] * n,
            "HOME_WIN_PROB": [0.45] * n,
        }
    )


# ---------------------------------------------------------------------------
# Mock DatasetAccessor
# ---------------------------------------------------------------------------


def make_accessor(
    *,
    games: pd.DataFrame | None = None,
    stadiums: pd.DataFrame | None = None,
    elo_state: pd.DataFrame | None = None,
    epa_by_game: pd.DataFrame | None = None,
    weather_enriched: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock DatasetAccessor with the specified DataFrames.

    Any dataset not provided returns an empty DataFrame.

    Args:
        games: Games DataFrame.
        stadiums: Stadiums DataFrame.
        elo_state: Elo state DataFrame.
        epa_by_game: EPA by game DataFrame.
        weather_enriched: Weather enriched DataFrame.

    Returns:
        MagicMock implementing the DatasetAccessor interface.
    """
    acc = MagicMock()
    acc.games.return_value = games if games is not None else pd.DataFrame()
    acc.stadiums.return_value = stadiums if stadiums is not None else pd.DataFrame()
    acc.elo_state.return_value = elo_state if elo_state is not None else pd.DataFrame()
    acc.epa_by_game.return_value = epa_by_game if epa_by_game is not None else pd.DataFrame()
    acc.weather_enriched.return_value = (
        weather_enriched if weather_enriched is not None else pd.DataFrame()
    )
    return acc


# ---------------------------------------------------------------------------
# Modeling and props training data
# ---------------------------------------------------------------------------
# These builders produce synthetic data sized for full fit-load-predict
# integration tests. They mirror the schemas of load_modeling_file() and
# build_prop_features() so that downstream training paths receive
# realistically-shaped inputs.


# Game-side feature columns used by the expanded feature set. Order
# matters for the trainer's NaN-detection logic.
def _build_games_modeling_cols() -> list[str]:
    """Build the games modeling column list from the production EPA schema.

    Sources the EPA suffix list from ``features.team.epa.EPA_COLS`` so the
    fixture stays in sync with production. Any new EPA stat added there
    automatically becomes part of the synthetic schema.
    """
    from gridiron_edge.features.team.epa import EPA_COLS

    base_cols: list[str] = [
        "GAME_ID",
        "TEAM_A",
        "TEAM_B",
        "YEAR",
        "WEEK_NUM",
        "RESULT",
        "HOME_FIELD",
        "TEAM_A_ELO",
        "TEAM_B_ELO",
    ]

    # EPA columns: TEAM_A_<UPPER_NAME>, TEAM_B_<UPPER_NAME> for every EPA stat.
    epa_cols: list[str] = []
    for epa_name in EPA_COLS:
        upper = epa_name.upper()
        epa_cols.append(f"TEAM_A_{upper}")
    for epa_name in EPA_COLS:
        upper = epa_name.upper()
        epa_cols.append(f"TEAM_B_{upper}")

    tail_cols: list[str] = ["PTS_WINNER", "PTS_LOSER"]
    return base_cols + epa_cols + tail_cols


_GAMES_MODELING_COLS: list[str] = _build_games_modeling_cols()


def make_games_modeling_df(
    *,
    seasons: tuple[int, ...] = (2006, 2007, 2008, 2009, 2010, 2023, 2024),
    games_per_season: int = 30,
    teams: tuple[str, ...] = ("KC", "SF", "BUF", "PHI", "DAL", "NYG", "MIA", "LAR"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic modeling DataFrame matching the games schema.

    Sized to produce non-empty train and holdout splits when
    ``HOLDOUT_SEASONS`` filters apply. Default span covers both training
    (2006-2010) and holdout (2023-2024) seasons, giving the trainer
    enough rows to run a small HP search without crashing.

    Args:
        seasons: Season years to generate. Use the "YYYY" format —
            converted to "YYYY-YYYY" for the YEAR column.
        games_per_season: Games per season. Each game produces TWO rows
            (TEAM_A and TEAM_B perspectives) to match the real schema.
        teams: Team abbreviations used as TEAM_A / TEAM_B values.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame conforming to the games modeling schema with EPA
        features filled in.
    """
    rng: Generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for season_int in seasons:
        season_str: str = f"{season_int}-{season_int + 1}"
        for game_idx in range(games_per_season):
            week: int = (game_idx % 17) + 1
            team_a: str = teams[game_idx % len(teams)]
            team_b: str = teams[(game_idx + 1) % len(teams)]
            game_id: str = f"{season_int}_{week:02d}_{team_a}_{team_b}"

            # Synthetic game outcome
            home_won: int = int(rng.random() > 0.45)  # mild home-field bias
            pts_winner: int = int(rng.uniform(17, 38))
            pts_loser: int = int(rng.uniform(7, pts_winner))

            # Generate two rows: TEAM_A=team_a (home), then swap
            for home_field in (1, 0):
                if home_field == 1:
                    ta, tb = team_a, team_b
                    result: int = home_won
                else:
                    ta, tb = team_b, team_a
                    result = 1 - home_won

                row: dict[str, Any] = {
                    "GAME_ID": game_id,
                    "TEAM_A": ta,
                    "TEAM_B": tb,
                    "YEAR": season_str,
                    "WEEK_NUM": week,
                    "RESULT": result,
                    "HOME_FIELD": home_field,
                    "TEAM_A_ELO": rng.uniform(1400, 1600),
                    "TEAM_B_ELO": rng.uniform(1400, 1600),
                    "PTS_WINNER": pts_winner,
                    "PTS_LOSER": pts_loser,
                }
                # Fill all EPA columns with realistic ranges
                for col in _GAMES_MODELING_COLS:
                    if col not in row:
                        if "EPA" in col:
                            row[col] = rng.uniform(-0.3, 0.4)
                        elif "RATE" in col or "PCT" in col:
                            row[col] = rng.uniform(0.2, 0.7)
                        else:
                            row[col] = rng.uniform(-0.2, 0.3)
                rows.append(row)

    return pd.DataFrame(rows)[_GAMES_MODELING_COLS]


def make_games_from_modeling_df(
    modeling_df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a games DataFrame matching the GAME_IDs in a modeling DataFrame.

    Use this when a test needs both a modeling file and a corresponding
    games file (e.g. for the total trainer, which joins them by GAME_ID).
    The games DataFrame is sized to match exactly the unique GAME_IDs in
    the modeling file.

    Args:
        modeling_df: Modeling DataFrame from :func:`make_games_modeling_df`.
        seed: RNG seed for outcome generation.

    Returns:
        Games DataFrame with one row per unique GAME_ID, populated with
        WINNER, LOSER, WIN_OR_TIE, GAME_LOCATION, PTS_WINNER, PTS_LOSER,
        and the YEAR/WEEK_NUM/GAME_DATE fields needed for the join.
    """
    rng: Generator = np.random.default_rng(seed)

    # One row per game (away-team perspective in modeling)
    away_rows = modeling_df.loc[modeling_df["HOME_FIELD"] == 0].drop_duplicates(subset=["GAME_ID"])

    rows: list[dict[str, Any]] = []
    for _, row in away_rows.iterrows():
        # In modeling: TEAM_A=away, TEAM_B=home
        # In games: WINNER/LOSER decided by RESULT; LOCATION marks side
        away_team: str = row["TEAM_A"]
        home_team: str = row["TEAM_B"]
        away_won: int = int(row["RESULT"])

        winner: str = away_team if away_won else home_team
        loser: str = home_team if away_won else away_team
        location: str = "@" if away_won else "NULL_VALUE"

        pts_winner: int = int(rng.uniform(20, 38))
        pts_loser: int = int(rng.uniform(7, pts_winner))

        rows.append(
            {
                "GAME_ID": row["GAME_ID"],
                "YEAR": row["YEAR"],
                "WEEK_NUM": row["WEEK_NUM"],
                "WINNER": winner,
                "LOSER": loser,
                "WIN_OR_TIE": 1,
                "GAME_LOCATION": location,
                "GAME_DATE": f"2024-{row['WEEK_NUM']:02d}-01",
                "STADIUM": f"{home_team} Stadium",
                "ROOF": "outdoors",
                "SURFACE": "grass",
                "GAMETIME": "13:00",
                "GAME_DAY_OF_WEEK": "Sunday",
                "PTS_WINNER": pts_winner,
                "PTS_LOSER": pts_loser,
                "VEGAS_LINE": float(rng.uniform(-10, 10)),
                "OVER_UNDER": float(rng.uniform(40, 55)),
                "FAVORITED": home_team,
            }
        )

    return pd.DataFrame(rows)


# Prop-side feature columns used by build_prop_features. Mirrors the
# columns produced by the real builder.
_PROPS_MODELING_COLS: list[str] = [
    "player_id",
    "player_name",
    "position",
    "team",
    "team_abbr",
    "opponent_team",
    "is_skill",
    "season",
    "week",
    "game_id",
    # Volume / target column candidates
    "attempts",
    "carries",
    "targets",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    # Rolling feature stand-ins (simulated)
    "passing_yards_L3_mean",
    "passing_yards_L3_std",
    "rushing_yards_L3_mean",
    "rushing_yards_L3_std",
    "receiving_yards_L3_mean",
    "receiving_yards_L3_std",
    # Game context
    "implied_team_total",
    "spread_line",
    "OVER_UNDER",
    "is_home",
    "roof_dome",
    "surface_turf",
    "TEMP_F",
    "WIND_SPEED_MPH",
    "rest_days",
    "opp_rest_days",
    "rest_diff",
    "DIV_GAME",
]


def make_props_modeling_df(
    *,
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024),
    players_per_position: int = 4,
    games_per_season: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic player-game DataFrame matching the prop feature schema.

    Generates rows for QB, RB, WR, TE positions across multiple seasons
    so PropTrainer's holdout split (HOLDOUT_SEASONS = 2023+) produces
    non-empty train and holdout subsets.

    Args:
        seasons: Season years.
        players_per_position: How many distinct players per position.
        games_per_season: Games per player per season.
        seed: RNG seed.

    Returns:
        DataFrame with player_id, position, target stats, rolling
        features, and game context.
    """
    rng: Generator = np.random.default_rng(seed)
    positions: list[str] = ["QB", "RB", "WR", "TE"]
    rows: list[dict[str, Any]] = []

    for season_int in seasons:
        for position in positions:
            for player_idx in range(players_per_position):
                player_id: str = f"{position}_{player_idx}"
                player_name: str = f"{position} Player {player_idx}"
                for week in range(1, games_per_season + 1):
                    # Position-aware target generation
                    if position == "QB":
                        passing_yards: float = float(rng.uniform(180, 320))
                        rushing_yards: float = float(rng.uniform(0, 40))
                        receiving_yards: float = float("nan")
                        attempts: int = int(rng.uniform(25, 45))
                        carries: int = int(rng.uniform(2, 8))
                        targets: int = 0
                    elif position == "RB":
                        passing_yards = float("nan")
                        rushing_yards = float(rng.uniform(30, 130))
                        receiving_yards = float(rng.uniform(0, 40))
                        attempts = 0
                        carries = int(rng.uniform(8, 25))
                        targets = int(rng.uniform(1, 6))
                    else:  # WR or TE
                        passing_yards = float("nan")
                        rushing_yards = float("nan")
                        receiving_yards = float(
                            rng.uniform(20, 110) if position == "WR" else rng.uniform(15, 80)
                        )
                        attempts = 0
                        carries = 0
                        targets = int(rng.uniform(3, 12))

                    row: dict[str, Any] = {
                        "player_id": player_id,
                        "player_name": player_name,
                        "position": position,
                        "team": "KC",
                        "team_abbr": "KC",
                        "opponent_team": "LAR",
                        "is_skill": position in ("QB", "RB", "WR", "TE"),
                        "season": season_int,
                        "week": week,
                        "game_id": f"{season_int}_{week:02d}_KC_LAR",
                        "attempts": attempts,
                        "carries": carries,
                        "targets": targets,
                        "passing_yards": passing_yards,
                        "rushing_yards": rushing_yards,
                        "receiving_yards": receiving_yards,
                        "passing_yards_L3_mean": (
                            float(rng.uniform(200, 280)) if position == "QB" else float("nan")
                        ),
                        "passing_yards_L3_std": (
                            float(rng.uniform(30, 60)) if position == "QB" else float("nan")
                        ),
                        "rushing_yards_L3_mean": (
                            float(rng.uniform(60, 100)) if position == "RB" else float("nan")
                        ),
                        "rushing_yards_L3_std": (
                            float(rng.uniform(15, 35)) if position == "RB" else float("nan")
                        ),
                        "receiving_yards_L3_mean": (
                            float(rng.uniform(40, 80)) if position in ("WR", "TE") else float("nan")
                        ),
                        "receiving_yards_L3_std": (
                            float(rng.uniform(15, 30)) if position in ("WR", "TE") else float("nan")
                        ),
                        "implied_team_total": float(rng.uniform(20, 30)),
                        "spread_line": float(rng.uniform(-7, 7)),
                        "OVER_UNDER": float(rng.uniform(42, 52)),
                        "is_home": int(rng.random() > 0.5),
                        "roof_dome": int(rng.random() > 0.7),
                        "surface_turf": int(rng.random() > 0.5),
                        "TEMP_F": float(rng.uniform(40, 80)),
                        "WIND_SPEED_MPH": float(rng.uniform(0, 15)),
                        "rest_days": int(rng.uniform(6, 10)),
                        "opp_rest_days": int(rng.uniform(6, 10)),
                        "rest_diff": int(rng.uniform(-3, 3)),
                        "DIV_GAME": int(rng.random() > 0.7),
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


def make_modeling_manifest(
    *,
    schema_version: int = 4,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal modeling_file_manifest.json structure.

    The manifest is required by ``load_modeling_file`` when called with
    ``required_schema_version=...`` (which is the case for the predict
    path in ``GamesPredictor``). Without this, predict-side integration
    tests fail with ``FileNotFoundError: No feature manifest found``.

    Args:
        schema_version: Schema version the manifest declares. Should
            match ``CURRENT_SCHEMA_VERSION`` from the feature manifest
            module for the tests to pass validation.
        columns: Column list to record in the manifest. If None, an
            empty list is recorded — the load path uses the manifest
            primarily for version validation, not column validation.

    Returns:
        Dict ready to be written via ``json.dump``.
    """
    return {
        "schema_version": schema_version,
        "columns": columns if columns is not None else [],
    }
