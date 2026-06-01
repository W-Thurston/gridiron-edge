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
                        "def_explosive_rate": rng.uniform(0.03, 0.15),
                        "def_epa_per_play": rng.uniform(-0.3, 0.2),
                        "def_pass_epa": rng.uniform(-0.4, 0.3),
                        "def_rush_epa": rng.uniform(-0.2, 0.2),
                        "def_success_rate": rng.uniform(0.3, 0.6),
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
