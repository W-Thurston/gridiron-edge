"""Tests for ratings.elo.simulator - canonical Elo history simulator."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.core.constants import AWAY_WIN_LOCATION
from gridiron_edge.ratings.elo.simulator import (
    EloSimulationResult,
    simulate_elo_history,
)


def _make_games() -> pd.DataFrame:
    """Tiny fake history with one game per season for two seasons."""
    return pd.DataFrame(
        {
            "YEAR": ["2023-2024", "2024-2025"],
            "WEEK_NUM": [1, 1],
            "WINNER": ["KC", "BUF"],
            "LOSER": ["LAC", "MIA"],
            "WIN_OR_TIE": [1.0, 1.0],
            "GAME_LOCATION": [AWAY_WIN_LOCATION, "H"],
            "GAME_ID": ["2023_01_KC_LAC", "2024_01_BUF_MIA"],
        }
    )


def test_returns_populated_result() -> None:
    games = _make_games()
    sorted_years = ["2023-2024", "2024-2025"]
    teams_by_year = {
        "2023-2024": {"KC", "LAC", "BUF", "MIA"},
        "2024-2025": {"KC", "LAC", "BUF", "MIA"},
    }
    expansion_start: dict[str, str] = {}

    result = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start,
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    assert isinstance(result, EloSimulationResult)
    assert len(result.away_probs) == 2
    assert len(result.away_outcomes) == 2
    assert len(result.game_seasons) == 2
    assert result.game_ids == ["2023_01_KC_LAC", "2024_01_BUF_MIA"]


def test_state_dict_keys_present_for_all_teams() -> None:
    games = _make_games()
    sorted_years = ["2023-2024"]
    teams_by_year = {"2023-2024": {"KC", "LAC", "BUF", "MIA"}}

    result = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    for team in teams_by_year["2023-2024"]:
        assert (team, "2023-2024", 1) in result.elo


def test_winner_gains_loser_loses() -> None:
    games = _make_games()
    sorted_years = ["2023-2024"]
    teams_by_year = {"2023-2024": {"KC", "LAC", "BUF", "MIA"}}

    result = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    # In the first game KC beat LAC (away win since GAME_LOCATION == "@").
    kc_week2 = result.elo[("KC", "2023-2024", 2)]
    lac_week2 = result.elo[("LAC", "2023-2024", 2)]

    assert kc_week2 > 1500.0
    assert lac_week2 < 1500.0


def test_zero_sum_invariant_within_a_game() -> None:
    games = _make_games().iloc[[0]].copy()
    sorted_years = ["2023-2024"]
    teams_by_year = {"2023-2024": {"KC", "LAC"}}

    result = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    kc_w2 = result.elo[("KC", "2023-2024", 2)]
    lac_w2 = result.elo[("LAC", "2023-2024", 2)]
    assert kc_w2 + lac_w2 == pytest.approx(3000.0, abs=1e-9)


def test_table_and_tuner_share_elo_state() -> None:
    """Both consumers see the same Elo dict for the same inputs."""
    games = _make_games()
    sorted_years = ["2023-2024", "2024-2025"]
    teams_by_year = {
        "2023-2024": {"KC", "LAC", "BUF", "MIA"},
        "2024-2025": {"KC", "LAC", "BUF", "MIA"},
    }

    first = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )
    second = simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    assert first.elo == second.elo
    assert first.away_probs == second.away_probs
