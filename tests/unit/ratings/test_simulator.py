"""Tests for ratings.elo.simulator - canonical Elo history simulator."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.ratings.elo.simulator import (
    EloSimulationResult,
    simulate_elo_history,
    transition_to_next_season,
)


def _make_games() -> pd.DataFrame:
    """Tiny canonical history with one game in each season."""
    return pd.DataFrame(
        {
            "YEAR": ["2023-2024", "2024-2025"],
            "WEEK_NUM": [1, 1],
            "AWAY_TEAM": ["KC", "BUF"],
            "HOME_TEAM": ["LAC", "MIA"],
            "AWAY_SCORE": [27, 17],
            "HOME_SCORE": [20, 24],
            "GAME_ID": [
                "2023_01_KC_LAC",
                "2024_01_BUF_MIA",
            ],
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

    # In the first game, Away team KC beat Home team LAC.
    kc_week2 = result.elo[("KC", "2023-2024", 2)]
    lac_week2 = result.elo[("LAC", "2023-2024", 2)]

    assert kc_week2 > 1500.0
    assert lac_week2 < 1500.0
    assert result.away_outcomes == [1.0]


def test_home_winner_gains_and_away_loser_loses() -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2023-2024"],
            "WEEK_NUM": [1],
            "AWAY_TEAM": ["KC"],
            "HOME_TEAM": ["LAC"],
            "AWAY_SCORE": [17],
            "HOME_SCORE": [24],
            "GAME_ID": ["2023_01_KC_LAC"],
        }
    )

    result = simulate_elo_history(
        games,
        ["2023-2024"],
        {"2023-2024": {"KC", "LAC"}},
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    assert result.away_outcomes == [0.0]
    assert result.elo[("KC", "2023-2024", 2)] < 1500.0
    assert result.elo[("LAC", "2023-2024", 2)] > 1500.0


def test_tie_records_half_outcome_and_preserves_ratings() -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2023-2024"],
            "WEEK_NUM": [1],
            "AWAY_TEAM": ["KC"],
            "HOME_TEAM": ["LAC"],
            "AWAY_SCORE": [21],
            "HOME_SCORE": [21],
            "GAME_ID": ["2023_01_KC_LAC"],
        }
    )

    result = simulate_elo_history(
        games,
        ["2023-2024"],
        {"2023-2024": {"KC", "LAC"}},
        expansion_start={},
        k_early=20.0,
        k_mid=20.0,
        k_week18=20.0,
        k_post=20.0,
        divisor=480.0,
        regress_frac=0.0,
    )

    assert result.away_outcomes == [0.5]
    assert result.elo[("KC", "2023-2024", 2)] == pytest.approx(1500.0)
    assert result.elo[("LAC", "2023-2024", 2)] == pytest.approx(1500.0)


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


class TestNextSeasonTransition:
    """Verify deterministic offseason transition behavior."""

    def test_returning_teams_regress_toward_mean(self) -> None:
        transitioned = transition_to_next_season(
            {
                "Kansas City Chiefs": 1600.0,
                "Los Angeles Chargers": 1400.0,
            },
            returning_teams={
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            },
            expansion_start={},
            next_year="2026-2027",
            regress_frac=1 / 3.0,
        )

        assert transitioned["Kansas City Chiefs"] == pytest.approx(1566.6666666667)
        assert transitioned["Los Angeles Chargers"] == pytest.approx(1433.3333333333)

    def test_transition_is_reproducible(self) -> None:
        final_ratings = {
            "Kansas City Chiefs": 1600.0,
            "Los Angeles Chargers": 1400.0,
        }
        returning_teams = {
            "Kansas City Chiefs",
            "Los Angeles Chargers",
        }

        first = transition_to_next_season(
            final_ratings,
            returning_teams=returning_teams,
            expansion_start={},
            next_year="2026-2027",
            regress_frac=1 / 3.0,
        )
        second = transition_to_next_season(
            final_ratings,
            returning_teams=returning_teams,
            expansion_start={},
            next_year="2026-2027",
            regress_frac=1 / 3.0,
        )

        assert first == second

    def test_expansion_team_uses_expansion_rating(
        self,
    ) -> None:
        transitioned = transition_to_next_season(
            {
                "Existing Team": 1520.0,
            },
            returning_teams={
                "Existing Team",
            },
            expansion_start={
                "Expansion Team": "2026-2027",
            },
            next_year="2026-2027",
            regress_frac=1 / 3.0,
            expansion_elo=1300.0,
        )

        assert transitioned["Expansion Team"] == 1300.0

    def test_expansion_from_other_season_is_not_added(
        self,
    ) -> None:
        transitioned = transition_to_next_season(
            {},
            returning_teams=set(),
            expansion_start={
                "Expansion Team": "2027-2028",
            },
            next_year="2026-2027",
            regress_frac=1 / 3.0,
        )

        assert "Expansion Team" not in transitioned

    def test_does_not_mutate_final_ratings(self) -> None:
        final_ratings = {
            "Kansas City Chiefs": 1600.0,
            "Los Angeles Chargers": 1400.0,
        }
        original = final_ratings.copy()

        transition_to_next_season(
            final_ratings,
            returning_teams=set(final_ratings),
            expansion_start={},
            next_year="2026-2027",
            regress_frac=1 / 3.0,
        )

        assert final_ratings == original
