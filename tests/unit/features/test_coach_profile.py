# tests/unit/features/test_coach_profile.py

"""Tests for the Away/Home head-coach profile feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.coach_profile import (
    HomeAwayCoachProfileFeature,
    _build_coach_result_history,
    _hc_win_pct_entering,
    _known_team_coach_seasons,
    _tenure_entering,
)


def _datasets(*, sidecar: DataFrame, games: DataFrame | None = None) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.qb_coach_sidecar.return_value = sidecar.copy()
    datasets.games.return_value = (games if games is not None else DataFrame()).copy()
    return datasets


def _sidecar_row(
    *,
    game_id: str,
    year: str,
    week: int,
    away_team: str,
    home_team: str,
    away_coach: object,
    home_coach: object,
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "AWAY_TEAM": away_team,
        "HOME_TEAM": home_team,
        "AWAY_QB_ID": "QB1",
        "AWAY_QB_NAME": "QB1 Name",
        "HOME_QB_ID": "QB2",
        "HOME_QB_NAME": "QB2 Name",
        "AWAY_COACH": away_coach,
        "HOME_COACH": home_coach,
    }


def _completed_game(
    *,
    game_id: str,
    year: str,
    week: int,
    away_score: int,
    home_score: int,
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
    }


def _empty_games() -> DataFrame:
    """Properly shaped, zero-row games frame - matches the real loader's shape."""
    return DataFrame(columns=["GAME_ID", "YEAR", "WEEK_NUM", "AWAY_SCORE", "HOME_SCORE"])


def _game_frame(
    *,
    game_id: str = "target-week-4",
    year: str = "2025-2026",
    week: int = 4,
    away_team: str = "Away Team",
    home_team: str = "Home Team",
) -> DataFrame:
    return DataFrame(
        {
            "GAME_ID": [game_id],
            "YEAR": [year],
            "WEEK_NUM": [week],
            "AWAY_TEAM": [away_team],
            "HOME_TEAM": [home_team],
            "MARKER": ["preserved"],
        }
    )


class TestBuildCoachResultHistory:
    def test_wins_and_losses_are_recorded_per_coach(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                )
            ]
        )
        games = DataFrame(
            [_completed_game(game_id="g1", year="2025-2026", week=1, away_score=20, home_score=10)]
        )

        history = _build_coach_result_history(sidecar, games)

        away_row = history.loc[history["COACH"] == "Coach X"].iloc[0]
        home_row = history.loc[history["COACH"] == "Coach Y"].iloc[0]
        assert away_row["RESULT"] == pytest.approx(1.0)
        assert home_row["RESULT"] == pytest.approx(0.0)

    def test_tie_is_half_credit(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                )
            ]
        )
        games = DataFrame(
            [_completed_game(game_id="g1", year="2025-2026", week=1, away_score=20, home_score=20)]
        )

        history = _build_coach_result_history(sidecar, games)

        assert (history["RESULT"] == 0.5).all()

    def test_unplayed_games_excluded(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                )
            ]
        )

        history = _build_coach_result_history(sidecar, _empty_games())

        assert history.empty


class TestHcWinPctEntering:
    def test_no_coach_is_nan(self) -> None:
        assert pd.isna(_hc_win_pct_entering(DataFrame(), coach=None, season=2025, week=3))

    def test_no_prior_games_is_nan(self) -> None:
        history = DataFrame(
            {
                "COACH": ["Coach X"],
                "SEASON": [2025],
                "WEEK_NUM": [3],
                "GAME_ID": ["g1"],
                "RESULT": [1.0],
            }
        )
        assert pd.isna(_hc_win_pct_entering(history, coach="Coach X", season=2025, week=1))

    def test_career_win_pct_spans_seasons(self) -> None:
        history = DataFrame(
            {
                "COACH": ["Coach X", "Coach X", "Coach X"],
                "SEASON": [2023, 2024, 2025],
                "WEEK_NUM": [1, 1, 1],
                "GAME_ID": ["g1", "g2", "g3"],
                "RESULT": [1.0, 0.0, 1.0],
            }
        )
        result = _hc_win_pct_entering(history, coach="Coach X", season=2025, week=2)
        assert result == pytest.approx(2.0 / 3.0)

    def test_no_lookahead_to_target_week(self) -> None:
        history = DataFrame(
            {
                "COACH": ["Coach X"],
                "SEASON": [2025],
                "WEEK_NUM": [3],
                "GAME_ID": ["g1"],
                "RESULT": [1.0],
            }
        )
        assert pd.isna(_hc_win_pct_entering(history, coach="Coach X", season=2025, week=3))


class TestTenureEntering:
    def test_no_coach_is_nan(self) -> None:
        assert pd.isna(_tenure_entering(set(), team="A", coach=None, season=2025))

    def test_first_year_is_tenure_one(self) -> None:
        triples = {("A", "Coach X", 2025)}
        assert _tenure_entering(triples, team="A", coach="Coach X", season=2025) == 1

    def test_consecutive_seasons_accumulate(self) -> None:
        triples = {("A", "Coach X", 2023), ("A", "Coach X", 2024), ("A", "Coach X", 2025)}
        assert _tenure_entering(triples, team="A", coach="Coach X", season=2025) == 3

    def test_gap_season_stops_the_count(self) -> None:
        triples = {("A", "Coach X", 2021), ("A", "Coach X", 2024), ("A", "Coach X", 2025)}
        assert _tenure_entering(triples, team="A", coach="Coach X", season=2025) == 2

    def test_new_team_resets_tenure(self) -> None:
        triples = {("B", "Coach X", 2023), ("B", "Coach X", 2024), ("A", "Coach X", 2025)}
        assert _tenure_entering(triples, team="A", coach="Coach X", season=2025) == 1


class TestKnownTeamCoachSeasons:
    def test_stacks_away_and_home(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                )
            ]
        )
        triples = _known_team_coach_seasons(sidecar)
        assert ("A", "Coach X", 2025) in triples
        assert ("B", "Coach Y", 2025) in triples


class TestHomeAwayCoachProfileFeature:
    def test_registered_as_home_away_coach_profile(self) -> None:
        assert FeatureRegistry.get("home_away_coach_profile") is HomeAwayCoachProfileFeature

    def test_spec_produces_all_six_columns(self) -> None:
        assert set(HomeAwayCoachProfileFeature.spec.produces) == {
            "AWAY_HC_WIN_PCT_CAREER",
            "HOME_HC_WIN_PCT_CAREER",
            "AWAY_HC_TENURE",
            "HOME_HC_TENURE",
            "AWAY_NEW_COACHING_STAFF_FLAG",
            "HOME_NEW_COACHING_STAFF_FLAG",
        }

    def test_empty_sidecar_produces_null_defaults(self) -> None:
        result = HomeAwayCoachProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=DataFrame())
        )
        row = result.iloc[0]
        assert pd.isna(row["AWAY_HC_WIN_PCT_CAREER"])
        assert row["AWAY_NEW_COACHING_STAFF_FLAG"] == 0
        assert row["MARKER"] == "preserved"

    def test_full_pipeline_computes_win_pct_and_tenure(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2024-2025",
                    week=1,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                ),
                _sidecar_row(
                    game_id="g2",
                    year="2025-2026",
                    week=1,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                ),
                _sidecar_row(
                    game_id="target-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_coach="Coach X",
                    home_coach="Coach Z",
                ),
            ]
        )
        games = DataFrame(
            [
                _completed_game(
                    game_id="g1", year="2024-2025", week=1, away_score=30, home_score=10
                ),
                _completed_game(
                    game_id="g2", year="2025-2026", week=1, away_score=10, home_score=30
                ),
            ]
        )

        result = HomeAwayCoachProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=sidecar, games=games)
        )
        row = result.iloc[0]

        assert row["AWAY_HC_WIN_PCT_CAREER"] == pytest.approx(0.5)
        assert row["AWAY_HC_TENURE"] == 2
        assert row["AWAY_NEW_COACHING_STAFF_FLAG"] == 0
        assert pd.isna(row["HOME_HC_WIN_PCT_CAREER"])
        assert row["HOME_HC_TENURE"] == 1
        assert row["HOME_NEW_COACHING_STAFF_FLAG"] == 1
        assert row["MARKER"] == "preserved"

    def test_preserves_row_order(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                ),
                _sidecar_row(
                    game_id="g-week-5",
                    year="2025-2026",
                    week=5,
                    away_team="A",
                    home_team="B",
                    away_coach="Coach X",
                    home_coach="Coach Y",
                ),
            ]
        )
        games = DataFrame(
            {
                "GAME_ID": ["g-week-5", "g-week-4"],
                "YEAR": ["2025-2026", "2025-2026"],
                "WEEK_NUM": [5, 4],
                "AWAY_TEAM": ["A", "A"],
                "HOME_TEAM": ["B", "B"],
            }
        )

        result = HomeAwayCoachProfileFeature().compute(
            df=games, datasets=_datasets(sidecar=sidecar, games=_empty_games())
        )
        assert result["GAME_ID"].tolist() == ["g-week-5", "g-week-4"]
