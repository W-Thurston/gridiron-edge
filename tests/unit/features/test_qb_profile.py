# tests/unit/features/test_qb_profile.py

"""Tests for the Away/Home starting-QB profile feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.qb_profile import (
    HomeAwayQbProfileFeature,
    _build_target_qb_profile,
    _qb_game_log_history,
    _qb_profile_entering,
    resolve_starting_qb_history,
)


def _datasets(*, sidecar: DataFrame, logs: DataFrame | None = None) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.qb_coach_sidecar.return_value = sidecar.copy()
    datasets.player_game_logs.return_value = (logs if logs is not None else DataFrame()).copy()
    return datasets


def _sidecar_row(
    *,
    game_id: str,
    year: str,
    week: int,
    away_team: str,
    home_team: str,
    away_qb_id: object,
    home_qb_id: object,
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "AWAY_TEAM": away_team,
        "HOME_TEAM": home_team,
        "AWAY_QB_ID": away_qb_id,
        "AWAY_QB_NAME": f"{away_qb_id} Name" if away_qb_id else None,
        "HOME_QB_ID": home_qb_id,
        "HOME_QB_NAME": f"{home_qb_id} Name" if home_qb_id else None,
        "AWAY_COACH": "Coach Away",
        "HOME_COACH": "Coach Home",
    }


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


def _qb_log(
    *,
    player_id: str,
    season: int,
    week: int,
    attempts: int,
    passing_epa: float,
    passing_cpoe: float | None = None,
    position: str = "QB",
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "position": position,
        "season": season,
        "week": week,
        "attempts": attempts,
        "passing_epa": passing_epa,
        "passing_cpoe": passing_cpoe,
    }


class TestResolveStartingQbHistory:
    def test_actual_starter_is_preserved(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                )
            ]
        )

        history = resolve_starting_qb_history(sidecar)

        away_row = history.loc[(history["TEAM"] == "A") & (history["SIDE"] == "AWAY")].iloc[0]
        assert away_row["RESOLVED_QB_ID"] == "QB1"

    def test_unplayed_game_carries_forward_last_known_starter(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="g2",
                    year="2025-2026",
                    week=2,
                    away_team="A",
                    home_team="C",
                    away_qb_id=None,
                    home_qb_id="QB3",
                ),
            ]
        )

        history = resolve_starting_qb_history(sidecar)

        week2_away = history.loc[
            (history["TEAM"] == "A") & (history["SIDE"] == "AWAY") & (history["WEEK_NUM"] == 2)
        ].iloc[0]
        assert week2_away["RESOLVED_QB_ID"] == "QB1"

    def test_qb_change_flag_set_when_starter_differs(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="g2",
                    year="2025-2026",
                    week=2,
                    away_team="A",
                    home_team="C",
                    away_qb_id="QB9",
                    home_qb_id="QB2",
                ),
            ]
        )

        history = resolve_starting_qb_history(sidecar)

        week2_away = history.loc[
            (history["TEAM"] == "A") & (history["SIDE"] == "AWAY") & (history["WEEK_NUM"] == 2)
        ].iloc[0]
        assert week2_away["QB_CHANGE_FLAG"] == 1

    def test_qb_change_flag_zero_with_no_prior_game(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                )
            ]
        )

        history = resolve_starting_qb_history(sidecar)

        row = history.loc[(history["TEAM"] == "A") & (history["SIDE"] == "AWAY")].iloc[0]
        assert row["QB_CHANGE_FLAG"] == 0

    def test_qb_change_flag_zero_when_same_starter(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="g2",
                    year="2025-2026",
                    week=2,
                    away_team="A",
                    home_team="C",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
            ]
        )

        history = resolve_starting_qb_history(sidecar)

        week2_away = history.loc[
            (history["TEAM"] == "A") & (history["SIDE"] == "AWAY") & (history["WEEK_NUM"] == 2)
        ].iloc[0]
        assert week2_away["QB_CHANGE_FLAG"] == 0


class TestQbGameLogHistory:
    def test_filters_to_qb_pass_attempts_only(self) -> None:
        logs = DataFrame(
            [
                _qb_log(player_id="QB1", season=2025, week=1, attempts=30, passing_epa=5.0),
                _qb_log(
                    player_id="RB1", season=2025, week=1, attempts=0, passing_epa=0.0, position="RB"
                ),
                _qb_log(player_id="QB2", season=2025, week=1, attempts=0, passing_epa=0.0),
            ]
        )

        history = _qb_game_log_history(logs)

        assert set(history["player_id"]) == {"QB1"}

    def test_sorted_chronologically_per_player(self) -> None:
        logs = DataFrame(
            [
                _qb_log(player_id="QB1", season=2025, week=3, attempts=20, passing_epa=1.0),
                _qb_log(player_id="QB1", season=2025, week=1, attempts=20, passing_epa=2.0),
                _qb_log(player_id="QB1", season=2024, week=18, attempts=20, passing_epa=3.0),
            ]
        )

        history = _qb_game_log_history(logs)

        assert history["week"].tolist() == [18, 1, 3]
        assert history["season"].tolist() == [2024, 2025, 2025]


class TestQbProfileEntering:
    def test_no_player_id_is_nan_with_zero_experience(self) -> None:
        result = _qb_profile_entering(
            DataFrame(
                columns=["player_id", "season", "week", "attempts", "passing_epa", "passing_cpoe"]
            ),
            player_id=None,
            season=2025,
            week=3,
            window=4,
        )
        assert all(pd.isna(value) for value in result[:3])
        assert result[3] == 0

    def test_no_prior_games_is_nan_with_zero_experience(self) -> None:
        history = _qb_game_log_history(
            DataFrame([_qb_log(player_id="QB1", season=2025, week=3, attempts=20, passing_epa=2.0)])
        )
        result = _qb_profile_entering(history, player_id="QB1", season=2025, week=1, window=4)
        assert all(pd.isna(value) for value in result[:3])
        assert result[3] == 0

    def test_career_epa_per_play_is_weighted_by_attempts(self) -> None:
        history = _qb_game_log_history(
            DataFrame(
                [
                    _qb_log(player_id="QB1", season=2025, week=1, attempts=10, passing_epa=5.0),
                    _qb_log(player_id="QB1", season=2025, week=2, attempts=30, passing_epa=3.0),
                ]
            )
        )
        career_epa, _, _, experience = _qb_profile_entering(
            history, player_id="QB1", season=2025, week=3, window=4
        )
        assert career_epa == pytest.approx((5.0 + 3.0) / (10 + 30))
        assert experience == 2

    def test_rolling_window_uses_only_most_recent_games(self) -> None:
        history = _qb_game_log_history(
            DataFrame(
                [
                    _qb_log(player_id="QB1", season=2025, week=1, attempts=10, passing_epa=-100.0),
                    _qb_log(player_id="QB1", season=2025, week=2, attempts=10, passing_epa=5.0),
                    _qb_log(player_id="QB1", season=2025, week=3, attempts=10, passing_epa=5.0),
                ]
            )
        )
        _, rolling_epa, _, _ = _qb_profile_entering(
            history, player_id="QB1", season=2025, week=4, window=2
        )
        assert rolling_epa == pytest.approx(0.5)

    def test_rolling_cpoe_ignores_null_games(self) -> None:
        history = _qb_game_log_history(
            DataFrame(
                [
                    _qb_log(
                        player_id="QB1",
                        season=2025,
                        week=1,
                        attempts=10,
                        passing_epa=1.0,
                        passing_cpoe=None,
                    ),
                    _qb_log(
                        player_id="QB1",
                        season=2025,
                        week=2,
                        attempts=10,
                        passing_epa=1.0,
                        passing_cpoe=4.0,
                    ),
                ]
            )
        )
        _, _, rolling_cpoe, _ = _qb_profile_entering(
            history, player_id="QB1", season=2025, week=3, window=4
        )
        assert rolling_cpoe == pytest.approx(4.0)

    def test_no_lookahead_to_target_week(self) -> None:
        history = _qb_game_log_history(
            DataFrame(
                [
                    _qb_log(player_id="QB1", season=2025, week=3, attempts=10, passing_epa=999.0),
                ]
            )
        )
        career_epa, _, _, experience = _qb_profile_entering(
            history, player_id="QB1", season=2025, week=3, window=4
        )
        assert pd.isna(career_epa)
        assert experience == 0


class TestBuildTargetQbProfile:
    def test_duplicate_targets_computed_once(self) -> None:
        history = _qb_game_log_history(
            DataFrame([_qb_log(player_id="QB1", season=2025, week=1, attempts=10, passing_epa=1.0)])
        )
        targets = DataFrame(
            [
                {"player_id": "QB1", "season": 2025, "week": 2},
                {"player_id": "QB1", "season": 2025, "week": 2},
            ]
        )
        result = _build_target_qb_profile(history, targets, window=4)
        assert len(result) == 1


class TestHomeAwayQbProfileFeature:
    def test_registered_as_home_away_qb_profile(self) -> None:
        assert FeatureRegistry.get("home_away_qb_profile") is HomeAwayQbProfileFeature

    def test_spec_produces_all_twelve_columns(self) -> None:
        assert set(HomeAwayQbProfileFeature.spec.produces) == {
            "AWAY_QB_EPA_PLAY_CAREER",
            "HOME_QB_EPA_PLAY_CAREER",
            "AWAY_QB_EPA_PLAY_ROLLING",
            "HOME_QB_EPA_PLAY_ROLLING",
            "AWAY_QB_CPOE_ROLLING",
            "HOME_QB_CPOE_ROLLING",
            "AWAY_QB_CPOE_ROLLING_AVAILABLE",
            "HOME_QB_CPOE_ROLLING_AVAILABLE",
            "AWAY_QB_EXPERIENCE",
            "HOME_QB_EXPERIENCE",
            "AWAY_QB_CHANGE_FLAG",
            "HOME_QB_CHANGE_FLAG",
        }

    def test_empty_sidecar_produces_null_defaults(self) -> None:
        result = HomeAwayQbProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=DataFrame())
        )
        row = result.iloc[0]
        assert pd.isna(row["AWAY_QB_EPA_PLAY_CAREER"])
        assert row["AWAY_QB_CPOE_ROLLING_AVAILABLE"] == 0
        assert row["AWAY_QB_EXPERIENCE"] == 0
        assert row["AWAY_QB_CHANGE_FLAG"] == 0
        assert row["MARKER"] == "preserved"

    def test_empty_player_logs_produces_null_metrics_but_keeps_change_flag(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="target-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB9",
                    home_qb_id="QB2",
                ),
            ]
        )
        result = HomeAwayQbProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=sidecar)
        )
        row = result.iloc[0]
        assert pd.isna(row["AWAY_QB_EPA_PLAY_CAREER"])
        assert row["AWAY_QB_CHANGE_FLAG"] == 1
        assert row["HOME_QB_CHANGE_FLAG"] == 0

    def test_full_pipeline_resolves_starter_and_profile(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="g2",
                    year="2025-2026",
                    week=2,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="target-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
            ]
        )
        logs = DataFrame(
            [
                _qb_log(
                    player_id="QB1",
                    season=2025,
                    week=1,
                    attempts=20,
                    passing_epa=4.0,
                    passing_cpoe=2.0,
                ),
                _qb_log(
                    player_id="QB1",
                    season=2025,
                    week=2,
                    attempts=20,
                    passing_epa=6.0,
                    passing_cpoe=6.0,
                ),
            ]
        )

        result = HomeAwayQbProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=sidecar, logs=logs)
        )
        row = result.iloc[0]

        assert row["AWAY_QB_EPA_PLAY_CAREER"] == pytest.approx((4.0 + 6.0) / 40)
        assert row["AWAY_QB_CPOE_ROLLING"] == pytest.approx((2.0 + 6.0) / 2)
        assert row["AWAY_QB_CPOE_ROLLING_AVAILABLE"] == 1
        assert row["AWAY_QB_EXPERIENCE"] == 2
        assert row["AWAY_QB_CHANGE_FLAG"] == 0
        assert pd.isna(row["HOME_QB_EPA_PLAY_CAREER"])
        assert row["MARKER"] == "preserved"

    def test_carry_forward_resolves_starter_for_unplayed_game(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g1",
                    year="2025-2026",
                    week=1,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="target-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="Away Team",
                    home_team="Home Team",
                    away_qb_id=None,
                    home_qb_id="QB2",
                ),
            ]
        )
        logs = DataFrame(
            [
                _qb_log(
                    player_id="QB1",
                    season=2025,
                    week=1,
                    attempts=20,
                    passing_epa=4.0,
                    passing_cpoe=2.0,
                )
            ]
        )

        result = HomeAwayQbProfileFeature().compute(
            df=_game_frame(), datasets=_datasets(sidecar=sidecar, logs=logs)
        )
        row = result.iloc[0]
        assert row["AWAY_QB_EXPERIENCE"] == 1
        assert row["AWAY_QB_EPA_PLAY_CAREER"] == pytest.approx(4.0 / 20)

    def test_preserves_row_order(self) -> None:
        sidecar = DataFrame(
            [
                _sidecar_row(
                    game_id="g-week-4",
                    year="2025-2026",
                    week=4,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
                ),
                _sidecar_row(
                    game_id="g-week-5",
                    year="2025-2026",
                    week=5,
                    away_team="A",
                    home_team="B",
                    away_qb_id="QB1",
                    home_qb_id="QB2",
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

        result = HomeAwayQbProfileFeature().compute(df=games, datasets=_datasets(sidecar=sidecar))
        assert result["GAME_ID"].tolist() == ["g-week-5", "g-week-4"]
