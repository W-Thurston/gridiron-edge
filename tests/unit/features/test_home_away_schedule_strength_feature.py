# tests/unit/features/test_home_away_schedule_strength_feature.py

"""Tests for canonical Away/Home schedule-strength features."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.schedule_strength import (
    HomeAwayScheduleStrengthFeature,
)


def _datasets(*, games: DataFrame, elo: DataFrame) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games.copy()
    datasets.elo_state.return_value = elo.copy()
    return datasets


def _game(
    *,
    game_id: str,
    week: int,
    away_team: str,
    home_team: str,
    away_score: int | None,
    home_score: int | None,
    year: str = "2025-2026",
) -> dict[str, object]:
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "AWAY_TEAM": away_team,
        "HOME_TEAM": home_team,
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
    }


def _elo_row(
    *,
    team: str,
    week: int,
    elo: float,
    year: str = "2025-2026",
) -> dict[str, object]:
    return {
        "NFL_TEAM": team,
        "NFL_YEAR": year,
        "NFL_WEEK": week,
        "ELO": elo,
    }


def _history() -> DataFrame:
    return DataFrame(
        [
            _game(
                game_id="away-win",
                week=1,
                away_team="Away Team",
                home_team="Opponent One",
                away_score=24,
                home_score=17,
            ),
            _game(
                game_id="away-loss",
                week=2,
                away_team="Opponent Two",
                home_team="Away Team",
                away_score=27,
                home_score=20,
            ),
            _game(
                game_id="home-win",
                week=2,
                away_team="Home Team",
                home_team="Opponent Three",
                away_score=21,
                home_score=14,
            ),
        ]
    )


def _elo() -> DataFrame:
    return DataFrame(
        [
            _elo_row(team="Opponent One", week=1, elo=1400.0),
            _elo_row(team="Opponent Two", week=2, elo=1600.0),
            _elo_row(team="Opponent Three", week=2, elo=1500.0),
        ]
    )


def _target() -> DataFrame:
    return DataFrame(
        {
            "GAME_ID": ["target-week-3"],
            "YEAR": ["2025-2026"],
            "WEEK_NUM": [3],
            "AWAY_TEAM": ["Away Team"],
            "HOME_TEAM": ["Home Team"],
            "MARKER": ["preserved"],
        }
    )


def _compute(
    *,
    target: DataFrame | None = None,
    games: DataFrame | None = None,
    elo: DataFrame | None = None,
) -> DataFrame:
    return HomeAwayScheduleStrengthFeature().compute(
        df=_target() if target is None else target,
        datasets=_datasets(
            games=_history() if games is None else games,
            elo=_elo() if elo is None else elo,
        ),
    )


def test_registered_under_canonical_name() -> None:
    assert FeatureRegistry.get("home_away_schedule_strength") is HomeAwayScheduleStrengthFeature


def test_spec_declares_outputs_and_dependency() -> None:
    assert HomeAwayScheduleStrengthFeature.spec.name == "home_away_schedule_strength"
    assert HomeAwayScheduleStrengthFeature.spec.produces == [
        "AWAY_SOS",
        "AWAY_SOV",
        "HOME_SOS",
        "HOME_SOV",
    ]
    assert HomeAwayScheduleStrengthFeature.spec.depends_on == ("home_away_elo",)


def test_computes_distinct_away_and_home_strength() -> None:
    row = _compute().iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1500.0)
    assert row["AWAY_SOV"] == pytest.approx(1400.0)
    assert row["HOME_SOS"] == pytest.approx(1500.0)
    assert row["HOME_SOV"] == pytest.approx(1500.0)


def test_tie_counts_for_sos_but_not_sov() -> None:
    games = DataFrame(
        [
            _game(
                game_id="away-win",
                week=1,
                away_team="Away Team",
                home_team="Opponent One",
                away_score=24,
                home_score=17,
            ),
            _game(
                game_id="away-tie",
                week=2,
                away_team="Away Team",
                home_team="Tie Opponent",
                away_score=21,
                home_score=21,
            ),
        ]
    )
    elo = DataFrame(
        [
            _elo_row(team="Opponent One", week=1, elo=1400.0),
            _elo_row(team="Tie Opponent", week=2, elo=1800.0),
        ]
    )

    row = _compute(games=games, elo=elo).iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1600.0)
    assert row["AWAY_SOV"] == pytest.approx(1400.0)


def test_current_future_and_unplayed_games_do_not_leak() -> None:
    extra_games = DataFrame(
        [
            _game(
                game_id="current",
                week=3,
                away_team="Away Team",
                home_team="Current Opponent",
                away_score=40,
                home_score=0,
            ),
            _game(
                game_id="future",
                week=4,
                away_team="Away Team",
                home_team="Future Opponent",
                away_score=40,
                home_score=0,
            ),
            _game(
                game_id="unplayed",
                week=2,
                away_team="Away Team",
                home_team="Unplayed Opponent",
                away_score=None,
                home_score=None,
            ),
        ]
    )
    extra_elo = DataFrame(
        [
            _elo_row(team="Current Opponent", week=3, elo=2000.0),
            _elo_row(team="Future Opponent", week=4, elo=2100.0),
            _elo_row(team="Unplayed Opponent", week=2, elo=2200.0),
        ]
    )

    row = _compute(
        games=pd.concat([_history(), extra_games], ignore_index=True),
        elo=pd.concat([_elo(), extra_elo], ignore_index=True),
    ).iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1500.0)
    assert row["AWAY_SOV"] == pytest.approx(1400.0)


def test_missing_opponent_elo_is_excluded() -> None:
    elo = _elo().loc[
        lambda frame: frame["NFL_TEAM"] == "Opponent One",
        :,
    ]

    row = _compute(elo=elo).iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1400.0)
    assert row["AWAY_SOV"] == pytest.approx(1400.0)
    assert pd.isna(row["HOME_SOS"])
    assert pd.isna(row["HOME_SOV"])


def test_no_prior_history_uses_exact_week_league_average() -> None:
    target = _target()
    target["WEEK_NUM"] = 1
    elo = DataFrame(
        [
            _elo_row(team="Away Team", week=1, elo=1400.0),
            _elo_row(team="Home Team", week=1, elo=1500.0),
            _elo_row(team="Other Team", week=1, elo=1600.0),
        ]
    )

    row = _compute(target=target, elo=elo).iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1500.0)
    assert row["AWAY_SOV"] == pytest.approx(1500.0)
    assert row["HOME_SOS"] == pytest.approx(1500.0)
    assert row["HOME_SOV"] == pytest.approx(1500.0)


def test_winless_team_uses_neutral_sov_only() -> None:
    games = DataFrame(
        [
            _game(
                game_id="away-loss",
                week=1,
                away_team="Away Team",
                home_team="Opponent One",
                away_score=10,
                home_score=20,
            )
        ]
    )
    elo = DataFrame(
        [
            _elo_row(team="Opponent One", week=1, elo=1400.0),
            _elo_row(team="Away Team", week=3, elo=1500.0),
            _elo_row(team="Home Team", week=3, elo=1600.0),
        ]
    )

    row = _compute(games=games, elo=elo).iloc[0]

    assert row["AWAY_SOS"] == pytest.approx(1400.0)
    assert row["AWAY_SOV"] == pytest.approx(1550.0)


def test_missing_exact_week_league_prior_remains_null() -> None:
    target = _target()
    target["YEAR"] = "2026-2027"

    result = _compute(target=target)

    assert result[["AWAY_SOS", "AWAY_SOV", "HOME_SOS", "HOME_SOV"]].isna().all().all()


def test_empty_history_produces_nulls() -> None:
    result = _compute(games=DataFrame(), elo=DataFrame())

    assert result[["AWAY_SOS", "AWAY_SOV", "HOME_SOS", "HOME_SOV"]].isna().all().all()


def test_preserves_input_and_unrelated_columns() -> None:
    target = _target()
    expected = target.copy(deep=True)

    result = _compute(target=target)

    pd.testing.assert_frame_equal(target, expected)
    assert result["GAME_ID"].tolist() == ["target-week-3"]
    assert result["MARKER"].tolist() == ["preserved"]


@pytest.mark.parametrize(
    "column",
    ["GAME_ID", "YEAR", "WEEK_NUM", "AWAY_TEAM", "HOME_TEAM"],
)
def test_missing_target_column_is_rejected(column: str) -> None:
    target = _target().drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=f"Home/away game frame is missing required columns: {column}",
    ):
        _compute(target=target)


def test_missing_historical_column_is_rejected() -> None:
    games = _history().drop(columns=["HOME_SCORE"])

    with pytest.raises(
        ValueError,
        match="Historical games is missing required columns: HOME_SCORE",
    ):
        _compute(games=games)


def test_missing_elo_column_is_rejected() -> None:
    elo = _elo().drop(columns=["ELO"])

    with pytest.raises(
        ValueError,
        match="Elo state is missing required columns: ELO",
    ):
        _compute(elo=elo)


def test_duplicate_historical_game_ids_are_rejected() -> None:
    games = pd.concat([_history(), _history().iloc[[0]]], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="Historical games contain duplicate game IDs",
    ):
        _compute(games=games)


def test_duplicate_elo_identity_is_rejected() -> None:
    elo = pd.concat([_elo(), _elo().iloc[[0]]], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="Elo state contains duplicate team-season-week identities",
    ):
        _compute(elo=elo)


def test_canonical_class_has_no_retired_orientation_names() -> None:
    source = inspect.getsource(HomeAwayScheduleStrengthFeature)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source
