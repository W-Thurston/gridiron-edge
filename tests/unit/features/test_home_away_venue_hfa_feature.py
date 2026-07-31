# tests/unit/features/test_home_away_venue_hfa_feature.py

"""Tests for canonical Home franchise advantage features."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.venue_hfa import HomeAwayVenueHFAFeature


def _datasets(games: DataFrame) -> MagicMock:
    """Return a controlled historical-games accessor."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games.copy()
    return datasets


def _target(
    *,
    home_team: str = "Kansas City Chiefs",
    year: str = "2025-2026",
    week: int = 10,
    neutral: int = 0,
) -> DataFrame:
    """Return one canonical target game."""
    return DataFrame(
        {
            "GAME_ID": ["target"],
            "YEAR": [year],
            "WEEK_NUM": [week],
            "AWAY_TEAM": ["Las Vegas Raiders"],
            "HOME_TEAM": [home_team],
            "IS_NEUTRAL_SITE": [neutral],
            "MARKER": ["preserved"],
        }
    )


def _game(
    *,
    game_id: str,
    year: str,
    week: int,
    home_team: str,
    away_score: int | None,
    home_score: int | None,
    neutral: int = 0,
) -> dict[str, object]:
    """Return one canonical historical game."""
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "HOME_TEAM": home_team,
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
        "IS_NEUTRAL_SITE": neutral,
    }


def _home_results(
    *,
    team: str,
    year: str,
    wins: int,
    losses: int,
    prefix: str,
) -> list[dict[str, object]]:
    """Return completed home wins and losses for one franchise."""
    rows: list[dict[str, object]] = []
    for index in range(wins):
        rows.append(
            _game(
                game_id=f"{prefix}-win-{index}",
                year=year,
                week=index + 1,
                home_team=team,
                away_score=17,
                home_score=24,
            )
        )
    for index in range(losses):
        rows.append(
            _game(
                game_id=f"{prefix}-loss-{index}",
                year=year,
                week=wins + index + 1,
                home_team=team,
                away_score=24,
                home_score=17,
            )
        )
    return rows


def _qualified_history() -> DataFrame:
    """Return prior history with a known league and franchise rate."""
    return DataFrame(
        [
            *_home_results(
                team="Kansas City Chiefs",
                year="2024-2025",
                wins=20,
                losses=0,
                prefix="kc",
            ),
            *_home_results(
                team="Buffalo Bills",
                year="2024-2025",
                wins=0,
                losses=20,
                prefix="buf",
            ),
        ]
    )


def _compute(
    *,
    target: DataFrame | None = None,
    games: DataFrame | None = None,
) -> DataFrame:
    """Run the canonical HFA feature with controlled inputs."""
    return HomeAwayVenueHFAFeature().compute(
        df=_target() if target is None else target,
        datasets=_datasets(
            _qualified_history() if games is None else games,
        ),
    )


def test_registered_under_canonical_name() -> None:
    assert FeatureRegistry.get("home_away_venue_hfa") is HomeAwayVenueHFAFeature


def test_spec_declares_single_home_output() -> None:
    assert HomeAwayVenueHFAFeature.spec.name == "home_away_venue_hfa"
    assert HomeAwayVenueHFAFeature.spec.produces == ["HOME_FRANCHISE_HFA"]
    assert HomeAwayVenueHFAFeature.spec.depends_on == ()


def test_computes_home_franchise_rate_minus_prior_league_rate() -> None:
    row = _compute().iloc[0]

    # KC is 20-0 at home, while the prior league fixture is 20-20.
    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(0.5)


def test_neutral_target_receives_zero() -> None:
    row = _compute(target=_target(neutral=1)).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == 0.0


def test_below_minimum_home_games_receives_zero_prior() -> None:
    games = DataFrame(
        _home_results(
            team="Kansas City Chiefs",
            year="2024-2025",
            wins=19,
            losses=0,
            prefix="kc",
        )
    )

    row = _compute(games=games).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == 0.0


def test_ties_count_as_half_home_wins() -> None:
    games = _qualified_history()
    kc_tie = DataFrame(
        [
            _game(
                game_id="kc-tie",
                year="2024-2025",
                week=21,
                home_team="Kansas City Chiefs",
                away_score=21,
                home_score=21,
            )
        ]
    )
    games = pd.concat([games, kc_tie], ignore_index=True)

    row = _compute(games=games).iloc[0]

    expected_team_rate = 20.5 / 21.0
    expected_league_rate = 20.5 / 41.0
    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(expected_team_rate - expected_league_rate)


def test_historical_neutral_games_are_excluded() -> None:
    neutral_losses = DataFrame(
        [
            _game(
                game_id=f"neutral-{index}",
                year="2024-2025",
                week=index + 1,
                home_team="Kansas City Chiefs",
                away_score=40,
                home_score=0,
                neutral=1,
            )
            for index in range(20)
        ]
    )
    games = pd.concat([_qualified_history(), neutral_losses], ignore_index=True)

    row = _compute(games=games).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(0.5)


def test_current_and_future_results_do_not_leak() -> None:
    future_losses = DataFrame(
        [
            _game(
                game_id=f"future-{index}",
                year="2025-2026",
                week=10 + index,
                home_team="Kansas City Chiefs",
                away_score=40,
                home_score=0,
            )
            for index in range(20)
        ]
    )
    games = pd.concat([_qualified_history(), future_losses], ignore_index=True)

    row = _compute(games=games).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(0.5)


def test_earlier_target_season_weeks_contribute() -> None:
    current_season_wins = DataFrame(
        [
            _game(
                game_id=f"current-prior-{index}",
                year="2025-2026",
                week=index + 1,
                home_team="Buffalo Bills",
                away_score=17,
                home_score=24,
            )
            for index in range(5)
        ]
    )
    games = pd.concat([_qualified_history(), current_season_wins], ignore_index=True)
    target = _target(home_team="Buffalo Bills", week=10)

    row = _compute(target=target, games=games).iloc[0]

    expected_team_rate = 5.0 / 25.0
    expected_league_rate = 25.0 / 45.0
    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(expected_team_rate - expected_league_rate)


def test_unplayed_history_is_excluded() -> None:
    unplayed = DataFrame(
        [
            _game(
                game_id="unplayed",
                year="2024-2025",
                week=22,
                home_team="Kansas City Chiefs",
                away_score=None,
                home_score=None,
            )
        ]
    )
    games = pd.concat([_qualified_history(), unplayed], ignore_index=True)

    row = _compute(games=games).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == pytest.approx(0.5)


def test_empty_history_uses_zero_prior() -> None:
    row = _compute(games=DataFrame()).iloc[0]

    assert row["HOME_FRANCHISE_HFA"] == 0.0


def test_preserves_input_columns_and_immutability() -> None:
    target = _target()
    expected = target.copy(deep=True)

    result = _compute(target=target)

    pd.testing.assert_frame_equal(target, expected)
    assert result["GAME_ID"].tolist() == ["target"]
    assert result["MARKER"].tolist() == ["preserved"]


@pytest.mark.parametrize(
    "column",
    ["GAME_ID", "YEAR", "WEEK_NUM", "HOME_TEAM", "IS_NEUTRAL_SITE"],
)
def test_missing_target_column_is_rejected(column: str) -> None:
    target = _target().drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=f"Home/away game frame is missing required columns: {column}",
    ):
        _compute(target=target)


def test_missing_historical_column_is_rejected() -> None:
    games = _qualified_history().drop(columns=["HOME_SCORE"])

    with pytest.raises(
        ValueError,
        match="Historical games is missing required columns: HOME_SCORE",
    ):
        _compute(games=games)


def test_duplicate_historical_game_ids_are_rejected() -> None:
    games = pd.concat(
        [_qualified_history(), _qualified_history().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Historical games contain duplicate game IDs",
    ):
        _compute(games=games)


@pytest.mark.parametrize("value", [-1, 2])
def test_invalid_target_neutral_state_is_rejected(value: int) -> None:
    target = _target(neutral=value)

    with pytest.raises(
        ValueError,
        match="Home/away game frame IS_NEUTRAL_SITE must contain only 0 or 1",
    ):
        _compute(target=target)


def test_invalid_season_label_is_rejected() -> None:
    target = _target(year="invalid")

    with pytest.raises(
        ValueError,
        match="YEAR must begin with a numeric season",
    ):
        _compute(target=target)


def test_canonical_class_has_no_retired_orientation_names() -> None:
    source = inspect.getsource(HomeAwayVenueHFAFeature)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source
    assert "WINNER" not in source
    assert "LOSER" not in source
    assert "GAME_LOCATION" not in source
