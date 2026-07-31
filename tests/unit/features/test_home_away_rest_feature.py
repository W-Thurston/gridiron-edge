# tests/unit/features/test_home_away_rest_feature.py

"""Tests for canonical Away/Home rest features."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.rest import (
    HomeAwayRestFeature,
)


def _datasets(
    games: DataFrame,
) -> MagicMock:
    """Return a controlled historical-game accessor."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games.copy()
    return datasets


def _historical_games() -> DataFrame:
    """Return completed history for two target teams."""
    return DataFrame(
        {
            "GAME_ID": [
                "away-previous",
                "home-previous",
                "future-game",
            ],
            "GAME_DATE": [
                "2024-09-12",
                "2024-09-15",
                "2024-09-29",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
                "Other Away",
                "Kansas City Chiefs",
            ],
            "HOME_TEAM": [
                "Other Home",
                "Las Vegas Raiders",
                "Future Opponent",
            ],
        }
    )


def _target_game() -> DataFrame:
    """Return one canonical upcoming matchup."""
    return DataFrame(
        {
            "GAME_ID": [
                "2024_03_KC_LV",
            ],
            "GAME_DATE": [
                "2024-09-22",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
            ],
            "HOME_TEAM": [
                "Las Vegas Raiders",
            ],
            "MARKER": [
                "preserved",
            ],
        }
    )


def test_registered_as_home_away_rest() -> None:
    assert FeatureRegistry.get("home_away_rest") is HomeAwayRestFeature


def test_spec_has_only_canonical_rest_outputs() -> None:
    assert HomeAwayRestFeature.spec.produces == [
        "AWAY_DAYS_REST",
        "HOME_DAYS_REST",
        "AWAY_SHORT_WEEK",
        "HOME_SHORT_WEEK",
        "AWAY_POST_BYE",
        "HOME_POST_BYE",
        "DAYS_REST_DIFF",
    ]


def test_computes_distinct_away_and_home_rest() -> None:
    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(_historical_games()),
    )

    row = result.iloc[0]

    assert row["AWAY_DAYS_REST"] == pytest.approx(10.0)
    assert row["HOME_DAYS_REST"] == pytest.approx(7.0)
    assert row["DAYS_REST_DIFF"] == pytest.approx(-3.0)


def test_differential_is_home_minus_away() -> None:
    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(_historical_games()),
    )

    row = result.iloc[0]

    assert row["DAYS_REST_DIFF"] == pytest.approx(row["HOME_DAYS_REST"] - row["AWAY_DAYS_REST"])


def test_future_game_does_not_leak() -> None:
    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(_historical_games()),
    )

    assert result.iloc[0]["AWAY_DAYS_REST"] == pytest.approx(10.0)


def test_short_week_threshold_matches_legacy_definition() -> None:
    history = _historical_games()
    history.loc[
        history["GAME_ID"] == "away-previous",
        "GAME_DATE",
    ] = "2024-09-18"

    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_DAYS_REST"] == pytest.approx(4.0)
    assert row["AWAY_SHORT_WEEK"] == 1.0


def test_six_days_is_not_short_week() -> None:
    history = _historical_games()
    history.loc[
        history["GAME_ID"] == "away-previous",
        "GAME_DATE",
    ] = "2024-09-16"

    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_DAYS_REST"] == pytest.approx(6.0)
    assert row["AWAY_SHORT_WEEK"] == 0.0


def test_thirteen_days_is_post_bye() -> None:
    history = _historical_games()
    history.loc[
        history["GAME_ID"] == "away-previous",
        "GAME_DATE",
    ] = "2024-09-09"

    result = HomeAwayRestFeature().compute(
        df=_target_game(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_DAYS_REST"] == pytest.approx(13.0)
    assert row["AWAY_POST_BYE"] == 1.0


def test_missing_history_remains_null() -> None:
    target = _target_game()
    target["AWAY_TEAM"] = "Unknown Team"

    result = HomeAwayRestFeature().compute(
        df=target,
        datasets=_datasets(_historical_games()),
    )

    row = result.iloc[0]

    assert pd.isna(row["AWAY_DAYS_REST"])
    assert pd.isna(row["AWAY_SHORT_WEEK"])
    assert pd.isna(row["AWAY_POST_BYE"])
    assert pd.isna(row["DAYS_REST_DIFF"])


def test_cross_season_history_is_preserved() -> None:
    history = _historical_games()
    history.loc[
        history["GAME_ID"] == "away-previous",
        "GAME_DATE",
    ] = "2024-01-07"

    target = _target_game()
    target["GAME_DATE"] = "2024-09-08"

    result = HomeAwayRestFeature().compute(
        df=target,
        datasets=_datasets(history),
    )

    assert result.iloc[0]["AWAY_DAYS_REST"] > 100


def test_preserves_input_and_unrelated_columns() -> None:
    target = _target_game()
    expected = target.copy(deep=True)

    result = HomeAwayRestFeature().compute(
        df=target,
        datasets=_datasets(_historical_games()),
    )

    pd.testing.assert_frame_equal(
        target,
        expected,
    )
    assert result["MARKER"].tolist() == ["preserved"]


@pytest.mark.parametrize(
    "column",
    [
        "GAME_ID",
        "GAME_DATE",
        "AWAY_TEAM",
        "HOME_TEAM",
    ],
)
def test_missing_target_column_is_rejected(
    column: str,
) -> None:
    target = _target_game().drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=(f"Home/away game frame is missing required columns: {column}"),
    ):
        HomeAwayRestFeature().compute(
            df=target,
            datasets=_datasets(_historical_games()),
        )


def test_duplicate_historical_team_game_is_rejected() -> None:
    history = pd.concat(
        [
            _historical_games(),
            _historical_games().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate team-game identities",
    ):
        HomeAwayRestFeature().compute(
            df=_target_game(),
            datasets=_datasets(history),
        )


def test_canonical_class_has_no_retired_orientation_names() -> None:
    source = inspect.getsource(HomeAwayRestFeature)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source
