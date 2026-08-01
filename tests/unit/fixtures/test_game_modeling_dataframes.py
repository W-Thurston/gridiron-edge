# tests/unit/fixtures/test_game_modeling_dataframes.py

"""Tests for shared canonical game-modeling fixtures."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.fixtures.dataframes import (
    make_games_from_modeling_df,
    make_games_modeling_df,
)

from gridiron_edge.features.pipeline import (
    canonical_feature_columns,
)


def test_modeling_fixture_is_one_row_per_game() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=6,
    )

    assert len(frame) == 6
    assert frame["GAME_ID"].is_unique


def test_modeling_fixture_contains_canonical_identity() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=2,
    )

    required = {
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_SCORE",
        "HOME_SCORE",
        "HOME_WIN",
        "ACTUAL_MARGIN",
        "ACTUAL_TOTAL",
    }

    assert required <= set(frame.columns)


def test_modeling_fixture_contains_all_canonical_features() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=2,
    )

    assert set(canonical_feature_columns()) <= set(frame.columns)


def test_modeling_fixture_targets_match_scores() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=8,
    )

    expected_margin = frame["HOME_SCORE"] - frame["AWAY_SCORE"]
    expected_total = frame["HOME_SCORE"] + frame["AWAY_SCORE"]
    expected_home_win = (frame["HOME_SCORE"] > frame["AWAY_SCORE"]).astype(int)

    pd.testing.assert_series_equal(
        frame["ACTUAL_MARGIN"],
        expected_margin,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        frame["ACTUAL_TOTAL"],
        expected_total,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        frame["HOME_WIN"],
        expected_home_win,
        check_names=False,
    )


def test_modeling_fixture_excludes_retired_orientation() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=2,
    )

    retired = {
        "TEAM_A",
        "TEAM_B",
        "HOME_FIELD",
        "RESULT",
        "TEAM_A_ELO",
        "TEAM_B_ELO",
        "PTS_WINNER",
        "PTS_LOSER",
    }

    assert retired.isdisjoint(frame.columns)
    assert not any(
        column.startswith("TEAM_A_") or column.startswith("TEAM_B_") for column in frame.columns
    )


def test_games_fixture_copies_canonical_identity() -> None:
    modeling = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=4,
    )

    games = make_games_from_modeling_df(modeling)

    pd.testing.assert_frame_equal(
        games.loc[
            :,
            [
                "GAME_ID",
                "YEAR",
                "WEEK_NUM",
                "GAME_DATE",
                "AWAY_TEAM",
                "HOME_TEAM",
                "AWAY_SCORE",
                "HOME_SCORE",
                "IS_NEUTRAL_SITE",
            ],
        ],
        modeling.loc[
            :,
            [
                "GAME_ID",
                "YEAR",
                "WEEK_NUM",
                "GAME_DATE",
                "AWAY_TEAM",
                "HOME_TEAM",
                "AWAY_SCORE",
                "HOME_SCORE",
                "IS_NEUTRAL_SITE",
            ],
        ].reset_index(drop=True),
    )


def test_games_fixture_rejects_duplicate_game_ids() -> None:
    modeling = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=2,
    )
    modeling = pd.concat(
        [
            modeling,
            modeling.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game IDs",
    ):
        make_games_from_modeling_df(modeling)


def test_modeling_fixture_has_noisy_elo_signal() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=30,
    )

    home_wins = frame.loc[
        frame["HOME_WIN"].eq(1),
        :,
    ]
    away_wins = frame.loc[
        frame["HOME_WIN"].eq(0),
        :,
    ]

    home_win_elo_diff = (home_wins["HOME_ELO"] - home_wins["AWAY_ELO"]).mean()
    away_win_elo_diff = (away_wins["HOME_ELO"] - away_wins["AWAY_ELO"]).mean()

    assert home_win_elo_diff > 0.0
    assert away_win_elo_diff < 0.0

    assert (frame["HOME_ELO"] - frame["AWAY_ELO"]).nunique() > 2


def test_modeling_fixture_balances_chronological_outcomes() -> None:
    frame = make_games_modeling_df(
        seasons=(2024,),
        games_per_season=30,
    )

    assert set(frame["HOME_WIN"].unique()) == {
        0,
        1,
    }

    for start in range(
        0,
        len(frame) - 3,
        4,
    ):
        window = frame.iloc[start : start + 4]

        assert set(window["HOME_WIN"].unique()) == {
            0,
            1,
        }
