# tests/unit/ingest/weather/test_backfill.py

"""Tests for historical weather backfill preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.ingest.weather.backfill import (
    _completed_games,
)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [
                "complete",
                "unplayed",
                "away-only",
                "home-only",
            ],
            "AWAY_SCORE": [
                24,
                pd.NA,
                17,
                pd.NA,
            ],
            "HOME_SCORE": [
                20,
                pd.NA,
                pd.NA,
                21,
            ],
        }
    )


def test_completed_games_requires_both_scores() -> None:
    completed = _completed_games(_games())

    assert completed["GAME_ID"].tolist() == [
        "complete",
    ]


def test_completed_games_does_not_require_result_fields() -> None:
    games = _games()

    assert "WIN_OR_TIE" not in games.columns
    assert "WINNER" not in games.columns
    assert "LOSER" not in games.columns
    assert "GAME_LOCATION" not in games.columns

    completed = _completed_games(games)

    assert completed["GAME_ID"].tolist() == [
        "complete",
    ]


@pytest.mark.parametrize(
    "missing_column",
    [
        "AWAY_SCORE",
        "HOME_SCORE",
    ],
)
def test_completed_games_requires_canonical_scores(
    missing_column: str,
) -> None:
    games = _games().drop(
        columns=[
            missing_column,
        ]
    )

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        _completed_games(games)


def test_completed_games_does_not_mutate_input() -> None:
    games = _games()
    expected = games.copy(deep=True)

    _completed_games(games)

    pd.testing.assert_frame_equal(
        games,
        expected,
    )
