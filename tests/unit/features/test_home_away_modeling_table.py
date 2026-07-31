# tests/unit/features/test_home_away_modeling_table.py

"""Tests for one-row home/away modeling-table construction."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.pipeline import (
    build_home_away_modeling_table,
)


def _game(
    *,
    game_id: str = "2025_01_PHI_GB",
    year: str = "2025-2026",
    week: int = 1,
    game_date: str = "2025-09-04",
    away_team: str = "Philadelphia Eagles",
    home_team: str = "Green Bay Packers",
    away_score: int = 20,
    home_score: int = 17,
    is_neutral_site: int = 0,
) -> dict[str, object]:
    """Return one cleaned historical game."""
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "GAME_DATE": game_date,
        "AWAY_TEAM": away_team,
        "HOME_TEAM": home_team,
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
        "IS_NEUTRAL_SITE": is_neutral_site,
    }


def test_builds_one_row_per_game() -> None:
    games = DataFrame(
        [
            _game(
                game_id="2025_01_PHI_GB",
            ),
            _game(
                game_id="2025_02_KC_BAL",
                week=2,
                game_date="2025-09-11",
                away_team="Kansas City Chiefs",
                home_team="Baltimore Ravens",
            ),
        ]
    )

    modeling = build_home_away_modeling_table(games)

    assert len(modeling) == 2
    assert modeling["GAME_ID"].is_unique


def test_home_win_uses_home_perspective() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(
                    away_score=17,
                    home_score=24,
                )
            ]
        )
    )

    row = modeling.iloc[0]

    assert row["HOME_WIN"] == 1
    assert row["ACTUAL_MARGIN"] == 7
    assert row["ACTUAL_TOTAL"] == 41


def test_away_win_uses_home_perspective() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(
                    away_score=27,
                    home_score=20,
                )
            ]
        )
    )

    row = modeling.iloc[0]

    assert row["HOME_WIN"] == 0
    assert row["ACTUAL_MARGIN"] == -7
    assert row["ACTUAL_TOTAL"] == 47


def test_tie_uses_nullable_home_win() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(
                    away_score=21,
                    home_score=21,
                )
            ]
        )
    )

    row = modeling.iloc[0]

    assert pd.isna(row["HOME_WIN"])
    assert str(modeling["HOME_WIN"].dtype) == "Int64"
    assert row["ACTUAL_MARGIN"] == 0
    assert row["ACTUAL_TOTAL"] == 42


def test_neutral_site_identity_is_preserved() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(
                    is_neutral_site=1,
                )
            ]
        )
    )

    row = modeling.iloc[0]

    assert row["AWAY_TEAM"] == "Philadelphia Eagles"
    assert row["HOME_TEAM"] == "Green Bay Packers"
    assert row["IS_NEUTRAL_SITE"] == 1


def test_output_schema_is_canonical() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(),
            ]
        )
    )

    assert modeling.columns.tolist() == [
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
        "GAME_DATE",
        "AWAY_SCORE",
        "HOME_SCORE",
        "IS_NEUTRAL_SITE",
        "HOME_WIN",
        "ACTUAL_MARGIN",
        "ACTUAL_TOTAL",
    ]


def test_output_contains_no_retired_orientation_columns() -> None:
    modeling = build_home_away_modeling_table(
        DataFrame(
            [
                _game(),
            ]
        )
    )

    retired = {
        "TEAM_A",
        "TEAM_B",
        "HOME_FIELD",
        "RESULT",
    }

    assert retired.isdisjoint(modeling.columns)


def test_output_is_sorted_chronologically() -> None:
    games = DataFrame(
        [
            _game(
                game_id="2025_02_KC_BAL",
                week=2,
                game_date="2025-09-11",
            ),
            _game(
                game_id="2024_18_BUF_NE",
                year="2024-2025",
                week=18,
                game_date="2025-01-05",
            ),
            _game(
                game_id="2025_01_PHI_GB",
                week=1,
                game_date="2025-09-04",
            ),
        ]
    )

    modeling = build_home_away_modeling_table(games)

    assert modeling["GAME_ID"].tolist() == [
        "2024_18_BUF_NE",
        "2025_01_PHI_GB",
        "2025_02_KC_BAL",
    ]


def test_input_is_not_mutated() -> None:
    games = DataFrame(
        [
            _game(),
        ]
    )
    expected: DataFrame = games.copy(deep=True)

    build_home_away_modeling_table(games)

    pd.testing.assert_frame_equal(
        games,
        expected,
    )


def test_missing_required_column_is_rejected() -> None:
    games = DataFrame(
        [
            _game(),
        ]
    ).drop(columns=["HOME_TEAM"])

    with pytest.raises(
        ValueError,
        match="missing required home/away columns: HOME_TEAM",
    ):
        build_home_away_modeling_table(games)


def test_duplicate_game_ids_are_rejected() -> None:
    games = DataFrame(
        [
            _game(),
            _game(),
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate game IDs",
    ):
        build_home_away_modeling_table(games)


@pytest.mark.parametrize(
    "column",
    [
        "GAME_ID",
        "YEAR",
        "AWAY_TEAM",
        "HOME_TEAM",
    ],
)
def test_null_identity_is_rejected(
    column: str,
) -> None:
    games = DataFrame(
        [
            _game(),
        ]
    )
    games.loc[0, column] = None

    with pytest.raises(
        ValueError,
        match=f"{column}.*null",
    ):
        build_home_away_modeling_table(games)


@pytest.mark.parametrize(
    "column",
    [
        "GAME_ID",
        "YEAR",
        "AWAY_TEAM",
        "HOME_TEAM",
    ],
)
def test_empty_identity_is_rejected(
    column: str,
) -> None:
    games = DataFrame(
        [
            _game(),
        ]
    )
    games.loc[0, column] = " "

    with pytest.raises(
        ValueError,
        match=f"{column}.*empty",
    ):
        build_home_away_modeling_table(games)


def test_same_team_is_rejected() -> None:
    games = DataFrame(
        [
            _game(
                away_team="Same Team",
                home_team="Same Team",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Away and home team must differ",
    ):
        build_home_away_modeling_table(games)


@pytest.mark.parametrize(
    "column",
    [
        "AWAY_SCORE",
        "HOME_SCORE",
    ],
)
def test_null_score_is_rejected(
    column: str,
) -> None:
    games = DataFrame(
        [
            _game(),
        ]
    )
    games.loc[0, column] = None

    with pytest.raises(
        ValueError,
        match=f"{column}.*null",
    ):
        build_home_away_modeling_table(games)


@pytest.mark.parametrize(
    "column",
    [
        "AWAY_SCORE",
        "HOME_SCORE",
    ],
)
def test_negative_score_is_rejected(
    column: str,
) -> None:
    games = DataFrame(
        [
            _game(),
        ]
    )
    games.loc[0, column] = -1

    with pytest.raises(
        ValueError,
        match=f"{column}.*negative",
    ):
        build_home_away_modeling_table(games)


@pytest.mark.parametrize(
    "value",
    [
        -1,
        2,
    ],
)
def test_invalid_neutral_site_value_is_rejected(
    value: int,
) -> None:
    games = DataFrame(
        [
            _game(
                is_neutral_site=value,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"IS_NEUTRAL_SITE.*0 or 1",
    ):
        build_home_away_modeling_table(games)


def test_invalid_week_is_rejected() -> None:
    games = DataFrame(
        [
            _game(
                week=0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="WEEK_NUM must be at least 1",
    ):
        build_home_away_modeling_table(games)
