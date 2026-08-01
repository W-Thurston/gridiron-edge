# tests/unit/models/game_prediction/test_game_schema.py

"""Tests for the canonical home/away game-prediction schema."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from gridiron_edge.models.game_prediction.game_schema import (
    ACTUAL_MARGIN_TARGET,
    ACTUAL_TOTAL_TARGET,
    AWAY_FEATURE_PREFIX,
    AWAY_WIN_PROBABILITY,
    GAME_IDENTITY_COLUMNS,
    GAME_PREDICTION_COLUMNS,
    GAME_SCORE_COLUMNS,
    GAME_TARGET_COLUMNS,
    HOME_FEATURE_PREFIX,
    HOME_WIN_PROBABILITY,
    HOME_WIN_TARGET,
    MODEL_SPREAD_COLUMN,
    MODEL_TOTAL_COLUMN,
    OPTIONAL_GAME_IDENTITY_COLUMNS,
    PROJECTED_AWAY_SCORE_COLUMN,
    PROJECTED_HOME_SCORE_COLUMN,
    WIN_PROBABILITY_COLUMNS,
    away_feature_name,
    home_feature_name,
    home_minus_away_feature_name,
)


def test_game_identity_uses_stable_home_away_orientation() -> None:
    assert GAME_IDENTITY_COLUMNS == (
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
    )


def test_optional_identity_preserves_date_and_neutral_site() -> None:
    assert OPTIONAL_GAME_IDENTITY_COLUMNS == (
        "GAME_DATE",
        "IS_NEUTRAL_SITE",
    )


def test_score_columns_use_home_away_orientation() -> None:
    assert GAME_SCORE_COLUMNS == (
        "AWAY_SCORE",
        "HOME_SCORE",
    )


def test_targets_use_home_team_perspective() -> None:
    assert GAME_TARGET_COLUMNS == (
        "HOME_WIN",
        "ACTUAL_MARGIN",
        "ACTUAL_TOTAL",
    )
    assert HOME_WIN_TARGET == "HOME_WIN"
    assert ACTUAL_MARGIN_TARGET == "ACTUAL_MARGIN"
    assert ACTUAL_TOTAL_TARGET == "ACTUAL_TOTAL"


def test_win_probability_columns_use_home_away_orientation() -> None:
    assert WIN_PROBABILITY_COLUMNS == (
        "AWAY_WIN_PROB",
        "HOME_WIN_PROB",
    )
    assert AWAY_WIN_PROBABILITY == "AWAY_WIN_PROB"
    assert HOME_WIN_PROBABILITY == "HOME_WIN_PROB"


def test_prediction_columns_include_canonical_identity_and_outputs() -> None:
    assert GAME_PREDICTION_COLUMNS == (
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_WIN_PROB",
        "HOME_WIN_PROB",
        "MODEL_SPREAD",
        "MODEL_TOTAL",
        "PROJECTED_AWAY_SCORE",
        "PROJECTED_HOME_SCORE",
    )


def test_model_output_column_constants_are_stable() -> None:
    assert MODEL_SPREAD_COLUMN == "MODEL_SPREAD"
    assert MODEL_TOTAL_COLUMN == "MODEL_TOTAL"
    assert PROJECTED_AWAY_SCORE_COLUMN == ("PROJECTED_AWAY_SCORE")
    assert PROJECTED_HOME_SCORE_COLUMN == ("PROJECTED_HOME_SCORE")


def test_feature_prefixes_are_home_away_specific() -> None:
    assert AWAY_FEATURE_PREFIX == "AWAY_"
    assert HOME_FEATURE_PREFIX == "HOME_"


@pytest.mark.parametrize(
    ("base_name", "expected"),
    [
        ("ELO", "AWAY_ELO"),
        ("DAYS_REST", "AWAY_DAYS_REST"),
        (
            "OFF_EPA_PER_PLAY",
            "AWAY_OFF_EPA_PER_PLAY",
        ),
    ],
)
def test_away_feature_name(
    base_name: str,
    expected: str,
) -> None:
    assert away_feature_name(base_name) == expected


@pytest.mark.parametrize(
    ("base_name", "expected"),
    [
        ("ELO", "HOME_ELO"),
        ("DAYS_REST", "HOME_DAYS_REST"),
        (
            "OFF_EPA_PER_PLAY",
            "HOME_OFF_EPA_PER_PLAY",
        ),
    ],
)
def test_home_feature_name(
    base_name: str,
    expected: str,
) -> None:
    assert home_feature_name(base_name) == expected


@pytest.mark.parametrize(
    ("base_name", "expected"),
    [
        ("ELO", "ELO_DIFF"),
        ("DAYS_REST", "DAYS_REST_DIFF"),
        (
            "OFF_EPA_PER_PLAY",
            "OFF_EPA_PER_PLAY_DIFF",
        ),
    ],
)
def test_differential_feature_name(
    base_name: str,
    expected: str,
) -> None:
    assert home_minus_away_feature_name(base_name) == expected


@pytest.mark.parametrize(
    "helper",
    [
        away_feature_name,
        home_feature_name,
        home_minus_away_feature_name,
    ],
)
def test_feature_name_helpers_reject_empty_stem(
    helper: Callable[[str], str],
) -> None:
    with pytest.raises(
        ValueError,
        match="base_name must not be empty",
    ):
        helper("")


@pytest.mark.parametrize(
    "helper",
    [
        away_feature_name,
        home_feature_name,
        home_minus_away_feature_name,
    ],
)
def test_feature_name_helpers_reject_surrounding_whitespace(
    helper: Callable[[str], str],
) -> None:
    with pytest.raises(
        ValueError,
        match="surrounding whitespace",
    ):
        helper(" ELO ")
