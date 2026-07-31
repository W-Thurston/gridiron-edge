# tests/unit/features/test_home_away_elo_feature.py

"""Tests for canonical Away/Home Elo feature generation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.registry import (
    FeatureRegistry,
)
from gridiron_edge.features.team.elo import (
    HomeAwayEloFeature,
)


@dataclass(frozen=True)
class _Datasets:
    """Minimal Elo dataset accessor for feature tests."""

    elo: DataFrame

    def elo_state(self) -> DataFrame:
        """Return a defensive copy of Elo state."""
        return self.elo.copy()


def _games() -> DataFrame:
    """Return representative one-row-per-game inputs."""
    return DataFrame(
        {
            "GAME_ID": [
                "2025_01_PHI_GB",
                "2025_01_KC_LAC",
            ],
            "YEAR": [
                "2025-2026",
                "2025-2026",
            ],
            "WEEK_NUM": [
                1,
                1,
            ],
            "AWAY_TEAM": [
                "Philadelphia Eagles",
                "Kansas City Chiefs",
            ],
            "HOME_TEAM": [
                "Green Bay Packers",
                "Los Angeles Chargers",
            ],
            "SOURCE_VALUE": [
                "first",
                "second",
            ],
        }
    )


def _elo() -> DataFrame:
    """Return exact weekly Elo state for both games."""
    return DataFrame(
        {
            "NFL_TEAM": [
                "Philadelphia Eagles",
                "Green Bay Packers",
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            ],
            "NFL_YEAR": [
                "2025-2026",
                "2025-2026",
                "2025-2026",
                "2025-2026",
            ],
            "NFL_WEEK": [
                1,
                1,
                1,
                1,
            ],
            "ELO": [
                1540.0,
                1490.0,
                1580.0,
                1510.0,
            ],
        }
    )


def test_feature_is_registered_under_canonical_name() -> None:
    feature_class = FeatureRegistry.get("home_away_elo")

    assert feature_class is HomeAwayEloFeature


def test_feature_spec_produces_away_and_home_elo() -> None:
    assert HomeAwayEloFeature.spec.name == ("home_away_elo")
    assert HomeAwayEloFeature.spec.produces == [
        "AWAY_ELO",
        "HOME_ELO",
    ]


def test_joins_exact_weekly_away_and_home_elo() -> None:
    result = HomeAwayEloFeature().compute(
        df=_games(),
        datasets=_Datasets(
            elo=_elo(),
        ),
    )

    assert result["AWAY_ELO"].tolist() == [
        1540.0,
        1580.0,
    ]
    assert result["HOME_ELO"].tolist() == [
        1490.0,
        1510.0,
    ]


def test_preserves_every_input_row_and_order() -> None:
    games = _games()

    result = HomeAwayEloFeature().compute(
        df=games,
        datasets=_Datasets(
            elo=_elo(),
        ),
    )

    assert result["GAME_ID"].tolist() == (games["GAME_ID"].tolist())
    assert result["SOURCE_VALUE"].tolist() == [
        "first",
        "second",
    ]
    assert len(result) == len(games)


def test_input_frame_is_not_mutated() -> None:
    games = _games()
    expected = games.copy(deep=True)

    HomeAwayEloFeature().compute(
        df=games,
        datasets=_Datasets(
            elo=_elo(),
        ),
    )

    pd.testing.assert_frame_equal(
        games,
        expected,
    )


def test_missing_away_elo_remains_null() -> None:
    elo = _elo().loc[
        lambda frame: frame["NFL_TEAM"] != "Philadelphia Eagles",
        :,
    ]

    result = HomeAwayEloFeature().compute(
        df=_games(),
        datasets=_Datasets(
            elo=elo,
        ),
    )

    first = result.iloc[0]

    assert pd.isna(first["AWAY_ELO"])
    assert first["HOME_ELO"] == 1490.0
    assert len(result) == 2


def test_missing_home_elo_remains_null() -> None:
    elo = _elo().loc[
        lambda frame: frame["NFL_TEAM"] != "Green Bay Packers",
        :,
    ]

    result = HomeAwayEloFeature().compute(
        df=_games(),
        datasets=_Datasets(
            elo=elo,
        ),
    )

    first = result.iloc[0]

    assert first["AWAY_ELO"] == 1540.0
    assert pd.isna(first["HOME_ELO"])
    assert len(result) == 2


def test_other_week_elo_does_not_satisfy_requested_week() -> None:
    elo = _elo()
    elo.loc[
        elo["NFL_TEAM"] == "Philadelphia Eagles",
        "NFL_WEEK",
    ] = 2

    result = HomeAwayEloFeature().compute(
        df=_games(),
        datasets=_Datasets(
            elo=elo,
        ),
    )

    assert pd.isna(result.iloc[0]["AWAY_ELO"])


def test_other_season_elo_does_not_satisfy_requested_season() -> None:
    elo = _elo()
    elo.loc[
        elo["NFL_TEAM"] == "Philadelphia Eagles",
        "NFL_YEAR",
    ] = "2024-2025"

    result = HomeAwayEloFeature().compute(
        df=_games(),
        datasets=_Datasets(
            elo=elo,
        ),
    )

    assert pd.isna(result.iloc[0]["AWAY_ELO"])


def test_duplicate_elo_identity_is_rejected() -> None:
    elo = pd.concat(
        [
            _elo(),
            _elo().iloc[
                [
                    0,
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Elo state contains duplicate identities",
    ):
        HomeAwayEloFeature().compute(
            df=_games(),
            datasets=_Datasets(
                elo=elo,
            ),
        )


@pytest.mark.parametrize(
    "column",
    [
        "AWAY_TEAM",
        "HOME_TEAM",
        "YEAR",
        "WEEK_NUM",
    ],
)
def test_missing_game_input_column_is_rejected(
    column: str,
) -> None:
    games = _games().drop(
        columns=[
            column,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(f"Home/away game frame is missing required columns: {column}"),
    ):
        HomeAwayEloFeature().compute(
            df=games,
            datasets=_Datasets(
                elo=_elo(),
            ),
        )


@pytest.mark.parametrize(
    "column",
    [
        "NFL_TEAM",
        "NFL_YEAR",
        "NFL_WEEK",
        "ELO",
    ],
)
def test_missing_elo_column_is_rejected(
    column: str,
) -> None:
    elo = _elo().drop(
        columns=[
            column,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(f"Elo state is missing required columns: {column}"),
    ):
        HomeAwayEloFeature().compute(
            df=_games(),
            datasets=_Datasets(
                elo=elo,
            ),
        )


def test_empty_game_frame_preserves_schema() -> None:
    games = _games().iloc[0:0,].copy()

    result = HomeAwayEloFeature().compute(
        df=games,
        datasets=_Datasets(
            elo=_elo(),
        ),
    )

    assert result.empty
    assert "AWAY_ELO" in result.columns
    assert "HOME_ELO" in result.columns


def test_canonical_feature_contains_no_retired_orientation_names() -> None:
    source = inspect.getsource(HomeAwayEloFeature)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source
