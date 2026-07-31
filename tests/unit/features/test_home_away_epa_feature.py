# tests/unit/features/test_home_away_epa_feature.py

"""Tests for canonical Away/Home EPA features."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import (
    DatasetAccessor,
)
from gridiron_edge.features.registry import (
    FeatureRegistry,
)
from gridiron_edge.features.team.epa import (
    EPA_COLS,
    HomeAwayEpaFeature,
)


def _datasets(
    epa: DataFrame,
) -> MagicMock:
    """Return a controlled EPA dataset accessor."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.epa_by_game.return_value = epa.copy()
    return datasets


def _game() -> DataFrame:
    """Return one canonical week-three game."""
    return DataFrame(
        {
            "GAME_ID": ["2025_03_PHI_GB"],
            "YEAR": ["2025-2026"],
            "WEEK_NUM": [3],
            "AWAY_TEAM": ["Philadelphia Eagles"],
            "HOME_TEAM": ["Green Bay Packers"],
            "MARKER": ["preserved"],
        }
    )


def _epa_row(
    *,
    game_id: str,
    week: int,
    team: str,
    value: float,
) -> dict[str, object]:
    """Return one complete game-level EPA row."""
    return {
        "game_id": game_id,
        "season": 2025,
        "week": week,
        "team": team,
        **dict.fromkeys(EPA_COLS, value),
    }


def _epa_history() -> DataFrame:
    """Return three weeks for the Away and Home teams."""
    return DataFrame(
        [
            _epa_row(
                game_id="away-week-1",
                week=1,
                team="Philadelphia Eagles",
                value=0.10,
            ),
            _epa_row(
                game_id="away-week-2",
                week=2,
                team="Philadelphia Eagles",
                value=0.20,
            ),
            _epa_row(
                game_id="away-week-3",
                week=3,
                team="Philadelphia Eagles",
                value=9.00,
            ),
            _epa_row(
                game_id="home-week-1",
                week=1,
                team="Green Bay Packers",
                value=0.30,
            ),
            _epa_row(
                game_id="home-week-2",
                week=2,
                team="Green Bay Packers",
                value=0.50,
            ),
            _epa_row(
                game_id="home-week-3",
                week=3,
                team="Green Bay Packers",
                value=9.00,
            ),
        ]
    )


def test_registered_as_home_away_epa() -> None:
    assert FeatureRegistry.get("home_away_epa") is HomeAwayEpaFeature


def test_produces_every_epa_metric_for_both_sides() -> None:
    produces = HomeAwayEpaFeature.spec.produces

    assert len(produces) == (len(EPA_COLS) * 2)

    for column in EPA_COLS:
        suffix = column.upper()
        assert f"AWAY_{suffix}" in produces
        assert f"HOME_{suffix}" in produces


def test_joins_distinct_away_and_home_epa() -> None:
    result = HomeAwayEpaFeature().compute(
        df=_game(),
        datasets=_datasets(_epa_history()),
    )

    row = result.iloc[0]

    assert row["AWAY_OFF_EPA_PER_PLAY"] == pytest.approx(0.15)
    assert row["HOME_OFF_EPA_PER_PLAY"] == pytest.approx(0.40)


def test_current_game_does_not_leak() -> None:
    result = HomeAwayEpaFeature().compute(
        df=_game(),
        datasets=_datasets(_epa_history()),
    )

    assert result.iloc[0]["AWAY_OFF_EPA_PER_PLAY"] == pytest.approx(0.15)


def test_empty_epa_source_adds_null_columns() -> None:
    result = HomeAwayEpaFeature().compute(
        df=_game(),
        datasets=_datasets(DataFrame()),
    )

    assert len(result) == 1
    assert result["AWAY_OFF_EPA_PER_PLAY"].isna().all()
    assert result["HOME_OFF_EPA_PER_PLAY"].isna().all()


def test_missing_away_history_does_not_remove_game() -> None:
    epa = _epa_history().loc[
        lambda frame: frame["team"] != "Philadelphia Eagles",
        :,
    ]

    result = HomeAwayEpaFeature().compute(
        df=_game(),
        datasets=_datasets(epa),
    )

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["AWAY_OFF_EPA_PER_PLAY"])
    assert result.iloc[0]["HOME_OFF_EPA_PER_PLAY"] == pytest.approx(0.40)


def test_preserves_input_and_unrelated_columns() -> None:
    game = _game()
    expected = game.copy(deep=True)

    result = HomeAwayEpaFeature().compute(
        df=game,
        datasets=_datasets(_epa_history()),
    )

    pd.testing.assert_frame_equal(
        game,
        expected,
    )
    assert result["MARKER"].tolist() == ["preserved"]


def test_duplicate_epa_identity_is_rejected() -> None:
    epa = pd.concat(
        [
            _epa_history(),
            _epa_history().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="EPA source contains duplicate identities",
    ):
        HomeAwayEpaFeature().compute(
            df=_game(),
            datasets=_datasets(epa),
        )


def test_invalid_season_label_is_rejected() -> None:
    game = _game()
    game["YEAR"] = "invalid"

    with pytest.raises(
        ValueError,
        match="YEAR must begin with a numeric season",
    ):
        HomeAwayEpaFeature().compute(
            df=game,
            datasets=_datasets(_epa_history()),
        )


def test_window_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="window must be at least 1",
    ):
        HomeAwayEpaFeature(window=0)
