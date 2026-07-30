# tests/unit/ratings/test_elo_table.py

"""Tests for deterministic synthetic Elo Week 1 state."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ratings.elo.table import (
    EloTableConfig,
    _add_next_season_week_one,
    _next_season_label,
)


def _latest_season_games() -> DataFrame:
    """Create a historical season ending in postseason Week 22."""
    return pd.DataFrame(
        {
            "YEAR": [
                "2025-2026",
                "2025-2026",
            ],
            "WEEK_NUM": [
                21,
                22,
            ],
        }
    )


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (
            "2024-2025",
            "2025-2026",
        ),
        (
            "2025-2026",
            "2026-2027",
        ),
        (
            "2099-2100",
            "2100-2101",
        ),
    ],
)
def test_next_season_is_derived_from_history(
    year: str,
    expected: str,
) -> None:
    assert _next_season_label(year) == expected


@pytest.mark.parametrize(
    "year",
    [
        "2026",
        "2026-27",
        "2026-2028",
        "not-a-season",
        "",
    ],
)
def test_rejects_invalid_historical_season_labels(
    year: str,
) -> None:
    with pytest.raises(ValueError):
        _next_season_label(year)


def test_synthetic_week_one_uses_final_postgame_state() -> None:
    cfg = EloTableConfig(
        offseason_regress_frac=0.0,
    )
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            22,
        ): 1550.0,
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            22,
        ): 1450.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            }
        },
        cfg=cfg,
    )

    assert (
        elo[
            (
                "Kansas City Chiefs",
                "2026-2027",
                1,
            )
        ]
        == 1600.0
    )
    assert (
        elo[
            (
                "Los Angeles Chargers",
                "2026-2027",
                1,
            )
        ]
        == 1400.0
    )


def test_returning_teams_receive_offseason_regression() -> None:
    cfg = EloTableConfig(
        offseason_regress_frac=1 / 3.0,
    )
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
                "Los Angeles Chargers",
            }
        },
        cfg=cfg,
    )

    assert elo[
        (
            "Kansas City Chiefs",
            "2026-2027",
            1,
        )
    ] == pytest.approx(1566.6666666667)
    assert elo[
        (
            "Los Angeles Chargers",
            "2026-2027",
            1,
        )
    ] == pytest.approx(1433.3333333333)


def test_synthetic_transition_is_reproducible() -> None:
    cfg = EloTableConfig()
    games = _latest_season_games()
    teams_by_year = {
        "2025-2026": {
            "Kansas City Chiefs",
            "Los Angeles Chargers",
        }
    }
    original = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
        (
            "Los Angeles Chargers",
            "2025-2026",
            23,
        ): 1400.0,
    }

    first = original.copy()
    second = original.copy()

    _add_next_season_week_one(
        first,
        games=games,
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year=teams_by_year,
        cfg=cfg,
    )
    _add_next_season_week_one(
        second,
        games=games,
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year=teams_by_year,
        cfg=cfg,
    )

    assert first == second


def test_only_next_season_week_one_is_created() -> None:
    elo = {
        (
            "Kansas City Chiefs",
            "2025-2026",
            23,
        ): 1600.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
            }
        },
        cfg=EloTableConfig(),
    )

    future_keys = [key for key in elo if key[1] == "2026-2027"]

    assert future_keys == [
        (
            "Kansas City Chiefs",
            "2026-2027",
            1,
        )
    ]


def test_historical_rows_are_not_modified() -> None:
    historical_key = (
        "Kansas City Chiefs",
        "2025-2026",
        23,
    )
    elo = {
        historical_key: 1600.0,
    }

    _add_next_season_week_one(
        elo,
        games=_latest_season_games(),
        sorted_years=[
            "2025-2026",
        ],
        teams_by_year={
            "2025-2026": {
                "Kansas City Chiefs",
            }
        },
        cfg=EloTableConfig(),
    )

    assert elo[historical_key] == 1600.0


def test_empty_history_creates_no_synthetic_state() -> None:
    elo: dict[tuple[str, str, int], float] = {}

    _add_next_season_week_one(
        elo,
        games=DataFrame(
            columns=[
                "YEAR",
                "WEEK_NUM",
            ]
        ),
        sorted_years=[],
        teams_by_year={},
        cfg=EloTableConfig(),
    )

    assert elo == {}
