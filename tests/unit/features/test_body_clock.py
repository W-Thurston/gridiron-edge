# tests/unit/features/test_body_clock.py

"""Tests for the body-clock kickoff-time feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.body_clock import HomeAwayBodyClockFeature

_YEAR = "2025-2026"

# Franchise home coordinates: (latitude, longitude).
_ORIGINS = {
    "East": (40.81, -74.07),  # New York, Eastern
    "Central": (39.05, -94.48),  # Kansas City, Central
    "Pacific": (47.60, -122.33),  # Seattle, Pacific
    "Arizona": (33.53, -112.26),  # Glendale, Mountain Standard year-round
}


def _stadiums(year: str = _YEAR) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "HOME_TEAM": team,
                "YEAR": year,
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "ALTITUDE": 10,
                "STADIUM": f"{team} Stadium",
            }
            for team, (lat, lon) in _ORIGINS.items()
        ]
    )


def _accessor(
    date: str,
    gametime: object,
    *,
    upcoming: bool = False,
    stadiums: pd.DataFrame | None = None,
) -> MagicMock:
    datasets = MagicMock(spec=DatasetAccessor)
    if upcoming:
        datasets.games.return_value = pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "GAMETIME"])
        datasets.schedule_upcoming_rich.return_value = pd.DataFrame(
            [{"game_id": "g", "game_date": date, "game_time": gametime}]
        )
    else:
        datasets.games.return_value = pd.DataFrame(
            [{"GAME_ID": "g", "GAME_DATE": date, "GAMETIME": gametime}]
        )
        datasets.schedule_upcoming_rich.side_effect = FileNotFoundError("none")
    datasets.stadiums.return_value = _stadiums() if stadiums is None else stadiums
    return datasets


def _hours(away: str, home: str, datasets: MagicMock) -> tuple[float, float]:
    frame = pd.DataFrame(
        {"GAME_ID": ["g"], "YEAR": [_YEAR], "AWAY_TEAM": [away], "HOME_TEAM": [home]}
    )
    row = HomeAwayBodyClockFeature().compute(df=frame, datasets=datasets).iloc[0]
    return row["AWAY_BODY_CLOCK_KICKOFF_HOUR"], row["HOME_BODY_CLOCK_KICKOFF_HOUR"]


def test_registered_and_spec() -> None:
    assert FeatureRegistry.get("home_away_body_clock") is HomeAwayBodyClockFeature
    assert HomeAwayBodyClockFeature.spec.produces == [
        "AWAY_BODY_CLOCK_KICKOFF_HOUR",
        "HOME_BODY_CLOCK_KICKOFF_HOUR",
    ]


def test_one_pm_eastern_is_ten_am_pacific() -> None:
    away, home = _hours("Pacific", "East", _accessor("2025-09-14", "13:00:00"))

    assert away == pytest.approx(10.0)
    assert home == pytest.approx(13.0)


def test_central_team_is_one_hour_behind_eastern() -> None:
    away, _ = _hours("Central", "East", _accessor("2025-09-14", "13:00:00"))

    assert away == pytest.approx(12.0)


def test_minutes_are_decimal_hours() -> None:
    away, home = _hours("Pacific", "East", _accessor("2025-09-14", "20:20:00"))

    assert home == pytest.approx(20 + 20 / 60)
    assert away == pytest.approx(17 + 20 / 60)


def test_arizona_offset_follows_daylight_saving_on_the_game_date() -> None:
    september, _ = _hours("Arizona", "East", _accessor("2025-09-14", "13:00:00"))
    december, _ = _hours("Arizona", "East", _accessor("2025-12-14", "13:00:00"))

    # Arizona is 3 hours behind Eastern in summer, 2 in winter.
    assert september == pytest.approx(10.0)
    assert december == pytest.approx(11.0)


def test_upcoming_and_historical_schedules_resolve_identically() -> None:
    historical = _hours("Pacific", "East", _accessor("2025-09-14", "16:25:00"))
    upcoming = _hours("Pacific", "East", _accessor("2025-09-14", "16:25:00", upcoming=True))

    assert historical == pytest.approx(upcoming)


@pytest.mark.parametrize("gametime", [None, pd.NA, float("nan"), "", "noon", "25:00:00"])
def test_unavailable_or_invalid_gametime_is_null(gametime: object) -> None:
    away, home = _hours("Pacific", "East", _accessor("2025-09-14", gametime))

    assert pd.isna(away)
    assert pd.isna(home)


def test_unknown_team_origin_is_null_for_that_side_only() -> None:
    away, home = _hours("Nowhere FC", "East", _accessor("2025-09-14", "13:00:00"))

    assert pd.isna(away)
    assert home == pytest.approx(13.0)


def test_preserves_row_order_and_extra_columns() -> None:
    frame = pd.DataFrame(
        {
            "GAME_ID": ["g"],
            "YEAR": [_YEAR],
            "AWAY_TEAM": ["Pacific"],
            "HOME_TEAM": ["East"],
            "MARK": ["kept"],
        }
    )

    result = HomeAwayBodyClockFeature().compute(
        df=frame, datasets=_accessor("2025-09-14", "13:00:00")
    )

    assert result["MARK"].tolist() == ["kept"]


def test_missing_input_columns_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        HomeAwayBodyClockFeature().compute(
            df=pd.DataFrame({"GAME_ID": ["g"]}), datasets=_accessor("2025-09-14", "13:00:00")
        )
