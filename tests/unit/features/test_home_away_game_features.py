# tests/unit/features/test_home_away_game_features.py

"""Tests for canonical schedule-complete game-level features."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)
from gridiron_edge.features.team.divisional import HomeAwayDivisionalFeature
from gridiron_edge.features.team.primetime import HomeAwayPrimetimeFeature
from gridiron_edge.features.team.weather import (
    _DOME_TEMP_F,
    HomeAwayWeatherFeature,
)


def _datasets(
    *,
    games: DataFrame | None = None,
    upcoming: DataFrame | None = None,
    weather: DataFrame | None = None,
) -> MagicMock:
    """Return controlled historical, upcoming, and weather datasets."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = DataFrame() if games is None else games.copy()
    datasets.schedule_upcoming_rich.return_value = (
        DataFrame() if upcoming is None else upcoming.copy()
    )
    if weather is None:
        datasets.weather_enriched.side_effect = FileNotFoundError
    else:
        datasets.weather_enriched.return_value = weather.copy()
    return datasets


def _target(*game_ids: str) -> DataFrame:
    """Return canonical target rows in the requested order."""
    return DataFrame(
        {
            "GAME_ID": list(game_ids),
            "YEAR": ["2025-2026"] * len(game_ids),
            "WEEK_NUM": [1] * len(game_ids),
            "AWAY_TEAM": ["Away Team"] * len(game_ids),
            "HOME_TEAM": ["Home Team"] * len(game_ids),
            "MARKER": list(range(len(game_ids))),
        }
    )


def _historical_metadata(
    *,
    game_id: str = "historical",
    divisional: object = 1,
    day: object = "Sunday",
    gametime: object = "20:20:00",
    roof: object = "outdoors",
) -> DataFrame:
    """Return one historical game metadata row."""
    return DataFrame(
        {
            "GAME_ID": [game_id],
            "DIV_GAME": [divisional],
            "GAME_DAY_OF_WEEK": [day],
            "GAMETIME": [gametime],
            "ROOF": [roof],
        }
    )


def _upcoming_metadata(
    *,
    game_id: str = "upcoming",
    divisional: object = 0,
    day: object = "Sunday",
    gametime: object = "13:00:00",
    roof: object = "dome",
) -> DataFrame:
    """Return one rich upcoming metadata row."""
    return DataFrame(
        {
            "game_id": [game_id],
            "divisional": [divisional],
            "game_day_of_week": [day],
            "game_time": [gametime],
            "roof": [roof],
        }
    )


def _weather(game_id: str = "historical") -> DataFrame:
    """Return one raw weather observation."""
    return DataFrame(
        {
            "GAME_ID": [game_id],
            "TEMP": [273.15],
            "WIND_SPEED": [10.0],
            "WEATHER_MAIN": ["Rain"],
            "FEELS_LIKE": [270.15],
            "HUMIDITY": [80],
            "VISIBILITY": [8000.0],
        }
    )


class TestGameMetadataLookup:
    """Tests for shared historical and upcoming metadata resolution."""

    def test_combines_historical_and_upcoming_rows(self) -> None:
        result = build_game_metadata_lookup(
            historical=_historical_metadata(),
            upcoming=_upcoming_metadata(),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "DIV_GAME": "VALUE",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "divisional": "VALUE",
            },
        )

        assert result["GAME_ID"].tolist() == ["historical", "upcoming"]
        assert result["VALUE"].tolist() == [1, 0]

    def test_identical_overlap_collapses_to_one_row(self) -> None:
        result = build_game_metadata_lookup(
            historical=_historical_metadata(game_id="same", divisional=1),
            upcoming=_upcoming_metadata(game_id="same", divisional=1),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "DIV_GAME": "VALUE",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "divisional": "VALUE",
            },
        )

        assert len(result) == 1
        assert result.iloc[0]["VALUE"] == 1

    def test_populated_value_wins_over_null(self) -> None:
        result = build_game_metadata_lookup(
            historical=_historical_metadata(game_id="same", divisional=None),
            upcoming=_upcoming_metadata(game_id="same", divisional=1),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "DIV_GAME": "VALUE",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "divisional": "VALUE",
            },
        )

        assert result.iloc[0]["VALUE"] == 1

    def test_conflicting_overlap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same/VALUE"):
            build_game_metadata_lookup(
                historical=_historical_metadata(game_id="same", divisional=0),
                upcoming=_upcoming_metadata(game_id="same", divisional=1),
                historical_mapping={
                    "GAME_ID": "GAME_ID",
                    "DIV_GAME": "VALUE",
                },
                upcoming_mapping={
                    "game_id": "GAME_ID",
                    "divisional": "VALUE",
                },
            )

    def test_missing_upcoming_artifact_returns_empty_frame(self) -> None:
        datasets = _datasets()
        datasets.schedule_upcoming_rich.side_effect = FileNotFoundError

        assert load_optional_upcoming_metadata(datasets).empty


class TestHomeAwayDivisionalFeature:
    """Tests for canonical divisional-game identity."""

    def test_registration_and_output(self) -> None:
        assert FeatureRegistry.get("home_away_divisional") is HomeAwayDivisionalFeature
        assert HomeAwayDivisionalFeature.spec.produces == ["IS_DIV_GAME"]

    @pytest.mark.parametrize("value", [0, 1])
    def test_historical_value(self, value: int) -> None:
        result = HomeAwayDivisionalFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(divisional=value),
            ),
        )

        assert result.iloc[0]["IS_DIV_GAME"] == value

    def test_upcoming_value_without_historical_game(self) -> None:
        result = HomeAwayDivisionalFeature().compute(
            df=_target("upcoming"),
            datasets=_datasets(upcoming=_upcoming_metadata(divisional=1)),
        )

        assert result.iloc[0]["IS_DIV_GAME"] == 1

    def test_missing_value_remains_null(self) -> None:
        result = HomeAwayDivisionalFeature().compute(
            df=_target("unknown"),
            datasets=_datasets(),
        )

        assert pd.isna(result.iloc[0]["IS_DIV_GAME"])

    @pytest.mark.parametrize("value", [2, -1, "invalid"])
    def test_invalid_value_is_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match="only 0, 1, or null"):
            HomeAwayDivisionalFeature().compute(
                df=_target("historical"),
                datasets=_datasets(
                    games=_historical_metadata(divisional=value),
                ),
            )


class TestHomeAwayPrimetimeFeature:
    """Tests for canonical nullable primetime identity."""

    def test_registration_and_output(self) -> None:
        assert FeatureRegistry.get("home_away_primetime") is HomeAwayPrimetimeFeature
        assert HomeAwayPrimetimeFeature.spec.produces == ["IS_PRIMETIME"]

    @pytest.mark.parametrize(
        ("day", "gametime", "expected"),
        [
            ("Monday", "13:00:00", 1),
            ("Sunday", "20:20:00", 1),
            ("Sunday", "13:00:00", 0),
            ("Thursday", "20:20:00", 1),
            ("Thursday", "12:30:00", 0),
            ("Saturday", "20:00:00", 1),
            ("Saturday", "12:00:00", 0),
            ("Tuesday", "20:00:00", 0),
        ],
    )
    def test_known_slots(
        self,
        day: str,
        gametime: str,
        expected: int,
    ) -> None:
        result = HomeAwayPrimetimeFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(day=day, gametime=gametime),
            ),
        )

        assert result.iloc[0]["IS_PRIMETIME"] == expected

    def test_upcoming_slot_without_historical_game(self) -> None:
        result = HomeAwayPrimetimeFeature().compute(
            df=_target("upcoming"),
            datasets=_datasets(
                upcoming=_upcoming_metadata(
                    day="Thursday",
                    gametime="20:15:00",
                ),
            ),
        )

        assert result.iloc[0]["IS_PRIMETIME"] == 1

    @pytest.mark.parametrize(
        ("day", "gametime"),
        [
            (None, "20:20:00"),
            ("Sunday", None),
            ("Sunday", "invalid"),
            ("Unknown", "20:20:00"),
        ],
    )
    def test_unusable_metadata_remains_null(
        self,
        day: object,
        gametime: object,
    ) -> None:
        result = HomeAwayPrimetimeFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(day=day, gametime=gametime),
            ),
        )

        assert pd.isna(result.iloc[0]["IS_PRIMETIME"])


class TestHomeAwayWeatherFeature:
    """Tests for canonical game-level weather and dome state."""

    def test_registration_and_outputs(self) -> None:
        assert FeatureRegistry.get("home_away_weather") is HomeAwayWeatherFeature
        assert HomeAwayWeatherFeature.spec.produces == [
            "IS_DOME",
            "WIND_SPEED_MPH",
            "TEMP_F",
            "PRECIP_FLAG",
            "FEELS_LIKE_F",
            "HUMIDITY_PCT",
            "VISIBILITY_M",
            "SNOW_FLAG",
            "LOW_VIS_FLAG",
            "WIND_CHILL_DELTA",
        ]

    @pytest.mark.parametrize(
        ("roof", "expected"),
        [
            ("dome", 1),
            ("DOME", 1),
            ("retractable", 1),
            ("outdoors", 0),
            ("open", 0),
        ],
    )
    def test_historical_roof_state(self, roof: str, expected: int) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(roof=roof),
            ),
        )

        assert result.iloc[0]["IS_DOME"] == expected

    def test_upcoming_roof_without_historical_game(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("upcoming"),
            datasets=_datasets(upcoming=_upcoming_metadata(roof="dome")),
        )

        assert result.iloc[0]["IS_DOME"] == 1
        assert result.iloc[0]["TEMP_F"] == pytest.approx(_DOME_TEMP_F)

    def test_outdoor_weather_conversions(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(roof="outdoors"),
                weather=_weather(),
            ),
        )
        row = result.iloc[0]

        assert row["IS_DOME"] == 0
        assert row["TEMP_F"] == pytest.approx(32.0, abs=0.01)
        assert row["WIND_SPEED_MPH"] == pytest.approx(22.3694, abs=0.01)
        assert row["PRECIP_FLAG"] == 1
        assert row["SNOW_FLAG"] == 0
        assert row["LOW_VIS_FLAG"] == 0
        assert row["HUMIDITY_PCT"] == pytest.approx(80.0)
        assert row["VISIBILITY_M"] == pytest.approx(8000.0)
        assert row["WIND_CHILL_DELTA"] == pytest.approx(5.4, abs=0.01)

    def test_dome_defaults_without_weather_artifact(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(roof="dome"),
            ),
        )
        row = result.iloc[0]

        assert row["IS_DOME"] == 1
        assert row["WIND_SPEED_MPH"] == pytest.approx(0.0)
        assert row["TEMP_F"] == pytest.approx(_DOME_TEMP_F)
        assert row["PRECIP_FLAG"] == 0
        assert row["WIND_CHILL_DELTA"] == pytest.approx(0.0)

    def test_outdoor_without_weather_remains_null(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("historical"),
            datasets=_datasets(
                games=_historical_metadata(roof="outdoors"),
            ),
        )
        row = result.iloc[0]

        assert row["IS_DOME"] == 0
        assert pd.isna(row["WIND_SPEED_MPH"])
        assert pd.isna(row["TEMP_F"])
        assert pd.isna(row["PRECIP_FLAG"])

    def test_missing_roof_remains_null(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_target("unknown"),
            datasets=_datasets(),
        )

        assert pd.isna(result.iloc[0]["IS_DOME"])


@pytest.mark.parametrize(
    "feature",
    [
        HomeAwayDivisionalFeature(),
        HomeAwayPrimetimeFeature(),
        HomeAwayWeatherFeature(),
    ],
)
def test_missing_game_id_is_rejected(feature: object) -> None:
    target = _target("historical").drop(columns=["GAME_ID"])

    with pytest.raises(
        ValueError,
        match="Home/away game frame is missing required columns: GAME_ID",
    ):
        feature.compute(
            df=target,
            datasets=_datasets(games=_historical_metadata()),
        )


@pytest.mark.parametrize(
    "feature_class",
    [
        HomeAwayDivisionalFeature,
        HomeAwayPrimetimeFeature,
        HomeAwayWeatherFeature,
    ],
)
def test_canonical_classes_have_no_retired_orientation_names(
    feature_class: type,
) -> None:
    source = inspect.getsource(feature_class)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source


def test_features_preserve_order_columns_and_input() -> None:
    target = _target("upcoming", "historical", "unknown")
    expected = target.copy(deep=True)
    datasets = _datasets(
        games=_historical_metadata(),
        upcoming=_upcoming_metadata(),
        weather=_weather(),
    )

    divisional = HomeAwayDivisionalFeature().compute(
        df=target,
        datasets=datasets,
    )
    primetime = HomeAwayPrimetimeFeature().compute(
        df=target,
        datasets=datasets,
    )
    weather = HomeAwayWeatherFeature().compute(
        df=target,
        datasets=datasets,
    )

    pd.testing.assert_frame_equal(target, expected)
    for result in (divisional, primetime, weather):
        assert result["GAME_ID"].tolist() == [
            "upcoming",
            "historical",
            "unknown",
        ]
        assert result["MARKER"].tolist() == [0, 1, 2]
