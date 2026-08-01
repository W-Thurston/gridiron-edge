# tests/features/test_weather.py

"""Tests for canonical game-level weather feature generation.

Covers roof-derived dome state, enriched-weather unit conversions,
weather-category flags, controlled-environment overrides, and explicit
missing-weather states.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from gridiron_edge.features.registry import (
    FeatureRegistry,
)
from gridiron_edge.features.team.weather import (
    _DOME_TEMP_F,
    HomeAwayWeatherFeature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_accessor(
    games: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> MagicMock:
    acc = MagicMock()
    acc.games.return_value = games
    if weather is not None:
        acc.weather_enriched.return_value = weather
    else:
        acc.weather_enriched.side_effect = FileNotFoundError("no weather data")
    return acc


def _make_games(roof: str = "outdoors", game_id: str = "2024_01_KC_LV") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Las Vegas Raiders",
                "YEAR": "2024-2025",
                "WEEK_NUM": 1,
                "GAME_DATE": "2024-09-05",
                "GAME_LOCATION": "NULL_VALUE",
                "STADIUM": "Arrowhead Stadium",
                "ROOF": roof,
            }
        ]
    )


def _make_modeling_row(
    game_id: str = "2024_01_KC_LV",
) -> pd.DataFrame:
    """Return one canonical game row."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "YEAR": "2024-2025",
                "WEEK_NUM": 1,
                "AWAY_TEAM": ("Las Vegas Raiders"),
                "HOME_TEAM": ("Kansas City Chiefs"),
                "IS_NEUTRAL_SITE": 0,
            }
        ]
    )


def _make_weather_row(
    game_id: str = "2024_01_KC_LV",
    temp_k: float = 295.0,
    wind_mps: float = 5.0,
    weather_main: str = "Clear",
    feels_like_k: float = 293.0,
    humidity: int = 65,
    visibility: float = 10000.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "TEMP": temp_k,
                "WIND_SPEED": wind_mps,
                "WEATHER_MAIN": weather_main,
                "FEELS_LIKE": feels_like_k,
                "HUMIDITY": humidity,
                "VISIBILITY": visibility,
            }
        ]
    )


# ---------------------------------------------------------------------------
# IS_DOME derivation
# ---------------------------------------------------------------------------


class TestIsDome:
    """Tests for IS_DOME derivation from the ROOF column."""

    def test_outdoors_is_not_dome(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("outdoors")),
        )
        assert result.iloc[0]["IS_DOME"] == 0

    def test_open_is_not_dome(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("open")),
        )
        assert result.iloc[0]["IS_DOME"] == 0

    def test_dome_is_dome(self) -> None:
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("dome")),
        )
        assert result.iloc[0]["IS_DOME"] == 1

    def test_retractable_is_dome(self) -> None:
        """Retractable roof is treated as dome (conservative assumption)."""

        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("retractable")),
        )
        assert result.iloc[0]["IS_DOME"] == 1

    def test_case_insensitive(self) -> None:
        """ROOF column values should match regardless of capitalisation."""

        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("DOME")),
        )
        assert result.iloc[0]["IS_DOME"] == 1


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


class TestUnitConversions:
    """Tests for Kelvin-to-Fahrenheit and m/s-to-mph conversions."""

    def test_freezing_point_kelvin_to_fahrenheit(self) -> None:
        """273.15 K should convert to exactly 32 degF."""

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=273.15, wind_mps=0.0)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(32.0, abs=0.1)

    def test_boiling_point_kelvin_to_fahrenheit(self) -> None:
        """373.15 K should convert to 212 degF."""

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=373.15, wind_mps=0.0)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(212.0, abs=0.1)

    def test_ten_mps_to_mph(self) -> None:
        """10 m/s should convert to approximately 22.4 mph."""

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=295.0, wind_mps=10.0)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(22.37, abs=0.05)

    def test_zero_wind_converts_to_zero(self) -> None:
        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=295.0, wind_mps=0.0)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Precipitation flag
# ---------------------------------------------------------------------------


class TestPrecipFlag:
    """Tests for PRECIP_FLAG derivation from WEATHER_MAIN."""

    @pytest.mark.parametrize("weather_main", ["Rain", "Snow", "Drizzle", "Thunderstorm"])
    def test_precipitation_types_flag_as_1(self, weather_main: str) -> None:
        games = _make_games("outdoors")
        weather = _make_weather_row(weather_main=weather_main)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 1

    @pytest.mark.parametrize("weather_main", ["Clear", "Clouds", "Mist", "Fog"])
    def test_non_precipitation_types_flag_as_0(self, weather_main: str) -> None:
        games = _make_games("outdoors")
        weather = _make_weather_row(weather_main=weather_main)
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 0

    def test_unknown_weather_string_is_not_precipitation(self) -> None:
        """An unknown non-null category does not indicate precipitation."""

        # Build a game with WEATHER_MAIN = "NULL_VALUE" and outdoor roof
        games = _make_games(roof="outdoors", game_id="g_null")
        modeling = _make_modeling_row(game_id="g_null")
        weather = _make_weather_row(
            game_id="g_null",
            temp_k=300.0,
            wind_mps=5.0,
            weather_main="NULL_VALUE",
        )
        acc = _make_accessor(games, weather)
        out = HomeAwayWeatherFeature().compute(df=modeling, datasets=acc)
        assert out["PRECIP_FLAG"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Dome environmental overrides
# ---------------------------------------------------------------------------


class TestDomeOverrides:
    """Tests for dome game environmental value overrides."""

    def test_dome_wind_is_zero(self) -> None:
        games = _make_games("dome")
        weather = _make_weather_row(wind_mps=20.0, weather_main="Rain")
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(0.0)

    def test_dome_temp_is_controlled(self) -> None:
        """Dome temperature should be the standard controlled value (72 degF)."""

        games = _make_games("dome")
        weather = _make_weather_row(temp_k=250.0)  # very cold outside
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(_DOME_TEMP_F)

    def test_dome_precip_is_zero(self) -> None:
        games = _make_games("dome")
        weather = _make_weather_row(weather_main="Snow")
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 0

    def test_dome_overrides_even_without_owm_data(self) -> None:
        """Dome values should be set even when no OWM data is available."""

        games = _make_games("dome")
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games),  # no weather data
        )
        assert result.iloc[0]["IS_DOME"] == 1
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(0.0)
        assert result.iloc[0]["TEMP_F"] == pytest.approx(_DOME_TEMP_F)
        assert result.iloc[0]["PRECIP_FLAG"] == 0


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------


class TestMissingData:
    """Tests for NaN propagation when OWM data is unavailable."""

    def test_outdoor_game_without_owm_gives_nan_weather(self) -> None:
        """Outdoor games with no OWM data should have NaN weather columns."""

        games = _make_games("outdoors")
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games),  # FileNotFoundError
        )
        assert pd.isna(result.iloc[0]["WIND_SPEED_MPH"])
        assert pd.isna(result.iloc[0]["TEMP_F"])
        assert pd.isna(result.iloc[0]["PRECIP_FLAG"])

    def test_is_dome_always_populated(self) -> None:
        """IS_DOME should never be NaN regardless of OWM availability."""

        games = _make_games("outdoors")
        result = HomeAwayWeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games),
        )
        assert not pd.isna(result.iloc[0]["IS_DOME"])


# ---------------------------------------------------------------------------
# Column completeness and registration
# ---------------------------------------------------------------------------


class TestHomeAwayWeatherFeatureSpec:
    """Tests the canonical Weather feature contract."""

    def test_spec_produces_expected_columns(
        self,
    ) -> None:
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

    def test_registered_under_canonical_name(
        self,
    ) -> None:
        assert FeatureRegistry.get("home_away_weather") is HomeAwayWeatherFeature

    def test_retired_registration_is_absent(
        self,
    ) -> None:
        with pytest.raises(
            KeyError,
            match="Feature 'weather' is not registered",
        ):
            FeatureRegistry.get("weather")

    def test_compute_does_not_mutate_input(
        self,
    ) -> None:
        frame = _make_modeling_row()
        expected = frame.copy(deep=True)

        HomeAwayWeatherFeature().compute(
            df=frame,
            datasets=_make_accessor(
                _make_games("outdoors"),
                weather=_make_weather_row(),
            ),
        )

        pd.testing.assert_frame_equal(
            frame,
            expected,
        )


# ---------------------------------------------------------------------------
# Feels-like temperature
# ---------------------------------------------------------------------------


class TestFeelsLikeF:
    """Tests for FEELS_LIKE_F derivation."""

    def test_outdoor_feels_like_conversion(self) -> None:
        """FEELS_LIKE in Kelvin is converted to Fahrenheit."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(feels_like_k=293.0)  # 293K = 67.73°F
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        expected_f = (293.0 - 273.15) * 9.0 / 5.0 + 32.0
        assert abs(result["FEELS_LIKE_F"].iloc[0] - expected_f) < 0.01

    def test_dome_feels_like_override(self) -> None:
        """Dome games get standard 72.0°F feels-like."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(feels_like_k=250.0)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["FEELS_LIKE_F"].iloc[0] == 72.0


# ---------------------------------------------------------------------------
# Humidity
# ---------------------------------------------------------------------------


class TestHumidityPct:
    """Tests for HUMIDITY_PCT derivation."""

    def test_outdoor_humidity_passthrough(self) -> None:
        """Humidity is passed through directly from OWM data."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(humidity=85)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["HUMIDITY_PCT"].iloc[0] == 85.0

    def test_dome_humidity_override(self) -> None:
        """Dome games get standard 50% humidity."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(humidity=95)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["HUMIDITY_PCT"].iloc[0] == 50.0


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


class TestVisibilityM:
    """Tests for VISIBILITY_M derivation."""

    def test_outdoor_visibility_passthrough(self) -> None:
        """Visibility is passed through from OWM data."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(visibility=5000.0)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["VISIBILITY_M"].iloc[0] == 5000.0

    def test_nan_visibility_filled_with_default(self) -> None:
        """NaN visibility is filled with 10000.0 (clear-sky default)."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row()
        weather["VISIBILITY"] = float("nan")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["VISIBILITY_M"].iloc[0] == 10000.0

    def test_dome_visibility_override(self) -> None:
        """Dome games get 10000.0m visibility."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(visibility=2000.0)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["VISIBILITY_M"].iloc[0] == 10000.0


# ---------------------------------------------------------------------------
# Snow flag
# ---------------------------------------------------------------------------


class TestSnowFlag:
    """Tests for SNOW_FLAG derivation."""

    def test_snow_detected(self) -> None:
        """SNOW_FLAG is 1 when WEATHER_MAIN is Snow."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(weather_main="Snow")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["SNOW_FLAG"].iloc[0] == 1

    def test_rain_is_not_snow(self) -> None:
        """Rain does not trigger SNOW_FLAG."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(weather_main="Rain")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["SNOW_FLAG"].iloc[0] == 0

    def test_dome_snow_override(self) -> None:
        """Dome games always have SNOW_FLAG = 0."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(weather_main="Snow")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["SNOW_FLAG"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Low-visibility flag
# ---------------------------------------------------------------------------


class TestLowVisFlag:
    """Tests for LOW_VIS_FLAG derivation."""

    @pytest.mark.parametrize("main", ["Fog", "Mist", "Haze", "Smoke"])
    def test_low_vis_conditions(self, main: str) -> None:
        """LOW_VIS_FLAG is 1 for fog, mist, haze, and smoke."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(weather_main=main)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["LOW_VIS_FLAG"].iloc[0] == 1

    def test_clear_is_not_low_vis(self) -> None:
        """Clear weather does not trigger LOW_VIS_FLAG."""
        games = _make_games(roof="outdoors")
        weather = _make_weather_row(weather_main="Clear")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["LOW_VIS_FLAG"].iloc[0] == 0

    def test_dome_low_vis_override(self) -> None:
        """Dome games always have LOW_VIS_FLAG = 0."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(weather_main="Fog")
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["LOW_VIS_FLAG"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Wind chill delta
# ---------------------------------------------------------------------------


class TestWindChillDelta:
    """Tests for WIND_CHILL_DELTA derivation."""

    def test_positive_delta_cold_windy(self) -> None:
        """When feels-like < temp, delta is positive."""
        games = _make_games(roof="outdoors")
        # temp=295K (71.3°F), feels_like=290K (62.3°F) → delta ≈ 9°F
        weather = _make_weather_row(temp_k=295.0, feels_like_k=290.0)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        delta = result["WIND_CHILL_DELTA"].iloc[0]
        assert delta > 0, f"Expected positive delta, got {delta}"
        expected = ((295.0 - 273.15) - (290.0 - 273.15)) * 9.0 / 5.0
        assert abs(delta - expected) < 0.01

    def test_dome_wind_chill_delta_zero(self) -> None:
        """Dome games have WIND_CHILL_DELTA = 0.0."""
        games = _make_games(roof="dome")
        weather = _make_weather_row(temp_k=295.0, feels_like_k=280.0)
        acc = _make_accessor(games, weather)
        df = _make_modeling_row()

        result = HomeAwayWeatherFeature().compute(df=df, datasets=acc)
        assert result["WIND_CHILL_DELTA"].iloc[0] == 0.0
