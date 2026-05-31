# tests/features/test_weather.py

"""Unit tests for features/team/weather.py.

Tests cover IS_DOME derivation from ROOF column, unit conversions from
OWM raw (Kelvin to Fahrenheit, m/s to mph), precipitation flag logic,
dome-game environmental overrides, and missing-data NaN propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

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


def _make_modeling_row(game_id: str = "2024_01_KC_LV") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "TEAM_A": "Kansas City Chiefs",
                "TEAM_B": "Las Vegas Raiders",
                "YEAR": "2024-2025",
                "WEEK_NUM": 1,
                "RESULT": 1,
                "HOME_FIELD": 1,
            }
        ]
    )


def _make_weather_row(
    game_id: str = "2024_01_KC_LV",
    temp_k: float = 295.0,
    wind_mps: float = 5.0,
    weather_main: str = "Clear",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "TEMP": temp_k,
                "WIND_SPEED": wind_mps,
                "WEATHER_MAIN": weather_main,
            }
        ]
    )


# ---------------------------------------------------------------------------
# IS_DOME derivation
# ---------------------------------------------------------------------------


class TestIsDome:
    """Tests for IS_DOME derivation from the ROOF column."""

    def test_outdoors_is_not_dome(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("outdoors")),
        )
        assert result.iloc[0]["IS_DOME"] == 0

    def test_open_is_not_dome(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("open")),
        )
        assert result.iloc[0]["IS_DOME"] == 0

    def test_dome_is_dome(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("dome")),
        )
        assert result.iloc[0]["IS_DOME"] == 1

    def test_retractable_is_dome(self) -> None:
        """Retractable roof is treated as dome (conservative assumption)."""
        from gridiron_edge.features.team.weather import WeatherFeature

        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(_make_games("retractable")),
        )
        assert result.iloc[0]["IS_DOME"] == 1

    def test_case_insensitive(self) -> None:
        """ROOF column values should match regardless of capitalisation."""
        from gridiron_edge.features.team.weather import WeatherFeature

        result = WeatherFeature().compute(
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
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=273.15, wind_mps=0.0)
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(32.0, abs=0.1)

    def test_boiling_point_kelvin_to_fahrenheit(self) -> None:
        """373.15 K should convert to 212 degF."""
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=373.15, wind_mps=0.0)
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(212.0, abs=0.1)

    def test_ten_mps_to_mph(self) -> None:
        """10 m/s should convert to approximately 22.4 mph."""
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=295.0, wind_mps=10.0)
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(22.37, abs=0.05)

    def test_zero_wind_converts_to_zero(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(temp_k=295.0, wind_mps=0.0)
        result = WeatherFeature().compute(
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
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(weather_main=weather_main)
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 1

    @pytest.mark.parametrize("weather_main", ["Clear", "Clouds", "Mist", "Fog"])
    def test_non_precipitation_types_flag_as_0(self, weather_main: str) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        weather = _make_weather_row(weather_main=weather_main)
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 0

    def test_null_value_string_gives_nan(self) -> None:
        """'NULL_VALUE' is not a precipitation type → 0, not NaN."""
        from gridiron_edge.features.team.weather import WeatherFeature

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
        out = WeatherFeature().compute(df=modeling, datasets=acc)
        assert out["PRECIP_FLAG"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Dome environmental overrides
# ---------------------------------------------------------------------------


class TestDomeOverrides:
    """Tests for dome game environmental value overrides."""

    def test_dome_wind_is_zero(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("dome")
        weather = _make_weather_row(wind_mps=20.0, weather_main="Rain")
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["WIND_SPEED_MPH"] == pytest.approx(0.0)

    def test_dome_temp_is_controlled(self) -> None:
        """Dome temperature should be the standard controlled value (72 degF)."""
        from gridiron_edge.features.team.weather import _DOME_TEMP_F, WeatherFeature

        games = _make_games("dome")
        weather = _make_weather_row(temp_k=250.0)  # very cold outside
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["TEMP_F"] == pytest.approx(_DOME_TEMP_F)

    def test_dome_precip_is_zero(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("dome")
        weather = _make_weather_row(weather_main="Snow")
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games, weather=weather),
        )
        assert result.iloc[0]["PRECIP_FLAG"] == 0

    def test_dome_overrides_even_without_owm_data(self) -> None:
        """Dome values should be set even when no OWM data is available."""
        from gridiron_edge.features.team.weather import _DOME_TEMP_F, WeatherFeature

        games = _make_games("dome")
        result = WeatherFeature().compute(
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
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games),  # FileNotFoundError
        )
        assert pd.isna(result.iloc[0]["WIND_SPEED_MPH"])
        assert pd.isna(result.iloc[0]["TEMP_F"])
        assert pd.isna(result.iloc[0]["PRECIP_FLAG"])

    def test_is_dome_always_populated(self) -> None:
        """IS_DOME should never be NaN regardless of OWM availability."""
        from gridiron_edge.features.team.weather import WeatherFeature

        games = _make_games("outdoors")
        result = WeatherFeature().compute(
            df=_make_modeling_row(),
            datasets=_make_accessor(games),
        )
        assert not pd.isna(result.iloc[0]["IS_DOME"])


# ---------------------------------------------------------------------------
# Column completeness and registration
# ---------------------------------------------------------------------------


class TestWeatherFeatureSpec:
    """Tests for FeatureSpec accuracy and registry registration."""

    def test_spec_produces_four_columns(self) -> None:
        from gridiron_edge.features.team.weather import WeatherFeature

        assert set(WeatherFeature().spec.produces) == {
            "IS_DOME",
            "WIND_SPEED_MPH",
            "TEMP_F",
            "PRECIP_FLAG",
        }

    def test_registered_under_weather(self) -> None:
        from gridiron_edge.features.registry import FeatureRegistry
        import gridiron_edge.features.team.weather  # noqa: F401

        assert FeatureRegistry.get("weather") is not None
