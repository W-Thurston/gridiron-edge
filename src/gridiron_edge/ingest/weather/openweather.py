# src/gridiron_edge/ingest/weather/openweather.py

"""OpenWeatherMap weather ingestion."""

from gridiron_edge.ingest.pfr.collector import fetch_historical_weather


def fetch_weather(*, season_year: str, owm_api_key: str) -> None:
    """Pull historical weather via OpenWeatherMap for the most recent week in a season.

    Args:
        season_year: NFL season label (e.g. ``"2025-2026"``).
        owm_api_key: OpenWeatherMap API key.
    """
    fetch_historical_weather(season_year=season_year, owm_api_key=owm_api_key)
