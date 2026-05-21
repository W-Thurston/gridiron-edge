# src/gridiron_edge/ingest/weather/openweather.py

from gridiron_edge.ingest.pfr.collector import fetch_historical_weather


def fetch_weather(*, season_year: str, owm_api_key: str) -> None:
    """Pull historical weather via OpenWeather for the most recent week in provided season.

    Args:
        season_year (str): NFL Season (e.g. '2023-2024')
        owm_api_key (str): Your Open Weather API key.

    """
    fetch_historical_weather(season_year=season_year, owm_api_key=owm_api_key)
