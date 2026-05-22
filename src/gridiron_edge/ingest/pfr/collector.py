# src/gridiron_edge/ingest/pfr/collector.py

"""PFR collector facade — weather and DK odds only.

Scrapy-based historical and upcoming schedule ingestion has been replaced
by nflverse (see ``gridiron_edge.ingest.nflverse``).
"""

from gridiron_edge.ingest.pfr.collector_impl import PFR_Data_Collector


def fetch_historical_weather(*, season_year: str, owm_api_key: str) -> None:
    """Pull historical weather via OpenWeatherMap for the most recent week in a season.

    Args:
        season_year: NFL season string (e.g. ``"2025-2026"``).
        owm_api_key: OpenWeatherMap API key.
    """
    collector = PFR_Data_Collector()
    collector.pull_weather_data(year=season_year, owm_api_key=owm_api_key)


def fetch_current_week_dk_odds() -> None:
    """Pull DraftKings odds for the current NFL week.

    Returns the raw wide-format DataFrame. Callers should use
    ``gridiron_edge.ingest.odds.draftkings.fetch_dk_odds`` for the full
    pipeline (wide → long → ledger → snapshot).
    """
    collector = PFR_Data_Collector()
    collector.pull_dk_sportsbook_odds()
