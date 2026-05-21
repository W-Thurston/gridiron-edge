# src/gridiron_edge/ingest/pfr/collector.py

from gridiron_edge.ingest.pfr.collector_impl import PFR_Data_Collector


def fetch_historical_games(*, all_years: bool, scrape_year: str) -> None:
    """Ingest historical week-by-week data from PFR via Scrapy.

    Args:
        all_years: If ``True``, performs a full historical scrape from 1991.
            If ``False``, appends only the season specified by ``scrape_year``.
        scrape_year: NFL season year to append (e.g. ``"2024"``).
            Ignored when ``all_years`` is ``True``.
    """
    collector = PFR_Data_Collector()
    collector.fetch_historical_data(all_years=all_years, scrape_year=scrape_year)


def fetch_upcoming_schedule() -> None:
    """Ingest upcoming schedule from PFR via Scrapy."""
    collector = PFR_Data_Collector()
    collector.fetch_upcoming_schedule_data()


def fetch_historical_weather(*, season_year: str, owm_api_key: str) -> None:
    """Pull historical weather via OpenWeatherMap for the most recent week in a season.

    Args:
        season_year: NFL season string (e.g. ``"2023-2024"``).
        owm_api_key: OpenWeatherMap API key.
    """
    collector = PFR_Data_Collector()
    collector.pull_weather_data(year=season_year, owm_api_key=owm_api_key)


def fetch_current_week_dk_odds() -> None:
    """Pull DraftKings odds for the current NFL week and write them to the legacy Excel output."""
    collector = PFR_Data_Collector()
    collector.pull_dk_sportsbook_odds()


def clean_and_transform_historical_games() -> None:
    """Clean historical week-by-week raw scrape into cleaned dataset."""
    collector = PFR_Data_Collector()
    collector.clean_historical_data()


def clean_and_transform_upcoming_schedule() -> None:
    """Clean upcoming schedule raw scrape into cleaned dataset."""
    collector = PFR_Data_Collector()
    collector.clean_upcoming_schedule_data()
