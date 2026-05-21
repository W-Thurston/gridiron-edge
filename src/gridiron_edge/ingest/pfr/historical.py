# src/gridiron_edge/ingest/pfr/historical.py

from gridiron_edge.ingest.pfr.collector import fetch_historical_games


def fetch_historical(*, all_years: bool, year: str) -> None:
    """Ingest historical week-by-week data from PFR via Scrapy.

    Args:
        all_years (bool):
            - True: full historical scrape
            - False: append only for 'year'
        year (str): NFL year to append

    """
    fetch_historical_games(all_years=all_years, scrape_year=year)
