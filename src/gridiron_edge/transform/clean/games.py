# src/gridiron_edge/transform/clean/games.py

from gridiron_edge.ingest.pfr.collector import clean_and_transform_historical_games


def clean_historical_games() -> None:
    """Clean historical week-by-week raw scrape into cleaned dataset."""
    clean_and_transform_historical_games()
