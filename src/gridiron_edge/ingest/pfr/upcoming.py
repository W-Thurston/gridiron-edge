# src/gridiron_edge/ingest/pfr/upcoming.py

from gridiron_edge.ingest.pfr.collector import fetch_upcoming_schedule


def fetch_upcoming() -> None:
    """Ingest upcoming schedule from PFR via Scrapy."""
    fetch_upcoming_schedule()
