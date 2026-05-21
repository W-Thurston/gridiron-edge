# src/gridiron_edge/ingest/odds/draftkings.py

from gridiron_edge.ingest.pfr.collector import fetch_current_week_dk_odds


def fetch_dk_odds() -> None:
    """Pull DraftKings odds for the current NFL week and write them to the legacy Excel output."""
    fetch_current_week_dk_odds()
