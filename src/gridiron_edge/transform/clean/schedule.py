# src/gridiron_edge/transform/clean/schedule.py

from gridiron_edge.ingest.pfr.collector import clean_and_transform_upcoming_schedule


def clean_upcoming_schedule() -> None:
    """Clean upcoming schedule raw scrape into cleaned dataset."""
    clean_and_transform_upcoming_schedule()
