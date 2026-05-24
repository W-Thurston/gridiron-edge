# src/gridiron_edge/transform/clean/schedule.py

"""Legacy shim — delegates to the nflverse schedule cleaner.

``clean_upcoming_schedule()`` now cleans nflverse raw data rather than
PFR scraped data. The function name is preserved for backwards compatibility.
"""

from gridiron_edge.transform.clean.schedule_nflverse import clean_nflverse_upcoming


def clean_upcoming_schedule() -> None:
    """Clean nflverse raw upcoming schedule into the canonical schedule CSV.

    Delegates to ``clean_nflverse_upcoming()``. Kept for backwards compatibility.
    """
    clean_nflverse_upcoming()
