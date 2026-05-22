# src/gridiron_edge/transform/clean/games.py

"""Legacy shim — delegates to the nflverse games cleaner.

``clean_historical_games()`` now cleans nflverse raw data rather than
PFR scraped data. The function name is preserved for backwards compatibility
with any callers outside the CLI.
"""

from gridiron_edge.transform.clean.games_nflverse import clean_nflverse_games


def clean_historical_games() -> None:
    """Clean nflverse raw games into the canonical games CSV.

    Delegates to ``clean_nflverse_games()``. Kept for backwards compatibility.
    """
    clean_nflverse_games()
