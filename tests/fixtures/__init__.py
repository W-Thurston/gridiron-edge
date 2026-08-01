# tests/fixtures/__init__.py
"""Shared test fixtures and factory functions.

Convenience re-exports so tests can write::

    from tests.fixtures import make_games, MiniRepoBuilder
"""

from tests.fixtures.dataframes import (
    make_accessor,
    make_elo_state,
    make_epa_by_game,
    make_eval_df,
    make_games,
    make_predictions,
    make_stadiums,
    make_weather_enriched,
)
from tests.fixtures.repos import MiniRepoBuilder

__all__ = [
    "MiniRepoBuilder",
    "make_accessor",
    "make_elo_state",
    "make_epa_by_game",
    "make_eval_df",
    "make_games",
    "make_predictions",
    "make_stadiums",
    "make_weather_enriched",
]
