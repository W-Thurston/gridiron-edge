# tests/integration/test_odds_join.py
"""Integration: verify DK odds join to predictions via game_id."""

from __future__ import annotations

from re import Pattern

import pandas as pd
from pandas import DataFrame

from gridiron_edge.ingest.odds._game_id import resolve_dk_game_ids


class TestOddsJoinViability:
    """Verify that game_id resolution produces joinable keys."""

    def test_resolved_ids_match_canonical_format(self) -> None:
        """game_ids should match YYYY_WW_AWAY_HOME pattern."""
        import re

        df = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs", "Buffalo Bills", "Green Bay Packers"],
                "home_team": ["Los Angeles Chargers", "Miami Dolphins", "Chicago Bears"],
            }
        )
        result: DataFrame = resolve_dk_game_ids(df, season_year=2025, week=1)
        pattern: Pattern[str] = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")
        for gid in result["game_id"]:
            assert pattern.match(gid), f"Bad game_id format: {gid}"

    def test_join_predictions_to_odds(self) -> None:
        """Simulate joining predictions and odds on game_id."""
        predictions = pd.DataFrame(
            {
                "game_id": ["2025_01_KC_LAC", "2025_01_BUF_MIA", "2025_01_GB_CHI"],
                "away_win_prob": [0.55, 0.60, 0.45],
            }
        )
        odds = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs", "Buffalo Bills", "Green Bay Packers"],
                "home_team": ["Los Angeles Chargers", "Miami Dolphins", "Chicago Bears"],
                "ml_away": [-150, -200, 120],
            }
        )
        odds_with_id: DataFrame = resolve_dk_game_ids(odds, season_year=2025, week=1)
        joined: DataFrame = predictions.merge(odds_with_id, on="game_id", how="inner")
        assert len(joined) == 3  # 100% match rate
        assert "away_win_prob" in joined.columns
        assert "ml_away" in joined.columns

    def test_unmatched_games_surface_as_nulls(self) -> None:
        """Left join should show which predictions lack odds."""
        predictions = pd.DataFrame(
            {
                "game_id": ["2025_01_KC_LAC", "2025_01_BUF_MIA", "2025_01_DAL_NYG"],
            }
        )
        odds = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs", "Buffalo Bills"],
                "home_team": ["Los Angeles Chargers", "Miami Dolphins"],
                "ml_away": [-150, -200],
            }
        )
        odds_with_id: DataFrame = resolve_dk_game_ids(odds, season_year=2025, week=1)
        joined: DataFrame = predictions.merge(odds_with_id, on="game_id", how="left")
        assert len(joined) == 3
        assert joined["ml_away"].isna().sum() == 1  # DAL@NYG has no odds
