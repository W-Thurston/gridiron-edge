# src/gridiron_edge/datasets/accessor.py

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gridiron_edge.datasets import loaders


@dataclass(frozen=True)
class DatasetAccessor:
    """Thin facade for loading canonical datasets from disk.

    Wraps ``gridiron_edge.datasets.loaders`` functions with a
    repository-root-scoped interface. Intended to be passed into
    feature and model pipeline stages so they remain decoupled from
    the filesystem layout.

    Attributes:
        repo: Absolute path to the repository root.
    """

    repo: Path

    def games(self) -> pd.DataFrame:
        """Load the cleaned historical games dataset.

        Returns:
            DataFrame of all historical NFL game results.
        """
        return loaders.load_games(self.repo)

    def elo_state(self) -> pd.DataFrame:
        """Load the Elo ratings state table.

        Returns:
            DataFrame with per-team Elo ratings for each week and season.
        """
        return loaders.load_elo_state(self.repo)

    def stadiums(self) -> pd.DataFrame:
        """Load the stadium reference dataset.

        Returns:
            DataFrame with stadium metadata including coordinates and altitude.
        """
        return loaders.load_stadiums(self.repo)

    def schedule_upcoming_rich(self) -> pd.DataFrame:
        """Load the rich schedule-complete upcoming-game artifact."""
        return loaders.load_schedule_upcoming_rich(self.repo)

    def epa_by_game(self) -> pd.DataFrame:
        """Load the pre-aggregated game-level EPA statistics.

        Returns:
            DataFrame with one row per (team, game) containing rolling
            EPA metrics. Empty DataFrame if no EPA data has been ingested.
        """
        return loaders.load_epa_by_game(self.repo)

    def weather_enriched(self) -> pd.DataFrame:
        """Load the weather-enriched game dataset.

        Returns:
            DataFrame with weather columns (TEMP, WIND_SPEED, WEATHER_MAIN,
            etc.) keyed on GAME_ID. Empty DataFrame if not yet ingested.

        Raises:
            FileNotFoundError: If weather_enriched dataset does not exist.
                Callers should handle this gracefully (weather is optional).
        """
        return loaders.load_csv(self.repo, "weather_enriched")
