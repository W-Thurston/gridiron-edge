# tests/fixtures/repos.py
"""Composable MiniRepoBuilder for integration and e2e tests.

Unifies the two existing ``mini_repo`` fixtures (root conftest and
integration conftest) into a single builder-pattern class.

Usage::

    from tests.fixtures.repos import MiniRepoBuilder


    def test_pipeline(tmp_path):
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().with_elo_state().build()
        # repo is a Path — use it as the repo= argument
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.fixtures.dataframes import (
    make_elo_state,
    make_epa_by_game,
    make_games,
    make_stadiums,
    make_weather_enriched,
)

from gridiron_edge.datasets.registry import DATASETS
from gridiron_edge.datasets.writers import write_csv


class MiniRepoBuilder:
    """Composable test repository builder.

    Creates the directory skeleton expected by ``datasets.registry`` and
    populates only the datasets you need for a given test.  This keeps
    integration tests fast (no unnecessary I/O) and explicit about their
    data dependencies.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._root = tmp_path
        self._ensure_dirs()

    # -- internal helpers --------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create all directories defined in the dataset registry."""
        for spec in DATASETS.values():
            (self._root / spec.relpath).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, key: str, df: pd.DataFrame) -> MiniRepoBuilder:
        """Write a DataFrame to the registry-defined path.

        Uses write_csv for .csv targets and to_parquet for .parquet targets.
        """
        spec = DATASETS[key]
        path = self._root / spec.relpath
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            write_csv(self._root, key, df)
        return self

    # -- public builder API ------------------------------------------------

    def with_games(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a games dataset.  Uses factory defaults if *df* is None."""
        games = (
            df
            if df is not None
            else make_games(
                [
                    {
                        "GAME_ID": "2025_01_A_B",
                        "YEAR": "2025-2026",
                        "WEEK_NUM": 1,
                        "WINNER": "Team A",
                        "LOSER": "Team B",
                        "WIN_OR_TIE": 1,
                        "GAME_DATE": "2025-09-07",
                        "GAME_LOCATION": "NULL_VALUE",
                        "STADIUM": "Stadium A",
                    },
                    {
                        "GAME_ID": "2025_02_B_A",
                        "YEAR": "2025-2026",
                        "WEEK_NUM": 2,
                        "WINNER": "Team B",
                        "LOSER": "Team A",
                        "WIN_OR_TIE": 1,
                        "GAME_DATE": "2025-09-14",
                        "GAME_LOCATION": "NULL_VALUE",
                        "STADIUM": "Stadium B",
                    },
                ]
            )
        )
        return self._write("games", games)

    def with_stadiums(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a stadiums dataset.  Uses factory defaults if *df* is None."""
        return self._write("stadiums", df if df is not None else make_stadiums())

    def with_elo_state(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add an Elo state dataset.  Uses factory defaults if *df* is None."""
        return self._write("elo_state", df if df is not None else make_elo_state())

    def with_epa_by_game(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add an EPA-by-game dataset.  Uses factory defaults if *df* is None."""
        epa = df if df is not None else make_epa_by_game()
        # EPA is always Parquet
        path = self._root / "data" / "cleaned" / "epa_by_game.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        epa.to_parquet(path, index=False)
        return self

    def with_weather(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a weather-enriched dataset.  Uses factory defaults if *df* is None."""
        return self._write(
            "weather_enriched",
            df if df is not None else make_weather_enriched(),
        )

    def build(self) -> Path:
        """Return the repository root path."""
        return self._root
