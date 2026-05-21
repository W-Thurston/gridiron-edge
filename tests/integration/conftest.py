from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.datasets.registry import DATASETS
from gridiron_edge.datasets.writers import write_csv


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Minimal repo tree with games + stadiums for pipeline integration tests."""
    for spec in DATASETS.values():
        (tmp_path / spec.relpath).parent.mkdir(parents=True, exist_ok=True)

    games = pd.DataFrame(
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
        ],
    )
    write_csv(tmp_path, "games", games)

    stadiums = pd.DataFrame(
        [
            {
                "HOME_TEAM": "Team A",
                "YEAR": "2025-2026",
                "STADIUM": "Stadium A",
                "LATITUDE": 40.0,
                "LONGITUDE": -75.0,
                "ALTITUDE": 10,
            },
            {
                "HOME_TEAM": "Team B",
                "YEAR": "2025-2026",
                "STADIUM": "Stadium B",
                "LATITUDE": 34.0,
                "LONGITUDE": -118.0,
                "ALTITUDE": 50,
            },
        ],
    )
    write_csv(tmp_path, "stadiums", stadiums)
    return tmp_path
