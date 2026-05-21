# tests/features/test_home_field_feature.py

import pandas as pd

from gridiron_edge.features.team.home_field import HomeFieldFeature


class DummyDatasets:
    def __init__(self, games: pd.DataFrame) -> None:
        self._games = games

    def games(self) -> pd.DataFrame:
        return self._games


def test_home_field_feature_sets_home_field_correctly() -> None:
    games = pd.DataFrame(
        [
            # Home game: GAME_LOCATION == "NULL_VALUE"
            {
                "GAME_ID": "g1",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
                "WINNER": "A",
                "LOSER": "B",
                "GAME_LOCATION": "NULL_VALUE",
            },
            # Away game: GAME_LOCATION == "@"
            {
                "GAME_ID": "g2",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
                "WINNER": "C",
                "LOSER": "D",
                "GAME_LOCATION": "@",
            },
        ],
    )

    modeling = pd.DataFrame(
        [
            {
                "GAME_ID": "g1",
                "TEAM_A": "A",
                "TEAM_B": "B",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
            },
            {
                "GAME_ID": "g1",
                "TEAM_A": "B",
                "TEAM_B": "A",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
            },
            {
                "GAME_ID": "g2",
                "TEAM_A": "C",
                "TEAM_B": "D",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
            },
            {
                "GAME_ID": "g2",
                "TEAM_A": "D",
                "TEAM_B": "C",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
            },
        ],
    )

    out: pd.DataFrame = HomeFieldFeature().compute(
        df=modeling,
        datasets=DummyDatasets(games),
    )

    assert (
        out.loc[(out.GAME_ID == "g1") & (out.TEAM_A == "A"), "HOME_FIELD"].iloc[0] == 1
    )
    assert (
        out.loc[(out.GAME_ID == "g1") & (out.TEAM_A == "B"), "HOME_FIELD"].iloc[0] == 0
    )

    assert (
        out.loc[(out.GAME_ID == "g2") & (out.TEAM_A == "C"), "HOME_FIELD"].iloc[0] == 0
    )
    assert (
        out.loc[(out.GAME_ID == "g2") & (out.TEAM_A == "D"), "HOME_FIELD"].iloc[0] == 1
    )
