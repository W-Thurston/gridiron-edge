# tests/features/test_team_elo_feature.py

import pandas as pd

from gridiron_edge.features.team.elo import TeamEloFeature


class DummyDatasets:
    def __init__(self, elo_state: pd.DataFrame) -> None:
        self._elo_state = elo_state

    def elo_state(self) -> pd.DataFrame:
        return self._elo_state


def test_team_elo_feature_merges_team_a_and_team_b_elo() -> None:
    elo_state = pd.DataFrame(
        [
            {"NFL_TEAM": "A", "NFL_YEAR": "2025-2026", "NFL_WEEK": 1, "ELO": 1500.0},
            {"NFL_TEAM": "B", "NFL_YEAR": "2025-2026", "NFL_WEEK": 1, "ELO": 1400.0},
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
        ],
    )

    out: pd.DataFrame = TeamEloFeature().compute(
        df=modeling,
        datasets=DummyDatasets(elo_state),
    )

    assert out["TEAM_A_ELO"].iloc[0] == 1500.0
    assert out["TEAM_B_ELO"].iloc[0] == 1400.0
