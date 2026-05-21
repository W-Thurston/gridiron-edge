# tests/features/test_travel_feature.py

import pandas as pd

import gridiron_edge.features.team.travel as travel_feature_mod


class DummyDatasets:
    def __init__(self, games: pd.DataFrame, stadiums: pd.DataFrame) -> None:
        self._games = games
        self._stadiums = stadiums

    def games(self) -> pd.DataFrame:
        return self._games

    def stadiums(self) -> pd.DataFrame:
        return self._stadiums


def test_travel_feature_calls_metrics_travel(monkeypatch) -> None:
    modeling = pd.DataFrame(
        [
            {
                "GAME_ID": "g1",
                "TEAM_A": "A",
                "TEAM_B": "B",
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
                "HOME_FIELD": 0,
            },
        ],
    )

    games = pd.DataFrame(
        [
            {
                "YEAR": "2025-2026",
                "WEEK_NUM": 1,
                "WINNER": "A",
                "LOSER": "B",
                "STADIUM": "X",
            },
        ],
    )
    stadiums = pd.DataFrame(
        [{"STADIUM": "X", "LATITUDE": 0.0, "LONGITUDE": 0.0, "ALTITUDE": 0.0}],
    )

    def fake_add_travel_timezone_altitude(df, _games, _stadiums):
        out = df.copy()
        out["TEAM_A_KM_TRAVELED"] = 123.0
        out["TEAM_A_TZ_TRAVELED"] = 2.0
        out["ALTITUDE"] = 999.0
        return out

    monkeypatch.setattr(
        travel_feature_mod,
        "add_travel_timezone_altitude",
        fake_add_travel_timezone_altitude,
    )

    from gridiron_edge.features.team.travel import TravelFeature

    out: pd.DataFrame = TravelFeature().compute(
        df=modeling,
        datasets=DummyDatasets(games, stadiums),
    )

    assert out["TEAM_A_KM_TRAVELED"].iloc[0] == 123.0
    assert out["TEAM_A_TZ_TRAVELED"].iloc[0] == 2.0
    assert out["ALTITUDE"].iloc[0] == 999.0
