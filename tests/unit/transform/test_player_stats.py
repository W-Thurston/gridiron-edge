# tests/unit/transform/test_player_stats.py
"""Tests for gridiron_edge.transform.clean.player_stats."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.transform.clean.player_stats import (
    _SKILL_POSITIONS,
    _TEAM_CODE_MAP,
    _normalize_team_codes,
)


class TestTeamCodeMap:
    def test_all_historical_codes_mapped(self) -> None:
        assert set(_TEAM_CODE_MAP.keys()) == {"OAK", "SD", "STL", "JAC"}

    def test_maps_to_current_codes(self) -> None:
        assert _TEAM_CODE_MAP["OAK"] == "LV"
        assert _TEAM_CODE_MAP["SD"] == "LAC"
        assert _TEAM_CODE_MAP["STL"] == "LA"
        assert _TEAM_CODE_MAP["JAC"] == "JAX"


class TestSkillPositions:
    def test_skill_positions_set(self) -> None:
        assert {"QB", "RB", "WR", "TE", "FB"} == _SKILL_POSITIONS

    def test_defensive_positions_excluded(self) -> None:
        defensive = {"DE", "DT", "LB", "CB", "S", "OLB", "ILB", "MLB", "FS", "SAF", "NT", "DL"}
        assert defensive.isdisjoint(_SKILL_POSITIONS)


class TestNormalizeTeamCodes:
    def test_normalizes_old_codes(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["OAK", "SD", "STL", "JAC", "KC"],
                "opponent_team": ["KC", "OAK", "SD", "STL", "JAC"],
            }
        )
        result = _normalize_team_codes(df)
        assert list(result["team"]) == ["LV", "LAC", "LA", "JAX", "KC"]
        assert list(result["opponent_team"]) == ["KC", "LV", "LAC", "LA", "JAX"]

    def test_leaves_current_codes_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "team": ["KC", "BUF", "SF"],
                "opponent_team": ["LV", "MIA", "SEA"],
            }
        )
        original_teams = list(df["team"])
        original_opps = list(df["opponent_team"])
        result = _normalize_team_codes(df)
        assert list(result["team"]) == original_teams
        assert list(result["opponent_team"]) == original_opps

    def test_handles_empty_dataframe(self) -> None:
        df = pd.DataFrame({"team": pd.Series(dtype=str), "opponent_team": pd.Series(dtype=str)})
        result = _normalize_team_codes(df)
        assert len(result) == 0
