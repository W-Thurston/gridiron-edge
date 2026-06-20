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


class TestJoinGameIdOpponentDisambiguation:
    """Verify (season, week, team, opponent_team) join semantics (player_stats/C1)."""

    def test_basic_home_join(self) -> None:
        from gridiron_edge.transform.clean.player_stats import _join_game_id

        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "team": ["KC"],
                "opponent_team": ["BUF"],
            }
        )
        schedule = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "home_team": ["KC"],
                "away_team": ["BUF"],
                "game_id": ["2024_01_BUF_KC"],
            }
        )

        result = _join_game_id(df, schedule)
        assert result["game_id"].iloc[0] == "2024_01_BUF_KC"

    def test_basic_away_join(self) -> None:
        from gridiron_edge.transform.clean.player_stats import _join_game_id

        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "team": ["BUF"],
                "opponent_team": ["KC"],
            }
        )
        schedule = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "home_team": ["KC"],
                "away_team": ["BUF"],
                "game_id": ["2024_01_BUF_KC"],
            }
        )

        result = _join_game_id(df, schedule)
        assert result["game_id"].iloc[0] == "2024_01_BUF_KC"

    def test_postseason_disambiguation(self) -> None:
        """Two different week-19 games must resolve to distinct game_ids."""
        from gridiron_edge.transform.clean.player_stats import _join_game_id

        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [19, 19],
                "team": ["KC", "BUF"],
                "opponent_team": ["LAC", "MIA"],
            }
        )
        schedule = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [19, 19],
                "home_team": ["KC", "MIA"],
                "away_team": ["LAC", "BUF"],
                "game_id": ["2024_19_LAC_KC", "2024_19_BUF_MIA"],
            }
        )

        result = _join_game_id(df, schedule)
        assert result.loc[result["team"] == "KC", "game_id"].iloc[0] == "2024_19_LAC_KC"
        assert result.loc[result["team"] == "BUF", "game_id"].iloc[0] == "2024_19_BUF_MIA"

    def test_no_match_yields_null(self) -> None:
        from gridiron_edge.transform.clean.player_stats import _join_game_id

        df = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "team": ["KC"],
                "opponent_team": ["BUF"],
            }
        )
        schedule = pd.DataFrame(
            {
                "season": [2024],
                "week": [1],
                "home_team": ["LAR"],
                "away_team": ["SF"],
                "game_id": ["2024_01_SF_LAR"],
            }
        )

        result = _join_game_id(df, schedule)
        assert pd.isna(result["game_id"].iloc[0])
