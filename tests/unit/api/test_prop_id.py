# tests/unit/api/test_prop_id.py

"""Unit tests for api/_prop_id.py."""

from __future__ import annotations

from gridiron_edge.api._prop_id import resolve_opponent_from_game_id


class TestResolveOpponentFromGameId:
    def test_player_on_away_team_returns_home(self) -> None:
        result = resolve_opponent_from_game_id("2026_01_KC_LAC", "KC")
        assert result == "LAC"

    def test_player_on_home_team_returns_away(self) -> None:
        result = resolve_opponent_from_game_id("2026_01_KC_LAC", "LAC")
        assert result == "KC"

    def test_player_team_not_in_game_returns_none(self) -> None:
        result = resolve_opponent_from_game_id("2026_01_KC_LAC", "SEA")
        assert result is None

    def test_malformed_game_id_returns_none(self) -> None:
        assert resolve_opponent_from_game_id("malformed", "KC") is None
        assert resolve_opponent_from_game_id("KC_LAC", "KC") is None
        assert resolve_opponent_from_game_id("", "KC") is None

    def test_game_id_with_extra_parts_returns_none(self) -> None:
        # 5 parts is malformed.
        result = resolve_opponent_from_game_id("2026_01_KC_LAC_extra", "KC")
        assert result is None
