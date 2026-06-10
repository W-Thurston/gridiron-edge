# tests/unit/features/test_player_matchup.py
"""Tests for gridiron_edge.features.player.matchup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.player.matchup import (
    _MATCHUP_STATS,
    DEFAULT_MATCHUP_WINDOW,
    _compute_def_allowed_per_game,
    _rank_defenses,
    _rolling_def_allowed,
    build_matchup_features,
)


def _make_player_logs(n_weeks: int = 6) -> DataFrame:
    """Build minimal player game logs with two teams playing each other."""
    rows: list[dict[str, bool | float | int | str]] = []
    for week in range(1, n_weeks + 1):
        # KC offense vs LV defense
        rows.append(
            {
                "player_id": "QB1",
                "player_name": "Q.Back",
                "position": "QB",
                "team": "KC",
                "opponent_team": "LV",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_KC_LV",
                "is_skill": True,
                "passing_yards": 250 + week * 10,
                "passing_tds": 2,
                "passing_interceptions": 0,
                "passing_epa": 5.0,
                "sacks_suffered": 2,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "rushing_epa": 0.0,
                "carries": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "receiving_epa": 0.0,
                "targets": 0,
                "receptions": 0,
            }
        )
        rows.append(
            {
                "player_id": "RB1",
                "player_name": "R.Back",
                "position": "RB",
                "team": "KC",
                "opponent_team": "LV",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_KC_LV",
                "is_skill": True,
                "passing_yards": 0,
                "passing_tds": 0,
                "passing_interceptions": 0,
                "passing_epa": 0.0,
                "sacks_suffered": 0,
                "rushing_yards": 80 + week * 5,
                "rushing_tds": 1,
                "rushing_epa": 2.0,
                "carries": 20,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "receiving_epa": 0.0,
                "targets": 0,
                "receptions": 0,
            }
        )
        rows.append(
            {
                "player_id": "WR1",
                "player_name": "W.Receiver",
                "position": "WR",
                "team": "KC",
                "opponent_team": "LV",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_KC_LV",
                "is_skill": True,
                "passing_yards": 0,
                "passing_tds": 0,
                "passing_interceptions": 0,
                "passing_epa": 0.0,
                "sacks_suffered": 0,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "rushing_epa": 0.0,
                "carries": 0,
                "receiving_yards": 100 + week * 5,
                "receiving_tds": 1,
                "receiving_epa": 3.0,
                "targets": 8,
                "receptions": 5,
            }
        )
        # LV offense vs KC defense (mirror)
        rows.append(
            {
                "player_id": "QB2",
                "player_name": "Q.Back2",
                "position": "QB",
                "team": "LV",
                "opponent_team": "KC",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_KC_LV",
                "is_skill": True,
                "passing_yards": 200 + week * 5,
                "passing_tds": 1,
                "passing_interceptions": 1,
                "passing_epa": -2.0,
                "sacks_suffered": 3,
                "rushing_yards": 0,
                "rushing_tds": 0,
                "rushing_epa": 0.0,
                "carries": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "receiving_epa": 0.0,
                "targets": 0,
                "receptions": 0,
            }
        )
    return pd.DataFrame(rows)


class TestConstants:
    def test_default_matchup_window(self) -> None:
        assert DEFAULT_MATCHUP_WINDOW == 6

    def test_matchup_stats_count(self) -> None:
        assert len(_MATCHUP_STATS) == 14

    def test_matchup_stats_cover_positions(self) -> None:
        """Must have stats for QB, RB, WR, and TE matchups."""
        positions_covered = set()
        for positions, _, _ in _MATCHUP_STATS:
            positions_covered.update(positions)
        assert {"QB", "RB", "WR", "TE"}.issubset(positions_covered)


class TestComputeDefAllowedPerGame:
    def test_produces_allowed_columns(self) -> None:
        logs = _make_player_logs(n_weeks=3)
        result = _compute_def_allowed_per_game(logs)

        allowed_cols = [c for c in result.columns if c.endswith("_allowed")]
        assert len(allowed_cols) > 0
        assert "pass_yards_allowed" in allowed_cols
        assert "rush_yards_allowed" in allowed_cols

    def test_groups_by_defense(self) -> None:
        """Each row should be one (team, season, week) — the defensive team."""
        logs = _make_player_logs(n_weeks=3)
        result = _compute_def_allowed_per_game(logs)

        # LV defense faced KC offense
        lv_def = result[(result["team"] == "LV") & (result["week"] == 1)]
        assert len(lv_def) == 1

    def test_sums_across_players(self) -> None:
        """Passing yards allowed should sum all opposing QBs in that game."""
        logs = _make_player_logs(n_weeks=1)
        result = _compute_def_allowed_per_game(logs)

        # LV defense: KC's QB1 threw 260 yards in week 1
        lv_def = result[(result["team"] == "LV") & (result["week"] == 1)]
        assert lv_def["pass_yards_allowed"].iloc[0] == 260


class TestRollingDefAllowed:
    def test_no_lookahead(self) -> None:
        """Week 1 rolling values must be NaN."""
        logs = _make_player_logs(n_weeks=4)
        def_allowed = _compute_def_allowed_per_game(logs)
        result = _rolling_def_allowed(def_allowed, window=3)

        week1 = result[(result["team"] == "LV") & (result["week"] == 1)]
        assert pd.isna(week1["opp_pass_yards_allowed_L3"].iloc[0])

    def test_week2_equals_week1_value(self) -> None:
        """Week 2 rolling should equal the week 1 raw value."""
        logs = _make_player_logs(n_weeks=4)
        def_allowed = _compute_def_allowed_per_game(logs)
        result = _rolling_def_allowed(def_allowed, window=3)

        lv = result[result["team"] == "LV"].sort_values("week")
        w1_raw = lv[lv["week"] == 1]["pass_yards_allowed"].iloc[0]
        w2_roll = lv[lv["week"] == 2]["opp_pass_yards_allowed_L3"].iloc[0]
        assert w2_roll == pytest.approx(w1_raw)


class TestRankDefenses:
    def test_ranks_in_valid_range(self) -> None:
        logs = _make_player_logs(n_weeks=4)
        def_allowed = _compute_def_allowed_per_game(logs)
        def_rolling = _rolling_def_allowed(def_allowed, window=3)
        result = _rank_defenses(def_rolling, window=3)

        rank_cols = [c for c in result.columns if "rank" in c]
        assert len(rank_cols) > 0
        for col in rank_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 1

    def test_rank_1_is_toughest(self) -> None:
        """Lower allowed stats should get rank 1 (ascending=True)."""
        logs = _make_player_logs(n_weeks=4)
        def_allowed = _compute_def_allowed_per_game(logs)
        def_rolling = _rolling_def_allowed(def_allowed, window=3)
        result = _rank_defenses(def_rolling, window=3)

        # In our fixture, KC allows fewer pass yards than LV
        # KC defense faces LV QB (200+), LV defense faces KC QB (250+)
        week4 = result[result["week"] == 4]
        if len(week4) == 2:
            kc = week4[week4["team"] == "KC"]
            lv = week4[week4["team"] == "LV"]
            if not kc.empty and not lv.empty:
                kc_rank = kc["opp_pass_yards_allowed_rank_L3"].iloc[0]
                lv_rank = lv["opp_pass_yards_allowed_rank_L3"].iloc[0]
                # KC allows fewer pass yards → lower rank number (tougher)
                assert kc_rank <= lv_rank


class TestBuildMatchupFeatures:
    def test_raises_when_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cleaned player game logs not found"):
            build_matchup_features(repo=tmp_path)
