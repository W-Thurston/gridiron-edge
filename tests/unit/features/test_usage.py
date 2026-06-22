# tests/unit/features/test_usage.py
"""Tests for gridiron_edge.features.player.usage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.player.usage import (
    _SHARE_COLS,
    DEFAULT_WINDOWS,
    _compute_per_game_shares,
    _compute_team_totals,
    _rolling_shares,
    build_usage_features,
)


def _make_team_game(
    season: int = 2024,
    week: int = 1,
    team: str = "KC",
    opponent: str = "LV",
    game_id: str | None = None,
) -> DataFrame:
    """Build a single team's worth of players for one game."""
    if game_id is None:
        game_id = f"{season}_{week:02d}_{team}_{opponent}"

    return pd.DataFrame(
        [
            {
                "player_id": f"QB1_{team}",
                "player_name": f"QB_{team}",
                "position": "QB",
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week,
                "game_id": game_id,
                "is_skill": True,
                "targets": 0,
                "carries": 3,
            },
            {
                "player_id": f"RB1_{team}",
                "player_name": f"RB_{team}",
                "position": "RB",
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week,
                "game_id": game_id,
                "is_skill": True,
                "targets": 4,
                "carries": 20,
            },
            {
                "player_id": f"WR1_{team}",
                "player_name": f"WR_{team}",
                "position": "WR",
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week,
                "game_id": game_id,
                "is_skill": True,
                "targets": 10,
                "carries": 0,
            },
            {
                "player_id": f"TE1_{team}",
                "player_name": f"TE_{team}",
                "position": "TE",
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week,
                "game_id": game_id,
                "is_skill": True,
                "targets": 6,
                "carries": 0,
            },
        ]
    )


def _make_multi_week(n_weeks: int = 6) -> DataFrame:
    """Build multiple weeks of data for two teams."""
    frames: list[DataFrame] = []
    for week in range(1, n_weeks + 1):
        frames.append(_make_team_game(week=week, team="KC", opponent="LV"))
        frames.append(_make_team_game(week=week, team="LV", opponent="KC"))
    return pd.concat(frames, ignore_index=True)


class TestConstants:
    """Verify module-level constants."""

    def test_default_windows(self) -> None:
        assert DEFAULT_WINDOWS == [3, 6]

    def test_share_cols_count(self) -> None:
        assert len(_SHARE_COLS) == 3

    def test_share_cols_contents(self) -> None:
        assert "usage_target_share" in _SHARE_COLS
        assert "usage_carry_share" in _SHARE_COLS
        assert "usage_touch_share" in _SHARE_COLS


class TestComputeTeamTotals:
    """Verify team-level aggregation."""

    def test_totals_one_game(self) -> None:
        df = _make_team_game()
        totals = _compute_team_totals(df)
        assert len(totals) == 1
        row = totals.iloc[0]
        # QB: 0 + RB: 4 + WR: 10 + TE: 6 = 20 targets
        assert row["team_total_targets"] == 20
        # QB: 3 + RB: 20 + WR: 0 + TE: 0 = 23 carries
        assert row["team_total_carries"] == 23
        assert row["team_total_touches"] == 43

    def test_separate_teams(self) -> None:
        """Each team's totals are computed independently."""
        df = _make_multi_week(n_weeks=1)
        totals = _compute_team_totals(df)
        assert len(totals) == 2


class TestComputePerGameShares:
    """Verify per-player per-game share calculations."""

    def test_target_shares_sum_to_one(self) -> None:
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        kc = result[result["team"] == "KC"]
        assert kc["usage_target_share"].sum() == pytest.approx(1.0)

    def test_carry_shares_sum_to_one(self) -> None:
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        kc = result[result["team"] == "KC"]
        assert kc["usage_carry_share"].sum() == pytest.approx(1.0)

    def test_touch_shares_sum_to_one(self) -> None:
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        kc = result[result["team"] == "KC"]
        assert kc["usage_touch_share"].sum() == pytest.approx(1.0)

    def test_wr_target_share_value(self) -> None:
        """WR with 10 targets out of 20 total → 0.5 target share."""
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        wr = result[result["player_id"] == "WR1_KC"]
        assert wr["usage_target_share"].iloc[0] == pytest.approx(0.5)

    def test_rb_carry_share_value(self) -> None:
        """RB with 20 carries out of 23 total → ~0.87 carry share."""
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        rb = result[result["player_id"] == "RB1_KC"]
        assert rb["usage_carry_share"].iloc[0] == pytest.approx(20.0 / 23.0)

    def test_zero_division_produces_zero(self) -> None:
        """If team has 0 targets, target share = 0 (not NaN)."""
        df = _make_team_game()
        df["targets"] = 0
        result = _compute_per_game_shares(df)
        assert (result["usage_target_share"] == 0.0).all()

    def test_no_extra_rows(self) -> None:
        """Merge must not create extra rows."""
        df = _make_team_game()
        result = _compute_per_game_shares(df)
        assert len(result) == len(df)


class TestRollingShares:
    """Verify shifted rolling mean computations."""

    def test_no_lookahead_week1(self) -> None:
        """Week 1 rolling usage shares must be NaN - no prior games."""
        df = _make_multi_week(n_weeks=5)
        df = _compute_per_game_shares(df)
        result = _rolling_shares(df, windows=[3])
        week1 = result[(result["week"] == 1) & (result["player_id"] == "WR1_KC")]
        assert pd.isna(week1["usage_target_share_L3"].iloc[0])
        assert pd.isna(week1["usage_carry_share_L3"].iloc[0])
        assert pd.isna(week1["usage_touch_share_L3"].iloc[0])

    def test_week2_uses_only_week1(self) -> None:
        """Week 2 rolling share should equal exactly the week 1 share."""
        df = _make_multi_week(n_weeks=5)
        df = _compute_per_game_shares(df)
        result = _rolling_shares(df, windows=[3])

        wr = result[result["player_id"] == "WR1_KC"].sort_values("week")
        wk1_share = wr[wr["week"] == 1]["usage_target_share"].iloc[0]
        wk2_rolling = wr[wr["week"] == 2]["usage_target_share_L3"].iloc[0]
        assert wk2_rolling == pytest.approx(wk1_share)

    def test_produces_expected_columns(self) -> None:
        """Windows [3, 6] x 3 share types = 6 rolling columns."""
        df = _make_multi_week(n_weeks=5)
        df = _compute_per_game_shares(df)
        result = _rolling_shares(df, windows=[3, 6])
        expected = {
            "usage_target_share_L3",
            "usage_target_share_L6",
            "usage_carry_share_L3",
            "usage_carry_share_L6",
            "usage_touch_share_L3",
            "usage_touch_share_L6",
        }
        actual = {c for c in result.columns if c.startswith("usage_") and "_L" in c}
        assert actual == expected

    def test_season_boundary_resets(self) -> None:
        """Rolling windows should not cross season boundaries by default."""
        # Season 1: 3 weeks, Season 2: 2 weeks
        frames: list[DataFrame] = []
        for wk in range(1, 4):
            frames.append(_make_team_game(season=2023, week=wk))
        for wk in range(1, 3):
            frames.append(_make_team_game(season=2024, week=wk))
        df = pd.concat(frames, ignore_index=True)
        df = _compute_per_game_shares(df)
        result = _rolling_shares(df, windows=[3], cross_season=False)

        # Season 2024, week 1 should be NaN (season boundary reset)
        s2_wk1 = result[
            (result["season"] == 2024) & (result["week"] == 1) & (result["player_id"] == "WR1_KC")
        ]
        assert pd.isna(s2_wk1["usage_target_share_L3"].iloc[0])


class TestBuildUsageFeatures:
    """Verify the public entry point."""

    def test_raises_when_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cleaned player game logs not found"):
            build_usage_features(repo=tmp_path)

    def test_intermediate_shares_not_in_output(self, tmp_path: Path) -> None:
        """Per-game share columns should be dropped from final output."""
        # Write minimal data to disk
        data_dir = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        df = _make_multi_week(n_weeks=4)
        df.to_parquet(data_dir / "player_game_logs.parquet", index=False)

        result = build_usage_features(repo=tmp_path)
        for col in _SHARE_COLS:
            assert col not in result.columns, f"{col} should be dropped"

    def test_output_has_rolling_columns(self, tmp_path: Path) -> None:
        """Output should include the 6 rolling usage feature columns."""
        data_dir = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        df = _make_multi_week(n_weeks=4)
        df.to_parquet(data_dir / "player_game_logs.parquet", index=False)

        result = build_usage_features(repo=tmp_path)
        usage_cols = [c for c in result.columns if c.startswith("usage_")]
        assert len(usage_cols) == 6
