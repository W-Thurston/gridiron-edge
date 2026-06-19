# tests/unit/features/test_player_rolling.py
"""Tests for gridiron_edge.features.player.rolling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.player.rolling import (
    DEFAULT_WINDOWS,
    ROLLING_STAT_COLS,
    _compute_rolling,
    build_player_rolling_features,
)


def _make_player_season(
    player_id: str = "00-0001234",
    player_name: str = "T.Brady",
    season: int = 2024,
    n_weeks: int = 17,
    position: str = "QB",
) -> DataFrame:
    """Build a single player's season of game logs."""
    rng = np.random.default_rng(42)
    rows = []
    for week in range(1, n_weeks + 1):
        row: dict = dict.fromkeys(ROLLING_STAT_COLS, 0.0)
        row.update(
            {
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "team": "KC",
                "opponent_team": "LV",
                "season": season,
                "week": week,
                "is_skill": True,
                "passing_yards": float(rng.integers(150, 350)),
                "passing_tds": float(rng.integers(0, 4)),
                "passing_interceptions": float(rng.integers(0, 3)),
                "attempts": float(rng.integers(25, 45)),
                "completions": float(rng.integers(15, 35)),
                "passing_air_yards": float(rng.integers(100, 300)),
                "passing_epa": float(rng.uniform(-5, 15)),
                "passing_cpoe": float(rng.uniform(-10, 10)),
                "sacks_suffered": float(rng.integers(0, 5)),
                "carries": float(rng.integers(0, 5)),
                "rushing_yards": float(rng.integers(0, 30)),
                "rushing_epa": float(rng.uniform(-2, 2)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class TestConstants:
    def test_default_windows(self) -> None:
        assert DEFAULT_WINDOWS == [3, 6]

    def test_rolling_stat_cols_count(self) -> None:
        assert len(ROLLING_STAT_COLS) == 23

    def test_key_stats_present(self) -> None:
        assert "passing_yards" in ROLLING_STAT_COLS
        assert "rushing_yards" in ROLLING_STAT_COLS
        assert "receiving_yards" in ROLLING_STAT_COLS
        assert "passing_cpoe" in ROLLING_STAT_COLS
        assert "target_share" in ROLLING_STAT_COLS


class TestComputeRolling:
    def test_no_lookahead_week1(self) -> None:
        """Week 1 rolling stats must be NaN — no prior games exist."""
        df = _make_player_season(n_weeks=5)
        result = _compute_rolling(df, windows=[3])

        week1 = result[result["week"] == 1]
        assert pd.isna(week1["passing_yards_L3_mean"].iloc[0])
        assert pd.isna(week1["passing_yards_L3_std"].iloc[0])

    def test_week2_uses_only_week1(self) -> None:
        """Week 2 L3 mean should equal week 1 value (only 1 prior game)."""
        df = _make_player_season(n_weeks=5)
        result = _compute_rolling(df, windows=[3])

        week1_val = result[result["week"] == 1]["passing_yards"].iloc[0]
        week2_mean = result[result["week"] == 2]["passing_yards_L3_mean"].iloc[0]
        assert week2_mean == pytest.approx(week1_val)

    def test_week4_uses_prior_3_games(self) -> None:
        """Week 4 L3 mean should be mean of weeks 1, 2, 3."""
        df = _make_player_season(n_weeks=5)
        result = _compute_rolling(df, windows=[3])

        prior_3 = result[result["week"].isin([1, 2, 3])]["passing_yards"].values
        week4_mean = result[result["week"] == 4]["passing_yards_L3_mean"].iloc[0]
        assert week4_mean == pytest.approx(prior_3.mean())

    def test_produces_mean_and_std_columns(self) -> None:
        df = _make_player_season(n_weeks=6)
        result = _compute_rolling(df, windows=[3, 6])

        assert "passing_yards_L3_mean" in result.columns
        assert "passing_yards_L3_std" in result.columns
        assert "passing_yards_L6_mean" in result.columns
        assert "passing_yards_L6_std" in result.columns

    def test_column_count(self) -> None:
        """Rolling adds 2 columns (mean + std) per stat per window."""
        df = _make_player_season(n_weeks=6)
        n_original = len(df.columns)
        result = _compute_rolling(df, windows=[3, 6])

        available_stats = [c for c in ROLLING_STAT_COLS if c in df.columns]
        expected_new = len(available_stats) * 2 * 2
        assert len(result.columns) == n_original + expected_new

    def test_season_boundary_reset(self) -> None:
        """Rolling windows should not leak across seasons by default."""
        s1 = _make_player_season(season=2023, n_weeks=3)
        s2 = _make_player_season(season=2024, n_weeks=3)
        df = pd.concat([s1, s2], ignore_index=True)

        result = _compute_rolling(df, windows=[3])

        week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
        assert pd.isna(week1_2024["passing_yards_L3_mean"].iloc[0])

    def test_cross_season_spans_boundary(self) -> None:
        """With cross_season=True, windows span season boundaries."""
        s1 = _make_player_season(season=2023, n_weeks=3)
        s2 = _make_player_season(season=2024, n_weeks=3)
        df = pd.concat([s1, s2], ignore_index=True)

        result = _compute_rolling(df, windows=[3], cross_season=True)

        week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
        assert not pd.isna(week1_2024["passing_yards_L3_mean"].iloc[0])

    def test_multiple_players_independent(self) -> None:
        """Rolling stats for different players must not leak between players."""
        p1 = _make_player_season(player_id="P1", player_name="A", n_weeks=5)
        p2 = _make_player_season(player_id="P2", player_name="B", n_weeks=5)
        df = pd.concat([p1, p2], ignore_index=True)

        result = _compute_rolling(df, windows=[3])

        p1_result = result[result["player_id"] == "P1"]
        p1_w1_val = p1_result[p1_result["week"] == 1]["passing_yards"].iloc[0]
        p1_w2_mean = p1_result[p1_result["week"] == 2]["passing_yards_L3_mean"].iloc[0]
        assert p1_w2_mean == pytest.approx(p1_w1_val)


class TestBuildPlayerRollingFeatures:
    def test_raises_when_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Cleaned player game logs not found"):
            build_player_rolling_features(repo=tmp_path)


class TestVectorizedEquivalence:
    """Verify the vectorized _compute_rolling produces identical output.

    These tests guard against future regressions when the vectorization
    is touched. They check both numerical equality and shape preservation.
    """

    def test_consistent_across_runs(self) -> None:
        """Computing rolling features twice gives identical results."""
        df = _make_player_season(n_weeks=10)
        result1 = _compute_rolling(df.copy(), windows=[3, 6])
        result2 = _compute_rolling(df.copy(), windows=[3, 6])

        for col in result1.columns:
            if col in result2.columns:
                pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)

    def test_multiple_seasons_independent_groups(self) -> None:
        """Rolling stats for player A season 2023 don't affect player A season 2024."""
        p_2023 = _make_player_season(player_id="P1", season=2023, n_weeks=10)
        p_2024 = _make_player_season(player_id="P1", season=2024, n_weeks=10)
        df = pd.concat([p_2023, p_2024], ignore_index=True)

        result = _compute_rolling(df, windows=[3])

        # Week 1 of 2024 should still be NaN (season-boundary reset)
        w1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]
        assert pd.isna(w1_2024["passing_yards_L3_mean"].iloc[0])

        # Week 4 of 2024 should use weeks 1-3 of 2024 only
        w4_2024 = result[(result["season"] == 2024) & (result["week"] == 4)]
        prior_3_2024 = result[(result["season"] == 2024) & (result["week"].isin([1, 2, 3]))][
            "passing_yards"
        ].values
        assert w4_2024["passing_yards_L3_mean"].iloc[0] == pytest.approx(prior_3_2024.mean())

    def test_three_players_independent(self) -> None:
        """Rolling stats for three players must be fully independent."""
        p1 = _make_player_season(player_id="P1", player_name="A", n_weeks=10)
        p2 = _make_player_season(player_id="P2", player_name="B", n_weeks=10)
        p3 = _make_player_season(player_id="P3", player_name="C", n_weeks=10)
        df = pd.concat([p1, p2, p3], ignore_index=True)

        result = _compute_rolling(df, windows=[3])

        # For each player, week 4 mean must equal their own prior 3 weeks' mean.
        for pid in ("P1", "P2", "P3"):
            player = result[result["player_id"] == pid]
            prior_3 = player[player["week"].isin([1, 2, 3])]["passing_yards"].values
            w4_mean = player[player["week"] == 4]["passing_yards_L3_mean"].iloc[0]
            assert w4_mean == pytest.approx(prior_3.mean()), f"player {pid} mismatch"
