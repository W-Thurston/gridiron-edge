# tests/unit/features/test_builder.py
"""Tests for gridiron_edge.features.player.builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.player._columns import (
    GAME_CONTEXT_COLS,
    PROP_FEATURE_COLS,
)
from gridiron_edge.features.player.builder import (
    build_prop_features,
)


def _make_player_logs(n_weeks: int = 10) -> DataFrame:
    """Build multi-week player game logs with enough data for rolling windows."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    teams = [("KC", "LV"), ("LV", "KC")]
    positions = {
        "KC": [
            ("QB1_KC", "P.Mahomes", "QB"),
            ("RB1_KC", "I.Pacheco", "RB"),
            ("WR1_KC", "R.Rice", "WR"),
            ("TE1_KC", "T.Kelce", "TE"),
        ],
        "LV": [
            ("QB1_LV", "A.OConnell", "QB"),
            ("RB1_LV", "Z.White", "RB"),
            ("WR1_LV", "D.Adams", "WR"),
            ("TE1_LV", "M.Moreau", "TE"),
        ],
    }

    for week in range(1, n_weeks + 1):
        for team, opp in teams:
            for pid, name, pos in positions[team]:
                row: dict = {
                    "player_id": pid,
                    "player_name": name,
                    "position": pos,
                    "position_group": pos,
                    "team": team,
                    "opponent_team": opp,
                    "season": 2024,
                    "season_type": "REG",
                    "week": week,
                    "game_id": f"2024_{week:02d}_LV_KC",
                    "is_skill": True,
                    "player_display_name": name,
                    # Passing stats
                    "completions": float(rng.integers(0, 30)),
                    "attempts": float(rng.integers(0, 40)),
                    "passing_yards": float(rng.integers(0, 350)),
                    "passing_tds": float(rng.integers(0, 4)),
                    "passing_interceptions": float(rng.integers(0, 3)),
                    "sacks_suffered": float(rng.integers(0, 5)),
                    "passing_air_yards": float(rng.integers(0, 300)),
                    "passing_yards_after_catch": float(rng.integers(0, 200)),
                    "passing_epa": float(rng.uniform(-5, 15)),
                    "passing_first_downs": float(rng.integers(0, 20)),
                    "passing_cpoe": float(rng.uniform(-10, 10)),
                    # Rushing stats
                    "carries": float(rng.integers(0, 20)),
                    "rushing_yards": float(rng.integers(0, 100)),
                    "rushing_tds": float(rng.integers(0, 2)),
                    "rushing_first_downs": float(rng.integers(0, 8)),
                    "rushing_fumbles": float(rng.integers(0, 2)),
                    "rushing_fumbles_lost": float(rng.integers(0, 1)),
                    "rushing_epa": float(rng.uniform(-3, 5)),
                    # Receiving stats
                    "receptions": float(rng.integers(0, 10)),
                    "targets": float(rng.integers(0, 12)),
                    "receiving_yards": float(rng.integers(0, 150)),
                    "receiving_tds": float(rng.integers(0, 2)),
                    "receiving_air_yards": float(rng.integers(0, 100)),
                    "receiving_yards_after_catch": float(rng.integers(0, 80)),
                    "receiving_first_downs": float(rng.integers(0, 8)),
                    "receiving_fumbles_lost": float(rng.integers(0, 1)),
                    "receiving_epa": float(rng.uniform(-2, 5)),
                    "target_share": float(rng.uniform(0, 0.3)),
                    "air_yards_share": float(rng.uniform(0, 0.3)),
                    "wopr": float(rng.uniform(0, 0.5)),
                    "pacr": float(rng.uniform(0, 2)),
                    "racr": float(rng.uniform(0, 2)),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def _make_games(n_weeks: int = 10) -> DataFrame:
    """Build matching games data."""
    rows: list[dict] = []
    base = pd.Timestamp("2024-09-08")
    for week in range(1, n_weeks + 1):
        rows.append(
            {
                "GAME_ID": f"2024_{week:02d}_LV_KC",
                "VEGAS_LINE": -3.5,
                "OVER_UNDER": 47.0,
                "FAVORITED": "Kansas City Chiefs",
                "ROOF": "outdoors",
                "GAME_DATE": (base + pd.Timedelta(weeks=week - 1)).strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def _setup_data(tmp_path: Path, n_weeks: int = 10) -> None:
    """Write player logs and games data to tmp_path."""
    data_dir = tmp_path / "data" / "cleaned"
    data_dir.mkdir(parents=True)
    _make_player_logs(n_weeks).to_parquet(data_dir / "player_game_logs.parquet")
    _make_games(n_weeks).to_csv(data_dir / "NFL_wk_by_wk_cleaned.csv", index=False)


class TestPropFeatureCols:
    """Verify the programmatic feature column list."""

    def test_count_is_positive(self) -> None:
        assert len(PROP_FEATURE_COLS) > 100

    def test_no_duplicates(self) -> None:
        assert len(PROP_FEATURE_COLS) == len(set(PROP_FEATURE_COLS))

    def test_contains_rolling_features(self) -> None:
        rolling = [c for c in PROP_FEATURE_COLS if "_L3_mean" in c or "_L6_mean" in c]
        assert len(rolling) > 0

    def test_contains_matchup_features(self) -> None:
        matchup = [c for c in PROP_FEATURE_COLS if "opp_" in c]
        assert len(matchup) > 0

    def test_contains_usage_features(self) -> None:
        usage = [c for c in PROP_FEATURE_COLS if "usage_" in c]
        assert len(usage) == 6

    def test_contains_game_context_features(self) -> None:
        for col in GAME_CONTEXT_COLS:
            assert col in PROP_FEATURE_COLS


class TestBuildPropFeatures:
    """Verify the unified builder end-to-end."""

    def test_raises_when_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="player game logs"):
            build_prop_features(position_filter=["QB"], repo=tmp_path)

    def test_raises_when_no_games_data(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        _make_player_logs().to_parquet(data_dir / "player_game_logs.parquet")
        with pytest.raises(FileNotFoundError, match="games data"):
            build_prop_features(position_filter=["QB"], repo=tmp_path)

    def test_qb_filter(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        assert (result["position"] == "QB").all()
        assert result["player_id"].nunique() == 2  # QB1_KC, QB1_LV

    def test_rb_filter(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["RB"], repo=tmp_path)
        assert (result["position"] == "RB").all()

    def test_multi_position_filter(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["WR", "TE"], repo=tmp_path)
        assert set(result["position"].unique()) <= {"WR", "TE"}

    def test_has_rolling_features(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        rolling = [c for c in result.columns if "_L3_mean" in c or "_L6_mean" in c]
        assert len(rolling) > 0

    def test_has_matchup_features(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        matchup = [c for c in result.columns if "opp_" in c]
        assert len(matchup) > 0

    def test_has_usage_features(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        usage = [c for c in result.columns if "usage_" in c]
        assert len(usage) == 6

    def test_has_game_context_features(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        for col in GAME_CONTEXT_COLS:
            assert col in result.columns

    def test_nan_handling_deferred_to_trainer(self, tmp_path: Path) -> None:
        """Builder should NOT drop NaN - trainer handles it with position context."""
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        # Builder returns all rows; some NaN is expected in cross-position features
        assert len(result) > 0

    def test_no_raw_game_columns(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        for col in ["VEGAS_LINE", "OVER_UNDER", "FAVORITED", "ROOF", "GAME_DATE"]:
            assert col not in result.columns

    def test_preserves_target_columns(self, tmp_path: Path) -> None:
        """Target stat columns must survive for training."""
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        assert "passing_yards" in result.columns
        assert "rushing_yards" in result.columns

    def test_preserves_identity_columns(self, tmp_path: Path) -> None:
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        for col in ["player_id", "player_name", "game_id", "season", "week"]:
            assert col in result.columns

    def test_no_row_duplication(self, tmp_path: Path) -> None:
        """No duplicate (player_id, game_id) rows in output."""
        _setup_data(tmp_path)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        dupes = result.duplicated(subset=["player_id", "game_id"], keep=False).sum()
        assert dupes == 0

    def test_row_count_reasonable(self, tmp_path: Path) -> None:
        """Should have fewer rows than input (NaN drop + position filter)."""
        _setup_data(tmp_path, n_weeks=10)
        result = build_prop_features(position_filter=["QB"], repo=tmp_path)
        # 2 QBs x 10 weeks = 20, minus early weeks lost to rolling NaN
        assert 0 < len(result) <= 20
