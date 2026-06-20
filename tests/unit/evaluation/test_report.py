"""Tests for gridiron_edge.evaluation.report heuristics."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.evaluation.report import (
    DriftFlag,
    EarlySeasonMissFlag,
    HighConfidenceFlag,
    OverconfidenceMissFlag,
    find_early_season_miss_pattern,
    find_high_confidence_warning,
    find_overconfidence_miss_pattern,
    find_season_drift_warning,
)

# ---------------------------------------------------------------------------
# find_high_confidence_warning
# ---------------------------------------------------------------------------


class TestHighConfidenceWarning:
    def test_returns_none_when_no_high_confidence_tier(self) -> None:
        df_tiers = pd.DataFrame(
            {
                "confidence_tier": ["low", "med"],
                "predicted_avg": [0.55, 0.65],
                "actual_win_rate": [0.55, 0.65],
                "calibration_gap": [0.0, 0.0],
            }
        )
        assert find_high_confidence_warning(df_tiers) is None

    def test_returns_none_when_gap_below_threshold(self) -> None:
        df_tiers = pd.DataFrame(
            {
                "confidence_tier": ["high"],
                "predicted_avg": [0.80],
                "actual_win_rate": [0.79],
                "calibration_gap": [0.01],
            }
        )
        assert find_high_confidence_warning(df_tiers) is None

    def test_returns_overconfident_flag(self) -> None:
        df_tiers = pd.DataFrame(
            {
                "confidence_tier": ["high"],
                "predicted_avg": [0.80],
                "actual_win_rate": [0.70],
                "calibration_gap": [0.10],
            }
        )
        flag = find_high_confidence_warning(df_tiers)
        assert isinstance(flag, HighConfidenceFlag)
        assert flag.direction == "overconfident"
        assert flag.confidence_tier == "high"

    def test_returns_underconfident_flag(self) -> None:
        df_tiers = pd.DataFrame(
            {
                "confidence_tier": ["high"],
                "predicted_avg": [0.80],
                "actual_win_rate": [0.90],
                "calibration_gap": [-0.10],
            }
        )
        flag = find_high_confidence_warning(df_tiers)
        assert isinstance(flag, HighConfidenceFlag)
        assert flag.direction == "underconfident"

    def test_picks_worst_gap_among_high_confidence_tiers(self) -> None:
        df_tiers = pd.DataFrame(
            {
                "confidence_tier": ["high_a", "high_b"],
                "predicted_avg": [0.80, 0.85],
                "actual_win_rate": [0.74, 0.65],
                "calibration_gap": [0.06, 0.20],
            }
        )
        flag = find_high_confidence_warning(df_tiers)
        assert flag is not None
        assert flag.confidence_tier == "high_b"


# ---------------------------------------------------------------------------
# find_season_drift_warning
# ---------------------------------------------------------------------------


class TestSeasonDriftWarning:
    def test_returns_none_when_no_drift_marker(self) -> None:
        df_seasons = pd.DataFrame(
            {
                "season": ["2023-2024", "2024-2025"],
                "trend": ["", ""],
                "delta_vs_mean": [0.001, -0.002],
            }
        )
        assert find_season_drift_warning(df_seasons) is None

    def test_returns_worst_warn_season(self) -> None:
        df_seasons = pd.DataFrame(
            {
                "season": ["2023-2024", "2024-2025", "2025-2026"],
                "trend": ["⚠", "", "⚠"],
                "delta_vs_mean": [0.030, 0.001, 0.010],
            }
        )
        flag = find_season_drift_warning(df_seasons)
        assert isinstance(flag, DriftFlag)
        assert flag.season == "2023-2024"
        assert flag.delta_vs_mean == 0.030


# ---------------------------------------------------------------------------
# find_early_season_miss_pattern
# ---------------------------------------------------------------------------


class TestEarlySeasonMissPattern:
    def test_returns_none_when_few_early_misses(self) -> None:
        df_misses = pd.DataFrame(
            {
                "week": [1, 5, 10],
            }
        )
        assert find_early_season_miss_pattern(df_misses, top_misses=10) is None

    def test_returns_flag_when_threshold_reached(self) -> None:
        df_misses = pd.DataFrame(
            {
                "week": [1, 2, 3, 8],
            }
        )
        flag = find_early_season_miss_pattern(df_misses, top_misses=10)
        assert isinstance(flag, EarlySeasonMissFlag)
        assert flag.n_early == 3
        assert flag.top_misses == 10


# ---------------------------------------------------------------------------
# find_overconfidence_miss_pattern
# ---------------------------------------------------------------------------


class TestOverconfidenceMissPattern:
    def test_returns_none_when_few_loss_misses(self) -> None:
        df_misses = pd.DataFrame(
            {
                "actual_result": ["LOSS", "WIN", "WIN", "WIN"],
            }
        )
        # top_misses=10 -> half = 5 -> 1 loss is below threshold.
        assert find_overconfidence_miss_pattern(df_misses, top_misses=10) is None

    def test_returns_flag_when_threshold_reached(self) -> None:
        df_misses = pd.DataFrame(
            {
                "actual_result": ["LOSS", "LOSS", "LOSS", "WIN"],
            }
        )
        # top_misses=4 -> half = 2 -> 3 losses meets threshold.
        flag = find_overconfidence_miss_pattern(df_misses, top_misses=4)
        assert isinstance(flag, OverconfidenceMissFlag)
        assert flag.n_losses == 3
        assert flag.top_misses == 4
