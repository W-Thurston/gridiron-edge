# tests/unit/evaluation/test_diagnostics.py
"""Smoke tests for gridiron_edge.evaluation.diagnostics — plot functions produce PNGs."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.evaluation.diagnostics import (
    plot_brier_decomposition,
    plot_calibration_curve,
    plot_confidence_distribution,
    plot_roc_curve,
)


def _make_eval_df(n: int = 200) -> pd.DataFrame:
    """Build a minimal evaluation DataFrame with realistic-ish predictions."""
    rng: Generator = np.random.default_rng(42)
    probs = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < probs).astype(int)
    return pd.DataFrame(
        {
            "away_win_prob": probs,
            "away_team_won": outcomes,
            "model_version": "test_v1",
            "season": "2024-2025",
            "week": rng.integers(1, 19, n),
        }
    )


class TestCalibrationCurve:
    def test_returns_path(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_calibration_curve(eval_df, repo=tmp_path)
        assert isinstance(result, Path)

    def test_creates_png(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_calibration_curve(eval_df, repo=tmp_path)
        assert result.is_file()
        assert result.suffix == ".png"


class TestConfidenceDistribution:
    def test_returns_path(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_confidence_distribution(eval_df, repo=tmp_path)
        assert isinstance(result, Path)

    def test_creates_png(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_confidence_distribution(eval_df, repo=tmp_path)
        assert result.is_file()


class TestRocCurve:
    def test_returns_path(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_roc_curve(eval_df, repo=tmp_path)
        assert isinstance(result, Path)

    def test_creates_png(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_roc_curve(eval_df, repo=tmp_path)
        assert result.is_file()


class TestBrierDecomposition:
    def test_returns_path(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_brier_decomposition(eval_df, repo=tmp_path)
        assert isinstance(result, Path)

    def test_creates_png(self, tmp_path: Path) -> None:
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_brier_decomposition(eval_df, repo=tmp_path)
        assert result.is_file()
