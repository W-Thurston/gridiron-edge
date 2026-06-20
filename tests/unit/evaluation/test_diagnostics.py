# tests/unit/evaluation/test_diagnostics.py
"""Smoke tests for gridiron_edge.evaluation.diagnostics — plot functions produce PNGs."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.diagnostics import (
    _model_key,
    plot_brier_decomposition,
    plot_calibration_curve,
    plot_confidence_distribution,
    plot_roc_curve,
)


def _make_eval_df(n: int = 200) -> pd.DataFrame:
    """Build a minimal evaluation DataFrame with the canonical archive schema.

    Uses (model_name, model_type) columns per WS2's composite key convention,
    matching what build_evaluation_df actually returns.
    """
    rng: Generator = np.random.default_rng(42)
    probs = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < probs).astype(int)
    return pd.DataFrame(
        {
            "away_win_prob": probs,
            "away_team_won": outcomes,
            "model_name": "win_prob",
            "model_type": "logistic",
            "season": "2024-2025",
            "week": rng.integers(1, 19, n),
            "game_id": [f"2024_{w:02d}_AAA_BBB" for w in rng.integers(1, 19, n)],
        }
    )


class TestModelKeyHelper:
    """Verify _model_key builds composite keys from canonical schema."""

    def test_builds_composite_key(self) -> None:
        df: DataFrame = _make_eval_df()
        assert _model_key(df) == "win_prob_logistic"

    def test_uses_first_row(self) -> None:
        """Per-archive, model_name/model_type should be constant; _model_key
        uses the first row as canonical."""
        df = _make_eval_df()
        df.iloc[1:, df.columns.get_loc("model_name")] = "different"
        # First row still drives the key
        assert _model_key(df) == "win_prob_logistic"

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"away_win_prob": [0.5], "away_team_won": [1]})
        with pytest.raises(KeyError):
            _model_key(df)


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

    def test_uses_composite_key_in_path(self, tmp_path: Path) -> None:
        """Output path should include the composite model key as the subdir."""
        eval_df: DataFrame = _make_eval_df()
        result: Path = plot_calibration_curve(eval_df, repo=tmp_path)
        assert "win_prob_logistic" in str(result)


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


class TestUnwrapEstimator:
    """Verify _unwrap_estimator handles each artifact shape."""

    def test_unwraps_pipeline_with_clf_step(self) -> None:
        """Pipeline with 'clf' step should unwrap to the clf."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from gridiron_edge.evaluation.diagnostics import _unwrap_estimator

        inner = LogisticRegression()
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", inner)])
        result = _unwrap_estimator(pipeline)
        assert result is inner

    def test_unwraps_bare_logistic(self) -> None:
        """A bare LogisticRegressionCV should be returned unchanged."""
        from sklearn.linear_model import LogisticRegressionCV

        from gridiron_edge.evaluation.diagnostics import _unwrap_estimator

        model = LogisticRegressionCV()
        result = _unwrap_estimator(model)
        assert result is model


class TestExtractImportance:
    """Verify _extract_importance handles linear and tree models."""

    def test_handles_linear_coef(self) -> None:
        from gridiron_edge.evaluation.diagnostics import _extract_importance

        class FakeLinear:
            coef_ = np.array([[0.1, 0.2, 0.3]])

        feature_names = ["f1", "f2", "f3"]
        values, kind = _extract_importance(FakeLinear(), feature_names)
        assert kind == "coefficient"
        assert values == [0.1, 0.2, 0.3]

    def test_handles_tree_importance(self) -> None:
        from gridiron_edge.evaluation.diagnostics import _extract_importance

        class FakeTree:
            feature_importances_ = np.array([0.4, 0.3, 0.3])

        feature_names = ["f1", "f2", "f3"]
        values, kind = _extract_importance(FakeTree(), feature_names)
        assert kind == "importance"
        assert values == [0.4, 0.3, 0.3]

    def test_returns_none_for_unsupported(self) -> None:
        from gridiron_edge.evaluation.diagnostics import _extract_importance

        class FakeUnsupported:
            pass

        feature_names = ["f1"]
        values, kind = _extract_importance(FakeUnsupported(), feature_names)
        assert values is None
        assert kind is None

    def test_handles_length_mismatch(self) -> None:
        from gridiron_edge.evaluation.diagnostics import _extract_importance

        class FakeLinear:
            coef_ = np.array([[0.1, 0.2]])

        feature_names = ["f1", "f2", "f3"]
        values, kind = _extract_importance(FakeLinear(), feature_names)
        assert values is None
        assert kind is None
