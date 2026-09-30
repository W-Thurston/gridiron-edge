# tests/unit/evaluation/test_feature_sweep_diagnostics.py

"""Tests for per-column sweep diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.evaluation.feature_sweep_diagnostics import (
    collinearity_report,
    leakage_smell_test,
    population_stability,
    suspect_correlations,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(11)


@pytest.mark.unit
def test_leakage_smell_flags_target_copy_but_not_noise() -> None:
    rng = _rng()
    y = pd.Series((rng.uniform(size=600) < 0.5).astype(float))
    x = pd.DataFrame(
        {
            "noise": rng.normal(size=600),
            "leak": y + rng.normal(0, 0.05, size=600),
            "weak": 0.2 * y + rng.normal(size=600),
            "const": np.ones(600),
        }
    )
    flags = leakage_smell_test(x, y, list(x.columns), binary_target=True)
    assert [f.column for f in flags] == ["leak"]
    assert flags[0].auc > 0.95


@pytest.mark.unit
def test_leakage_smell_continuous_target_uses_correlation() -> None:
    rng = _rng()
    y = pd.Series(rng.normal(size=400))
    x = pd.DataFrame({"copy": y * 2 + rng.normal(0, 0.1, 400), "noise": rng.normal(size=400)})
    flags = leakage_smell_test(x, y, ["copy", "noise"], binary_target=False)
    assert [f.column for f in flags] == ["copy"]
    assert np.isnan(flags[0].auc)


@pytest.mark.unit
def test_population_stability_detects_drift_and_is_near_zero_without() -> None:
    rng = _rng()
    train = pd.Series(rng.normal(size=3000))
    same = pd.Series(rng.normal(size=1000))
    shifted = pd.Series(rng.normal(1.5, 1.0, size=1000))
    assert population_stability(train, same) < 0.05
    assert population_stability(train, shifted) > 0.25
    assert population_stability(pd.Series([1.0] * 10), pd.Series([1.0] * 5)) == 0.0
    assert population_stability(pd.Series([1.0] * 10), pd.Series([2.0] * 5)) == float("inf")
    assert np.isnan(population_stability(train, pd.Series([np.nan])))


@pytest.mark.unit
def test_collinearity_report_finds_the_redundant_column() -> None:
    rng = _rng()
    a = rng.normal(size=500)
    x = pd.DataFrame(
        {
            "a": a,
            "a_copy": a + rng.normal(0, 0.01, 500),
            "b": rng.normal(size=500),
            "c": np.ones(500),
        }
    )
    report = collinearity_report(x)
    assert "c" not in report.vif
    top = list(report.vif)[:2]
    assert set(top) == {"a", "a_copy"}
    assert report.vif["a"] > 100
    assert report.vif["b"] < 2
    assert report.condition_number > 20


@pytest.mark.unit
def test_suspect_correlations_skip_missing_columns() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0], "z": [3.0, 1.0, 2.0]})
    out = suspect_correlations(frame, {"x": ["y", "missing"], "absent": ["y"]})
    assert list(out) == ["x"]
    assert out["x"] == {"y": pytest.approx(1.0)}
