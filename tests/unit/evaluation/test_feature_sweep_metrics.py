# tests/unit/evaluation/test_feature_sweep_metrics.py

"""Tests for sweep metrics and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron_edge.evaluation.feature_sweep_metrics import (
    absolute_losses,
    brier_losses,
    holm_adjust,
    paired_bootstrap_improvement,
    season_improvements,
    total_metrics,
    win_metrics,
)


@pytest.mark.unit
def test_win_metrics_known_values() -> None:
    p = np.array([0.8, 0.2, 0.6, 0.4])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    m = win_metrics(p, y)
    assert m.n == 4
    assert m.brier == pytest.approx((0.04 + 0.04 + 0.16 + 0.16) / 4)
    assert m.auc == pytest.approx(1.0)
    assert m.log_loss == pytest.approx(
        -np.mean(np.log([0.8, 0.8, 0.6, 0.6])),
    )


@pytest.mark.unit
def test_total_metrics_known_values() -> None:
    pred = np.array([50.0, 40.0, 45.0])
    y = np.array([47.0, 44.0, 45.0])
    m = total_metrics(pred, y)
    assert m.mae == pytest.approx(7 / 3)
    assert m.bias == pytest.approx((3 - 4 + 0) / 3)
    assert m.rmse == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert m.median_ae == pytest.approx(3.0)


@pytest.mark.unit
def test_per_game_losses() -> None:
    assert brier_losses(np.array([0.7]), np.array([1.0]))[0] == pytest.approx(0.09)
    assert absolute_losses(np.array([48.0]), np.array([50.0]))[0] == pytest.approx(2.0)


@pytest.mark.unit
def test_bootstrap_detects_real_improvement_and_no_improvement() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(0.1, 0.4, size=800)
    better = base - 0.01 + rng.normal(0, 0.002, size=800)
    win = paired_bootstrap_improvement(base, better)
    assert win.improvement == pytest.approx(0.01, abs=0.001)
    assert win.lower > 0.0
    assert win.p_no_improvement < 0.01

    noise = base + rng.normal(0, 0.05, size=800)
    flat = paired_bootstrap_improvement(base, noise)
    assert flat.lower < 0.0 < flat.upper


@pytest.mark.unit
def test_bootstrap_is_deterministic_and_paired() -> None:
    a = np.linspace(0.1, 0.3, 200)
    b = a - 0.005
    first = paired_bootstrap_improvement(a, b)
    second = paired_bootstrap_improvement(a, b)
    assert first == second
    # A constant per-game gain has zero pairing variance: interval collapses.
    assert first.lower == pytest.approx(0.005)
    assert first.upper == pytest.approx(0.005)


@pytest.mark.unit
def test_bootstrap_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_improvement(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_improvement(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="NaN"):
        paired_bootstrap_improvement(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))


@pytest.mark.unit
def test_season_improvements_group_and_sort() -> None:
    prev = np.array([0.3, 0.3, 0.2, 0.2])
    new = np.array([0.2, 0.2, 0.25, 0.25])
    out = season_improvements(
        prev, new, np.array(["2024-2025", "2024-2025", "2023-2024", "2023-2024"])
    )
    assert list(out) == ["2023-2024", "2024-2025"]
    assert out["2024-2025"] == pytest.approx(0.1)
    assert out["2023-2024"] == pytest.approx(-0.05)
    with pytest.raises(ValueError, match="align"):
        season_improvements(prev, new, np.array(["a"]))


@pytest.mark.unit
def test_holm_adjust_matches_textbook_example() -> None:
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert holm_adjust({"x": 0.9})["x"] == pytest.approx(0.9)
    assert holm_adjust({"a": 0.6, "b": 0.7})["a"] == pytest.approx(1.0)
