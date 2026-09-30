# tests/unit/evaluation/test_feature_sweep_gates.py

"""Tests for the sweep adopt/reject bars and cross-family attribution."""

from __future__ import annotations

import pytest

from gridiron_edge.evaluation.feature_sweep_gates import (
    DEFAULT_TOTAL_BAR,
    DEFAULT_WIN_BAR,
    FamilyVerdict,
    decide_batch,
    evaluate_total_family,
    evaluate_win_family,
)
from gridiron_edge.evaluation.feature_sweep_metrics import (
    BootstrapDelta,
    TotalMetrics,
    WinMetrics,
)


def _delta(improvement: float, lower: float, upper: float) -> BootstrapDelta:
    return BootstrapDelta(improvement, lower, upper, 0.02, 800, 0.9)


def _win(brier: float = 0.22, ece: float = 0.03, ll: float = 0.63) -> WinMetrics:
    return WinMetrics(800, brier, ll, ece, 0.69, 1.0, 0.0)


def _total(mae: float = 10.4, rmse: float = 13.3, bias: float = 0.2) -> TotalMetrics:
    return TotalMetrics(800, mae, rmse, bias, 8.9)


GOOD_SEASONS = {"a": 0.003, "b": 0.002, "c": -0.001}


@pytest.mark.unit
def test_bars_are_pinned_to_the_locked_thresholds() -> None:
    assert DEFAULT_WIN_BAR.min_brier_improvement == 0.002
    assert DEFAULT_WIN_BAR.max_ece_worsening == 0.01
    assert DEFAULT_WIN_BAR.min_seasons_improved == 2
    assert DEFAULT_WIN_BAR.harm_brier == 0.001
    assert DEFAULT_TOTAL_BAR.min_mae_improvement == 0.09
    assert DEFAULT_TOTAL_BAR.max_bias_shift == 0.25
    assert DEFAULT_TOTAL_BAR.min_seasons_improved == 2
    assert DEFAULT_TOTAL_BAR.harm_mae == 0.05


@pytest.mark.unit
def test_win_passes_when_every_criterion_holds() -> None:
    v = evaluate_win_family(
        "win/logistic", _win(), _win(brier=0.217), _delta(0.003, 0.0005, 0.0055), GOOD_SEASONS
    )
    assert v.passed and not v.harmed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "failed"),
    [
        ({"delta": _delta(0.0015, 0.0002, 0.003)}, "brier_improvement"),
        ({"delta": _delta(0.003, -0.0001, 0.006)}, "bootstrap_lower_above_zero"),
        ({"seasons": {"a": 0.004, "b": -0.001, "c": -0.001}}, "seasons_improved"),
        ({"new": _win(brier=0.217, ece=0.0401)}, "ece_not_worse"),
        ({"new": _win(brier=0.217, ll=0.6301)}, "log_loss_not_worse"),
        ({"noise_floor": 0.004}, "exceeds_noise_floor"),
    ],
)
def test_win_each_criterion_can_fail(kwargs: dict, failed: str) -> None:
    v = evaluate_win_family(
        "win/x",
        _win(),
        kwargs.get("new", _win(brier=0.217)),
        kwargs.get("delta", _delta(0.003, 0.0005, 0.0055)),
        kwargs.get("seasons", GOOD_SEASONS),
        noise_floor=kwargs.get("noise_floor"),
    )
    assert not v.passed
    assert v.criteria[failed] is False
    assert sum(1 for ok in v.criteria.values() if not ok) == 1


@pytest.mark.unit
def test_win_harm_needs_whole_interval_below_threshold() -> None:
    worse = evaluate_win_family(
        "w", _win(), _win(brier=0.225), _delta(-0.005, -0.008, -0.0015), GOOD_SEASONS
    )
    assert worse.harmed
    ambiguous = evaluate_win_family(
        "w", _win(), _win(brier=0.222), _delta(-0.002, -0.005, 0.0005), GOOD_SEASONS
    )
    assert not ambiguous.harmed


@pytest.mark.unit
def test_total_passes_and_fails_per_criterion() -> None:
    good = evaluate_total_family(
        "total/rf", _total(), _total(mae=10.3, rmse=13.2), _delta(0.10, 0.02, 0.18), GOOD_SEASONS
    )
    assert good.passed
    small = evaluate_total_family(
        "t", _total(), _total(mae=10.32), _delta(0.08, 0.02, 0.14), GOOD_SEASONS
    )
    assert small.criteria["mae_improvement"] is False
    rmse = evaluate_total_family(
        "t", _total(), _total(mae=10.3, rmse=13.31), _delta(0.10, 0.02, 0.18), GOOD_SEASONS
    )
    assert rmse.criteria["rmse_not_worse"] is False
    bias = evaluate_total_family(
        "t", _total(), _total(mae=10.3, bias=0.5), _delta(0.10, 0.02, 0.18), GOOD_SEASONS
    )
    assert bias.criteria["bias_shift_within_limit"] is False
    harm = evaluate_total_family(
        "t", _total(), _total(mae=10.6), _delta(-0.2, -0.3, -0.06), GOOD_SEASONS
    )
    assert harm.harmed


def _v(family: str, passed: bool, harmed: bool = False) -> FamilyVerdict:
    return FamilyVerdict(family, {}, passed, harmed, 0.0, 0.5)


@pytest.mark.unit
def test_decide_batch_attribution() -> None:
    assert (
        decide_batch("b", [_v("a", False), _v("c", False)], live_complete=True).status == "reject"
    )
    assert decide_batch("b", [_v("a", True), _v("c", False)], live_complete=True).status == "adopt"
    harmed = decide_batch("b", [_v("a", True), _v("c", False, True)], live_complete=True)
    assert harmed.status == "reject" and harmed.harmed == ("c",)
    fix = decide_batch("b", [_v("a", True)], live_complete=False)
    assert fix.status == "adopt_only_if_fixed"
    # A failing batch is rejected regardless of first-week completeness.
    assert decide_batch("b", [_v("a", False)], live_complete=False).status == "reject"
    with pytest.raises(ValueError, match="at least one"):
        decide_batch("b", [], live_complete=True)
