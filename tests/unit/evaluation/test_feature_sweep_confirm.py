# tests/unit/evaluation/test_feature_sweep_confirm.py

"""Tests for the final-composition confirmation stage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from gridiron_edge.evaluation import feature_sweep_confirm as mod
from gridiron_edge.evaluation.feature_sweep_confirm import (
    MIN_POSITIVE_FAMILIES,
    ConfirmationVerdict,
    FamilyConfirmation,
    build_confirmation_report,
    judge_confirmation,
    run_confirmation,
)
from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_metrics import BootstrapDelta
from gridiron_edge.evaluation.feature_sweep_run import make_feature_provider, run_family
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.evaluation.feature_sweep_wide import WideComparison
from gridiron_edge.models.game_prediction.base import GameModelType

from .test_feature_sweep_run import _rows, _tiny_budgets  # noqa: F401


def _cmp(improvement: float, lower: float, upper: float) -> WideComparison:
    delta = BootstrapDelta(improvement, lower, upper, 0.1, 800, 0.9)
    return WideComparison("final", "baseline", 800, delta, {}, 0.001, lower > -0.001)


def _fam(label: str, target: str, wide: WideComparison, hold: WideComparison) -> FamilyConfirmation:
    return FamilyConfirmation(label, target, 152, 177, wide, hold, None)


@pytest.mark.unit
def test_criterion_confirms_when_positive_and_unharmed() -> None:
    results = {
        f"win/f{i}": _fam(
            f"win/f{i}", "win", _cmp(0.002, -0.001, 0.005), _cmp(0.001, -0.004, 0.005)
        )
        for i in range(MIN_POSITIVE_FAMILIES)
    }
    results["total/x"] = _fam("total/x", "total", _cmp(-0.01, -0.05, 0.03), _cmp(0.0, -0.1, 0.1))
    verdict = judge_confirmation(results)
    assert verdict.confirmed and verdict.positive_families == MIN_POSITIVE_FAMILIES


@pytest.mark.unit
def test_criterion_rejects_significant_harm_and_too_few_positives() -> None:
    harmed = {
        "win/a": _fam("win/a", "win", _cmp(-0.004, -0.007, -0.0015), _cmp(0.0, -0.003, 0.003)),
        "win/b": _fam("win/b", "win", _cmp(0.002, 0.0, 0.004), _cmp(0.0, -0.003, 0.003)),
    }
    v = judge_confirmation(harmed)
    assert not v.confirmed and v.harmed == ("win/a",)

    few = {
        f"win/{i}": _fam(
            f"win/{i}",
            "win",
            _cmp(-0.0005 if i else 0.001, -0.002, 0.002),
            _cmp(0.0, -0.003, 0.003),
        )
        for i in range(5)
    }
    v2 = judge_confirmation(few)
    assert not v2.confirmed and not v2.harmed and v2.positive_families == 1


@pytest.mark.unit
def test_total_uses_the_mae_harm_threshold() -> None:
    bad = {"total/x": _fam("total/x", "total", _cmp(-0.2, -0.3, -0.06), _cmp(0.0, -0.1, 0.1))}
    assert judge_confirmation(bad).harmed == ("total/x",)
    ok = {"total/x": _fam("total/x", "total", _cmp(-0.06, -0.16, -0.04), _cmp(0.0, -0.1, 0.1))}
    assert judge_confirmation(ok).harmed == ()


@pytest.mark.unit
def test_confirmation_end_to_end_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "eval_starts", lambda: [2022, 2023, 2024, 2025])
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    run = run_family(
        spec,
        rows,
        build_sweep_steps("expanded"),
        make_feature_provider(rows, spec, Path(".")),
        placebo_draws=0,
    )
    results = run_confirmation({spec.label: run}, {"total": rows}, Path("."))
    r = results["total/xgboost"]
    assert r.n_baseline == 152 and r.n_final == 177
    assert r.holdout.n_games == rows.n_holdout
    assert r.wide.n_games > r.holdout.n_games  # includes 2022
    assert r.week1 is None or r.week1.n_games <= r.wide.n_games
    verdict = judge_confirmation(results)
    text = build_confirmation_report(results, verdict, generated_on=date(2026, 9, 30))
    assert text.startswith("# Phase A final-composition confirmation - 2026-09-30")
    assert "## total/xgboost" in text and "Holdout 2023-2025" in text
    assert isinstance(verdict, ConfirmationVerdict)
