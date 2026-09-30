# tests/unit/evaluation/test_feature_sweep_stage3.py

"""Tests for the wide-window per-batch attribution stage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_metrics import bh_adjust
from gridiron_edge.evaluation.feature_sweep_run import make_feature_provider, run_family
from gridiron_edge.evaluation.feature_sweep_stage3 import (
    BH_Q,
    HARM,
    MIN_IMPROVEMENT,
    MIN_SEASON_FRACTION,
    build_stage3_report,
    decide_stage3,
    load_stage3,
    run_stage3,
)
from gridiron_edge.evaluation.feature_sweep_steps import FEATURE_BATCHES, build_sweep_steps
from gridiron_edge.models.game_prediction.base import GameModelType

from .test_feature_sweep_run import _rows, _tiny_budgets  # noqa: F401


@pytest.mark.unit
def test_pre_declared_criterion_is_pinned() -> None:
    assert MIN_IMPROVEMENT == {"win": 0.001, "total": 0.045}
    assert HARM == {"win": 0.001, "total": 0.05}
    assert BH_Q == 0.10
    assert MIN_SEASON_FRACTION == 0.60


@pytest.mark.unit
def test_bh_adjust_matches_textbook_example() -> None:
    adj = bh_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.0533333, rel=1e-4)
    assert adj["b"] == pytest.approx(0.0533333, rel=1e-4)
    assert adj["d"] == pytest.approx(0.20)
    assert bh_adjust({"x": 0.9})["x"] == pytest.approx(0.9)
    assert all(v <= 1.0 for v in bh_adjust({"a": 0.9, "b": 0.95}).values())


@pytest.mark.unit
def test_stage3_scores_every_batch_checkpoints_and_reports(tmp_path: Path) -> None:
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    run = run_family(
        spec,
        rows,
        build_sweep_steps("expanded"),
        make_feature_provider(rows, spec, Path(".")),
        placebo_draws=1,
    )
    seen: list[str] = []
    results = run_stage3(
        {spec.label: run},
        {"total": rows},
        Path("."),
        tmp_path,
        window=[2022, 2023, 2024, 2025],
        on_family=lambda label, _s: seen.append(label),
    )
    fam = results["total/xgboost"]
    assert [b.batch for b in fam.batches] == ["qb_and_coach"]
    for b in fam.batches:
        assert b.seasons_total == 4
        assert 0 <= b.seasons_improved <= 4
        assert b.comparison.margin == HARM["total"]

    again = run_stage3(
        {spec.label: run}, {"total": rows}, Path("."), tmp_path, window=[2022, 2023, 2024, 2025]
    )
    assert seen == ["total/xgboost"] and set(again) == set(load_stage3(tmp_path))

    live = {b.name: 0 for b in FEATURE_BATCHES}
    live["qb_and_coach"] = 5
    decisions, adjusted = decide_stage3(results, live)
    assert set(decisions) == {"qb_and_coach"}
    assert len(adjusted) == 1
    assert decisions["qb_and_coach"].status != "adopt"  # 5 rows with a NaN
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())
    text = build_stage3_report(results, decisions, adjusted, live, generated_on=date(2026, 9, 30))
    assert text.startswith("# Phase A per-batch attribution (wide window) - 2026-09-30")
    assert "## Batch decisions" in text and "## total/xgboost" in text
    assert "revised, before results" in text
