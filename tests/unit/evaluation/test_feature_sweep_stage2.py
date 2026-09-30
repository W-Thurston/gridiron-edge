# tests/unit/evaluation/test_feature_sweep_stage2.py

"""Tests for the stage 2 driver and report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_run import make_feature_provider, run_family
from gridiron_edge.evaluation.feature_sweep_stage2 import (
    build_stage2_report,
    load_stage2,
    run_stage2,
)
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.models.game_prediction.base import GameModelType

from .test_feature_sweep_run import _rows, _tiny_budgets  # noqa: F401

SELECTION = {
    "n_placebo": 4,
    "n_shadow": 6,
    "n_repeats": 1,
    "n_splits": 2,
    "min_train_rows": 30,
}


@pytest.mark.unit
def test_stage2_selects_evaluates_checkpoints_and_reports(tmp_path: Path) -> None:
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    run = run_family(
        spec,
        rows,
        build_sweep_steps("expanded"),
        make_feature_provider(rows, spec, Path(".")),
        placebo_draws=1,
    )
    done: list[str] = []
    results = run_stage2(
        {spec.label: run},
        {"total": rows},
        Path("."),
        tmp_path,
        threads=1,
        early_cutoff=2022,
        last_eval=2025,
        selection_kwargs=SELECTION,
        on_family=lambda label, _s: done.append(label),
    )
    r = results["total/xgboost"]
    assert done == ["total/xgboost"]
    assert r.n_baseline == 152 and r.n_full == 260
    assert r.early.n_pool == r.final.n_pool == 260
    assert r.early.placebo_candidates == r.final.placebo_candidates == 4
    assert set(r.holdout) >= {"full_vs_baseline"}
    assert r.holdout["full_vs_baseline"].n_games == rows.n_holdout
    if r.early.selected:
        assert r.wide["selected_vs_full"].n_games > rows.n_holdout  # includes 2022
    # The early subset never saw 2022 onward: its training rows precede the cutoff.
    assert r.wide["full_vs_baseline"].n_games == int(
        (rows.matched & (rows.frame["YEAR"].str[:4].astype(int) >= 2022)).sum()
    )

    # A second call resumes from the checkpoint without recomputing.
    again = run_stage2(
        {spec.label: run},
        {"total": rows},
        Path("."),
        tmp_path,
        selection_kwargs=SELECTION,
        on_family=lambda label, _s: done.append(f"again:{label}"),
    )
    assert done == ["total/xgboost"]
    assert again["total/xgboost"].final.selected == r.final.selected
    assert set(load_stage2(tmp_path)) == {"total/xgboost"}

    text = build_stage2_report(results, generated_on=date(2026, 9, 29))
    for heading in (
        "# Phase A feature selection - 2026-09-29",
        "## Selection summary",
        "## total/xgboost",
        "### Wide walk-forward (early subset)",
        "### Holdout (final subset)",
        "### Final subset",
    ):
        assert heading in text
