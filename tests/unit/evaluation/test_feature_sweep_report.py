# tests/unit/evaluation/test_feature_sweep_report.py

"""Tests for the consolidated sweep report and prediction payload."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pytest

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_report import (
    batch_live_complete,
    build_report,
    predictions_payload,
)
from gridiron_edge.evaluation.feature_sweep_run import make_feature_provider, run_family
from gridiron_edge.evaluation.feature_sweep_steps import FEATURE_BATCHES, build_sweep_steps
from gridiron_edge.models.game_prediction.base import GameModelType

from .test_feature_sweep_run import _frame, _rows, _tiny_budgets  # noqa: F401


@pytest.mark.unit
def test_batch_live_complete_counts_nan_rows_from_the_holdout_onward() -> None:
    df = _frame()
    clean = batch_live_complete(df)
    assert set(clean) == {b.name for b in FEATURE_BATCHES}
    assert all(v["rows_with_nan"] == 0 for v in clean.values())

    recent_week1 = df.index[(df["YEAR"] == "2024-2025") & (df["WEEK_NUM"] == 2)][0]
    old = df.index[df["YEAR"] == "2021-2022"][0]
    df.loc[recent_week1, "HOME_OFF_PASS_EPA_ADJ"] = np.nan
    df.loc[old, "HOME_QB_EPA_PLAY_CAREER"] = np.nan  # before the holdout: ignored
    out = batch_live_complete(df)
    assert out["opponent_adjusted_epa"]["rows_with_nan"] == 1
    assert out["opponent_adjusted_epa"]["week1_rows_with_nan"] == 0
    assert out["qb_and_coach"]["rows_with_nan"] == 0


@pytest.mark.unit
def test_report_and_payload_cover_every_batch_and_holdout_game() -> None:
    df = _frame()
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    run = run_family(
        spec,
        rows,
        build_sweep_steps("expanded"),
        make_feature_provider(rows, spec, Path(".")),
        placebo_draws=1,
    )
    text, decisions = build_report(
        {spec.label: run}, {"total": rows}, df, generated_on=date(2026, 9, 28)
    )
    assert text.startswith("# Phase A feature sweep - 2026-09-28")
    for batch in FEATURE_BATCHES:
        assert batch.name in text
        assert batch.name in decisions
        assert decisions[batch.name].status in {"adopt", "reject", "adopt_only_if_fixed"}
    for heading in ("## Protocol", "## Batch decisions", "## Multiple-comparison view"):
        assert heading in text
    assert "### total/xgboost" in text
    assert "reference: league_mean_total" in text

    payload = predictions_payload({spec.label: run}, {"total": rows})
    entry = payload["total/xgboost"]
    assert len(entry["game_ids"]) == rows.n_holdout == len(entry["actual"])
    assert set(entry["steps"]) == {s.name for s in build_sweep_steps("expanded")}
    assert all(len(v) == rows.n_holdout for v in entry["steps"].values())
    json.dumps(payload)


@pytest.mark.unit
def test_column_level_stage_feeds_the_report() -> None:
    from gridiron_edge.evaluation.feature_sweep_columns import run_column_level

    df = _frame()
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    run = run_family(
        spec,
        rows,
        build_sweep_steps("expanded"),
        make_feature_provider(rows, spec, Path(".")),
        placebo_draws=1,
    )
    cl = run_column_level({spec.label: run}, {"total": rows}, Path("."))
    assert set(cl.ablations["total/xgboost"]) == {
        "form_trend",
        "glicko2_rd",
        "glicko2_rating",
        "linear_ratings",
        "qb_elo",
        "qb_history",
    }
    assert set(cl.leakage) == set(cl.drift) == set(cl.correlations) == {"total"}
    assert "GLICKO2_RD_DIFF" in cl.correlations["total"]
    text, _ = build_report(
        {spec.label: run},
        {"total": rows},
        df,
        generated_on=date(2026, 9, 28),
        column_level=cl,
    )
    for heading in (
        "## Column-level findings",
        "### Leakage smell test",
        "### Train-versus-holdout",
    ):
        assert heading in text
