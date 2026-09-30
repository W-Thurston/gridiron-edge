# tests/unit/evaluation/test_feature_sweep_driver.py

"""Tests for the checkpointed sweep driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridiron_edge.evaluation import feature_sweep_driver as driver
from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_run import FamilyRun
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.models.game_prediction.base import GameModelType


@pytest.mark.unit
def test_all_families_match_the_sweep_scope() -> None:
    labels = [f.label for f in driver.ALL_FAMILIES]
    assert sorted(labels) == [
        "total/random_forest",
        "total/xgboost",
        "win/logistic",
        "win/random_forest",
        "win/xgboost",
    ]
    by_label = {f.label: f.feature_set_key for f in driver.ALL_FAMILIES}
    assert by_label["win/logistic"] == "combined"
    assert all(v == "expanded" for k, v in by_label.items() if k != "win/logistic")


@pytest.mark.unit
def test_checkpoint_round_trip_and_load_runs(tmp_path: Path) -> None:
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    path = driver._checkpoint_path(tmp_path, spec.label)
    assert path.name == "total__xgboost.pkl"
    assert driver._load(path) is None
    driver._save(path, FamilyRun(spec=spec))
    loaded = driver._load(path)
    assert loaded is not None and loaded.spec == spec
    assert list(driver.load_runs(tmp_path)) == ["total/xgboost"]
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.unit
def test_run_sweep_skips_a_completed_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    steps = build_sweep_steps("expanded")
    complete = FamilyRun(spec=spec)
    complete.steps = [object() for _ in steps]  # type: ignore[list-item]
    driver._save(driver._checkpoint_path(tmp_path, spec.label), complete)

    monkeypatch.setattr(driver, "build_matched_rows", lambda *_a, **_k: "rows")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("run_family must not run for a completed family")

    monkeypatch.setattr(driver, "run_family", _boom)
    runs, matched = driver.run_sweep(None, Path("."), tmp_path, families=(spec,))  # type: ignore[arg-type]
    assert runs["total/xgboost"].spec == spec
    assert matched == {"total": "rows"}
