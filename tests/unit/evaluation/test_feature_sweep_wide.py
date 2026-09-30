# tests/unit/evaluation/test_feature_sweep_wide.py

"""Tests for the wide walk-forward evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron_edge.evaluation import feature_sweep_fit as fit_mod
from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec, SearchOutcome
from gridiron_edge.evaluation.feature_sweep_wide import (
    SELECTION_CUTOFF_START,
    WideArm,
    compare_non_inferiority,
    eval_starts,
    train_mask_before,
    wide_predictions,
)
from gridiron_edge.models.game_prediction.base import GameModelType

from .test_feature_sweep_run import _rows, _tiny_budgets  # noqa: F401

PARAMS = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}


def _outcome() -> SearchOutcome:
    return SearchOutcome(PARAMS, 4, 0.0, False, None)


@pytest.mark.unit
def test_window_and_masks() -> None:
    assert eval_starts()[0] == SELECTION_CUTOFF_START == 2013
    assert eval_starts()[-1] == 2025
    assert eval_starts(2020, 2022) == [2020, 2021, 2022]
    rows = _rows("total")
    mask = train_mask_before(rows, 2023)
    years = set(rows.frame.loc[mask, "YEAR"])
    assert years == {"2021-2022", "2022-2023"}
    assert not (mask & ~rows.matched).any()


@pytest.mark.unit
def test_informative_arm_beats_uninformative_and_is_not_non_inferior_reverse() -> None:
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    good = WideArm("good", ("AWAY_OFF_EPA_PER_PLAY", "HOME_OFF_EPA_PER_PLAY"), _outcome())
    bad = WideArm("bad", ("HOME_ELO", "AWAY_ELO"), _outcome())
    starts = [2023, 2024, 2025]
    preds = wide_predictions(spec, rows, lambda _w: rows.features, [good, bad], starts=starts)
    assert preds["good"][rows.matched & (rows.frame["YEAR"] < "2023")].isna().all()

    forward = compare_non_inferiority(spec, rows, preds, "good", "bad", starts=starts)
    assert forward.delta.improvement > 0.5
    assert forward.non_inferior
    assert forward.n_games == int(
        (rows.matched & rows.frame["YEAR"].isin(["2023-2024", "2024-2025", "2025-2026"])).sum()
    )
    assert set(forward.season_deltas) == {"2023-2024", "2024-2025", "2025-2026"}

    backward = compare_non_inferiority(spec, rows, preds, "bad", "good", starts=starts)
    assert not backward.non_inferior
    assert backward.margin == 0.045


@pytest.mark.unit
def test_identical_arms_are_non_inferior_with_zero_delta() -> None:
    rows = _rows("win")
    spec = FamilySpec("win", GameModelType.XGBOOST, "expanded")
    cols = ("HOME_ELO", "AWAY_ELO")
    a = WideArm("a", cols, _outcome())
    b = WideArm("b", cols, _outcome())
    preds = wide_predictions(spec, rows, lambda _w: rows.features, [a, b], starts=[2024, 2025])
    result = compare_non_inferiority(spec, rows, preds, "a", "b", starts=[2024, 2025])
    assert result.delta.improvement == pytest.approx(0.0)
    assert result.non_inferior
    assert result.margin == 0.001


@pytest.mark.unit
def test_nan_predictions_raise() -> None:
    rows = _rows("win")
    spec = FamilySpec("win", GameModelType.XGBOOST, "expanded")
    preds = {"a": rows.y * np.nan, "b": rows.y * np.nan}
    with pytest.raises(ValueError, match="NaN"):
        compare_non_inferiority(spec, rows, preds, "a", "b", starts=[2024])


@pytest.mark.unit
def test_fit_estimator_applies_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []
    real = fit_mod._set_threads

    def spy(model: object, threads: int) -> None:
        real(model, threads)  # type: ignore[arg-type]
        seen.append({k: v for k, v in model.get_params().items() if k.endswith("n_jobs")})  # type: ignore[attr-defined]

    monkeypatch.setattr(fit_mod, "_set_threads", spy)
    rows = _rows("total")
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    fit_mod.fit_estimator(
        spec,
        PARAMS,
        rows.features.loc[rows.train_mask, ["HOME_ELO"]],
        rows.y[rows.train_mask],
        threads=1,
    )
    assert seen == [{"n_jobs": 1}]
