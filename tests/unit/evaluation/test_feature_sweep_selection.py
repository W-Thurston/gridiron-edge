# tests/unit/evaluation/test_feature_sweep_selection.py

"""Tests for model-driven feature selection and its placebo control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_selection import (
    select_logistic,
    select_tree,
)
from gridiron_edge.models.game_prediction.base import GameModelType

N = 2400
SIGNAL = ["sig_a", "sig_b", "sig_c"]
NOISE = [f"noise_{i}" for i in range(17)]


def _data(seed: int = 5) -> tuple[DataFrame, Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(N, len(SIGNAL) + len(NOISE))), columns=SIGNAL + NOISE)
    logit = 1.2 * x["sig_a"] + 0.9 * x["sig_b"] - 0.8 * x["sig_c"]
    y = pd.Series((rng.uniform(size=N) < 1 / (1 + np.exp(-logit))).astype(float))
    return x, y


@pytest.mark.unit
def test_logistic_selection_keeps_signal_and_reports_placebo() -> None:
    x, y = _data()
    result = select_logistic(x, y, n_placebo=20)
    assert set(SIGNAL) <= set(result.selected)
    assert len(result.selected) <= len(SIGNAL) + 4
    assert result.placebo_candidates == 20
    assert result.placebo_survival_rate <= 0.25
    assert result.n_pool == 20 and result.n_folds >= 2
    assert all(0.0 <= v <= 1.0 for v in result.stability.values())
    assert result.detail["chosen_C"] > 0


@pytest.mark.unit
def test_logistic_selection_is_deterministic() -> None:
    x, y = _data()
    assert select_logistic(x, y, n_placebo=10) == select_logistic(x, y, n_placebo=10)


@pytest.mark.unit
def test_tree_selection_finds_signal_and_rejects_placebo() -> None:
    x, y = _data()
    spec = FamilySpec("win", GameModelType.XGBOOST, "expanded")
    params = {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1}
    result = select_tree(
        spec, params, x, y, n_placebo=20, n_shadow=15, n_repeats=2, n_splits=4, threads=1
    )
    assert set(SIGNAL) <= set(result.selected)
    assert result.placebo_survival_rate <= 0.1
    assert len(result.selected) < len(x.columns)
    assert result.detail["mean_shadow_bar"] >= 0.0
    assert result.method == "shadow_permutation"


@pytest.mark.unit
def test_tree_selection_regression_target() -> None:
    x, _ = _data()
    total = 45 + 6 * x["sig_a"] + np.random.default_rng(1).normal(0, 2, size=N)
    spec = FamilySpec("total", GameModelType.XGBOOST, "expanded")
    params = {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1}
    result = select_tree(
        spec, params, x, total, n_placebo=10, n_shadow=15, n_repeats=2, n_splits=4, threads=1
    )
    assert "sig_a" in result.selected
    assert result.placebo_survival_rate <= 0.1


@pytest.mark.unit
def test_selection_never_reads_rows_it_is_not_given() -> None:
    # A column that is a perfect copy of the target on *unseen* rows must not
    # matter: selection is a function of the rows passed in.
    x, y = _data()
    train = x.iloc[:1800]
    result_a = select_logistic(train, y.iloc[:1800], n_placebo=10)
    leaked = x.copy()
    leaked.loc[1800:, "noise_0"] = y.iloc[1800:]
    result_b = select_logistic(leaked.iloc[:1800], y.iloc[:1800], n_placebo=10)
    assert result_a == result_b


@pytest.mark.unit
def test_too_little_data_raises() -> None:
    x, y = _data()
    with pytest.raises(ValueError, match="No CV fold"):
        select_logistic(x.iloc[:200], y.iloc[:200], min_train_rows=1500)
