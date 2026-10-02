# tests/unit/models/test_logistic_neutral_site.py

"""Tests for the Logistic Win model's neutral-site handling (D74)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.models.game_prediction._columns import _NEUTRAL_SITE_FEATURE
from gridiron_edge.models.game_prediction._features import FEATURE_SETS
from gridiron_edge.models.game_prediction.base import (
    GameModelType,
    _create_model,
    _fit_final_model,
    _fit_rows_for_model,
    _pin_neutral_site_term,
)

pytestmark = pytest.mark.unit


def _frame(rows: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "ELO_DIFF": rng.normal(0, 60, rows),
            "OFF_EPA_PER_PLAY_DIFF": rng.normal(0, 0.1, rows),
            _NEUTRAL_SITE_FEATURE: (rng.random(rows) < 0.1).astype(int),
        }
    )
    logit = 0.3 * (1 - x[_NEUTRAL_SITE_FEATURE]) + x["ELO_DIFF"] / 100
    y = pd.Series((rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int))
    return x, y


def test_combined_set_carries_neutral_flag_once_and_expanded_does_not_duplicate_it() -> None:
    combined = FEATURE_SETS["combined"].feature_names
    expanded = FEATURE_SETS["expanded"].feature_names

    assert combined.count(_NEUTRAL_SITE_FEATURE) == 1
    assert expanded.count(_NEUTRAL_SITE_FEATURE) == 1
    assert len(expanded) == len(set(expanded))


def test_logistic_fits_on_non_neutral_rows_only() -> None:
    x, y = _frame()

    x_fit, y_fit = _fit_rows_for_model(GameModelType.LOGISTIC, x, y)

    assert (x_fit[_NEUTRAL_SITE_FEATURE] == 0).all()
    assert len(x_fit) == int((x[_NEUTRAL_SITE_FEATURE] == 0).sum())
    assert y_fit.index.equals(x_fit.index)


@pytest.mark.parametrize("model_type", [GameModelType.RANDOM_FOREST, GameModelType.XGBOOST])
def test_tree_models_keep_every_row(model_type: GameModelType) -> None:
    x, y = _frame()

    x_fit, y_fit = _fit_rows_for_model(model_type, x, y)

    assert len(x_fit) == len(x)
    assert len(y_fit) == len(y)


def test_pinned_neutral_coefficient_removes_the_intercept_at_neutral_sites() -> None:
    x, y = _frame()
    x_fit, y_fit = _fit_rows_for_model(GameModelType.LOGISTIC, x, y)

    model, scaler = _fit_final_model(GameModelType.LOGISTIC, "classification", {}, x_fit, y_fit)

    index = list(x.columns).index(_NEUTRAL_SITE_FEATURE)
    assert model.coef_[0, index] == -model.intercept_[0]

    home_site = x.assign(**{_NEUTRAL_SITE_FEATURE: 0})
    neutral_site = x.assign(**{_NEUTRAL_SITE_FEATURE: 1})
    gap = model.decision_function(scaler.transform(neutral_site)) - model.decision_function(
        scaler.transform(home_site)
    )
    np.testing.assert_allclose(gap, -model.intercept_[0])


def test_pinning_leaves_non_neutral_predictions_unchanged() -> None:
    x, y = _frame()
    x_fit, y_fit = _fit_rows_for_model(GameModelType.LOGISTIC, x, y)
    pinned, pinned_scaler = _fit_final_model(
        GameModelType.LOGISTIC, "classification", {}, x_fit, y_fit
    )

    unpinned, unpinned_scaler = _create_model(GameModelType.LOGISTIC, "classification")
    unpinned.fit(unpinned_scaler.fit_transform(x_fit), y_fit)

    non_neutral = x.loc[x[_NEUTRAL_SITE_FEATURE] == 0]
    np.testing.assert_allclose(
        pinned.predict_proba(pinned_scaler.transform(non_neutral)),
        unpinned.predict_proba(unpinned_scaler.transform(non_neutral)),
    )


def test_log_odds_reconstruct_from_coefficients_scaler_and_intercept() -> None:
    x, y = _frame()
    x_fit, y_fit = _fit_rows_for_model(GameModelType.LOGISTIC, x, y)
    model, scaler = _fit_final_model(GameModelType.LOGISTIC, "classification", {}, x_fit, y_fit)

    scaled = scaler.transform(x)
    reconstructed = scaled @ model.coef_[0] + model.intercept_[0]

    np.testing.assert_allclose(reconstructed, model.decision_function(scaled), atol=1e-9)


def test_pin_requires_the_neutral_column_to_be_constant_zero_in_the_fit() -> None:
    x, y = _frame()
    model, scaler = _create_model(GameModelType.LOGISTIC, "classification")
    model.fit(scaler.fit_transform(x), y)

    with pytest.raises(ValueError, match="constant zero"):
        _pin_neutral_site_term(GameModelType.LOGISTIC, model, scaler, list(x.columns))


def test_pin_ignores_models_other_than_logistic() -> None:
    x, y = _frame()
    model, scaler = _create_model(GameModelType.LOGISTIC, "classification")
    model.fit(scaler.fit_transform(x), y)
    before = model.coef_.copy()

    _pin_neutral_site_term(GameModelType.RANDOM_FOREST, model, scaler, list(x.columns))

    np.testing.assert_array_equal(model.coef_, before)
