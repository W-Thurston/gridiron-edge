# tests/fixtures/helpers.py

"""Shared test helpers for integration and end-to-end tests.

Three named helpers, organized by what they do — not where they're used.
Tests import from this module instead of duplicating context-manager and
assertion patterns across multiple test files.

Public API
----------
patch_minimal_param_grid    Context manager: replace HP grids with a
                            single combination for fast tests.
assert_predictions_reasonable  Assert prediction outputs are in valid range
                               and not all NaN / not all extreme.
assert_archive_schema_valid    Assert archive DataFrame matches the
                               canonical prediction-archive schema.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# ---------------------------------------------------------------------------
# HP grid minimization
# ---------------------------------------------------------------------------


@contextmanager
def patch_minimal_param_grid(
    *,
    grid_size: int = 1,
    min_cv_train_rows: int = 10,
) -> Generator[None, None, None]:
    """Replace ``_get_param_grid`` and ``_n_iter_for`` with minimal versions.

    Game-side trainers run randomized hyperparameter searches over grids
    with hundreds of combinations. For integration tests, this would take
    minutes per model. This context manager patches the factory to return
    a single (or a few) parameter combinations, reducing per-test time from
    minutes to seconds.

    Also patches ``MIN_CV_TRAIN_ROWS`` so CV folds aren't skipped on
    small synthetic data (production value is 4000; tests typically have
    a few hundred rows).

    Args:
        grid_size: Number of parameter combinations to keep. Defaults to 1
            (fastest possible — single fit per CV fold per model).
        min_cv_train_rows: Minimum fold size required before the trainer
            considers the fold valid. Defaults to 10 (small enough to
            accept synthetic test data, large enough to avoid degenerate
            single-row folds).

    Yields:
        None. The patches are active inside the ``with`` block.

    Example::

        with patch_minimal_param_grid():
            trainer = WinProbTrainer()
            metadata = trainer.train(df, model_type=GameModelType.LOGISTIC)
    """
    from gridiron_edge.models.game_prediction import _features as features_module
    from gridiron_edge.models.game_prediction import base as game_base

    real_get_param_grid = game_base._get_param_grid
    real_n_iter_for = game_base._n_iter_for

    def minimal_grid(model_type: Any, task: str) -> list[dict[str, Any]]:
        full_grid: list[dict[str, Any]] = real_get_param_grid(model_type, task)
        return full_grid[:grid_size]

    def minimal_n_iter(model_type: Any, task: str) -> int:
        return min(grid_size, real_n_iter_for(model_type, task))

    with (
        patch.object(game_base, "_get_param_grid", side_effect=minimal_grid),
        patch.object(game_base, "_n_iter_for", side_effect=minimal_n_iter),
        patch.object(features_module, "MIN_CV_TRAIN_ROWS", min_cv_train_rows),
    ):
        yield


# ---------------------------------------------------------------------------
# Prediction assertions
# ---------------------------------------------------------------------------


def assert_predictions_reasonable(
    predictions: Series | np.ndarray,
    *,
    task: str,
    allow_extreme: bool = False,
    name: str = "predictions",
) -> None:
    """Assert prediction outputs are in valid range and not pathological.

    Used as a single assertion at the end of fit-load-predict tests to
    catch a class of bugs where the model trains successfully but predicts
    nonsense at inference time (e.g. the scaler-not-applied bug).

    For classification: probabilities in [0, 1], finite, not all NaN, std
    in [0.05, 0.30] unless ``allow_extreme=True``. The std band rejects
    the scaler bug's signature of std ≈ 0.5.

    For regression: predictions finite, not all NaN, not all the same
    value.

    Args:
        predictions: Predicted values from the model.
        task: Either ``"classification"`` or ``"regression"``.
        allow_extreme: When True, skip the std band check for
            classification. Useful for tests with very small data where
            the model may legitimately produce concentrated probabilities.
        name: Display name for assertion error messages.

    Raises:
        AssertionError: If any check fails.
    """
    arr: np.ndarray = np.asarray(predictions, dtype=float)

    if len(arr) == 0:
        msg: str = f"{name}: predictions array is empty"
        raise AssertionError(msg)

    n_nan: int = int(np.isnan(arr).sum())
    n_total: int = len(arr)
    if n_nan == n_total:
        msg = f"{name}: all {n_total} predictions are NaN"
        raise AssertionError(msg)

    finite_arr: np.ndarray = arr[np.isfinite(arr)]
    if len(finite_arr) == 0:
        msg = f"{name}: no finite predictions (all NaN or inf)"
        raise AssertionError(msg)

    if task == "classification":
        if finite_arr.min() < 0.0 or finite_arr.max() > 1.0:
            msg = (
                f"{name}: classification probs out of [0, 1] range: "
                f"min={finite_arr.min():.4f}, max={finite_arr.max():.4f}"
            )
            raise AssertionError(msg)

        if not allow_extreme:
            std: float = float(finite_arr.std())
            if std < 0.01:
                msg = (
                    f"{name}: classification probs essentially constant "
                    f"(std={std:.4f} < 0.01). Model is not learning."
                )
                raise AssertionError(msg)
            if std > 0.30:
                msg = (
                    f"{name}: classification probs too dispersed "
                    f"(std={std:.4f} > 0.30). Model may be slamming "
                    f"predictions to corners — check if scaler was applied "
                    f"at predict time."
                )
                raise AssertionError(msg)

    elif task == "regression":
        if len(np.unique(finite_arr)) == 1:
            msg = (
                f"{name}: all regression predictions are the same value "
                f"({finite_arr[0]:.4f}). Model may not be learning."
            )
            raise AssertionError(msg)

    else:
        msg = f"task must be 'classification' or 'regression', got {task!r}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Archive schema assertions
# ---------------------------------------------------------------------------


def assert_archive_schema_valid(df: DataFrame) -> None:
    """Assert a DataFrame conforms to the prediction-archive schema.

    Used in archive round-trip tests to catch schema drift. The check
    is column-presence-based — extra columns are allowed (downstream code
    handles them), but required columns must be present and have the
    expected dtypes for the identity fields.

    Args:
        df: DataFrame written to or read from the prediction archive.

    Raises:
        AssertionError: If schema mismatch is detected.
    """
    from gridiron_edge.evaluation.archive import _ARCHIVE_COLUMNS

    missing: set[str] = set(_ARCHIVE_COLUMNS) - set(df.columns)
    if missing:
        msg: str = (
            f"archive DataFrame missing required columns: {sorted(missing)}. "
            f"Expected columns: {_ARCHIVE_COLUMNS}"
        )
        raise AssertionError(msg)

    # Identity field type checks — these are the dedup key fields.
    if not pd.api.types.is_string_dtype(df["model_name"]) and df["model_name"].dtype != "object":
        msg = f"model_name column has unexpected dtype: {df['model_name'].dtype}"
        raise AssertionError(msg)
    if not pd.api.types.is_string_dtype(df["model_type"]) and df["model_type"].dtype != "object":
        msg = f"model_type column has unexpected dtype: {df['model_type'].dtype}"
        raise AssertionError(msg)
    if not pd.api.types.is_string_dtype(df["game_id"]) and df["game_id"].dtype != "object":
        msg = f"game_id column has unexpected dtype: {df['game_id'].dtype}"
        raise AssertionError(msg)
