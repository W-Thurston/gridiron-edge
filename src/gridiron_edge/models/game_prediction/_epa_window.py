# src/gridiron_edge/models/game_prediction/_epa_window.py

"""EPA rolling-window hyperparameter infrastructure.

The persisted canonical modeling artifact uses a four-game EPA window.
This module supports alternate tuning windows by delegating EPA
recalculation to ``HomeAwayEpaFeature`` so standard feature generation
and hyperparameter search share one implementation.

Public API
----------
_EPA_RAW_COLS       list[str]       EPA source columns.
_EPA_COL_MAP        dict[str, str]  Source-to-model suffix mapping.
_EPA_WINDOW_OPTIONS list[int]       Window sizes searched during tuning.
WindowData          NamedTuple      Cached train/holdout data per window.
_rebuild_features_with_window       Rebuild canonical EPA for a window.
_get_cached_window_data             Retrieve or populate the cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.team.epa import (
    EPA_COLS as _EPA_COLS_RAW,
)
from gridiron_edge.features.team.epa import (
    HomeAwayEpaFeature,
)

# ---------------------------------------------------------------------------
# EPA column definitions
# ---------------------------------------------------------------------------

# EPA column names as they appear in epa_by_game.parquet (lowercase).
# Derived from the feature module - do not enumerate independently.
_EPA_RAW_COLS: Final[list[str]] = list(_EPA_COLS_RAW)

# Mapping from epa_by_game column name → modeling file column suffix.
# Derived programmatically: "off_epa_per_play" → "OFF_EPA_PER_PLAY".
_EPA_COL_MAP: Final[dict[str, str]] = {c: c.upper() for c in _EPA_RAW_COLS}

# Window sizes searched as a hyperparameter during training.
_EPA_WINDOW_OPTIONS: Final[list[int]] = [1, 2, 3, 4, 6, 8]


# ---------------------------------------------------------------------------
# Window cache type
# ---------------------------------------------------------------------------


class WindowData(NamedTuple):
    """Pre-computed train/holdout split for a given EPA rolling window size.

    Cached by ``_get_cached_window_data`` to avoid redundant disk reads
    during hyperparameter search over ``_EPA_WINDOW_OPTIONS``.
    """

    df_windowed: pd.DataFrame
    x_train: pd.DataFrame
    y_train: Series
    x_holdout: pd.DataFrame
    y_holdout: Series
    train_seasons: list[str]
    holdout_seasons: list[str]


# ---------------------------------------------------------------------------
# Window feature rebuild
# ---------------------------------------------------------------------------


def _rebuild_features_with_window(
    df: pd.DataFrame,
    *,
    window: int,
    repo: Path,
) -> pd.DataFrame:
    """Recompute canonical Away/Home EPA using another rolling window.

    The persisted modeling artifact contains the standard four-game EPA
    values. Window four therefore returns the input unchanged. Other
    windows replace the canonical EPA columns by delegating to the same
    feature implementation used by the active modeling pipeline.

    Args:
        df: Canonical one-row-per-game modeling DataFrame.
        window: Number of prior games included in each rolling average.
        repo: Repository root used to load game-level EPA data.

    Returns:
        A new DataFrame with canonical Away and Home EPA columns for the
        requested window. The window-four fast path returns ``df``
        unchanged.

    Raises:
        ValueError: If ``window`` is less than one or the canonical EPA
            source or target identities are invalid.
    """
    if window == 4:
        return df

    feature_columns = list(HomeAwayEpaFeature.spec.produces)
    source = df.drop(
        columns=feature_columns,
        errors="ignore",
    ).copy()

    return HomeAwayEpaFeature(window=window).compute(
        df=source,
        datasets=DatasetAccessor(repo),
    )


# ---------------------------------------------------------------------------
# Window cache helper
# ---------------------------------------------------------------------------


def _get_cached_window_data(
    cache: dict[int, WindowData],
    window: int,
    df: pd.DataFrame,
    feature_fn: Callable,
    repo: Path,
) -> WindowData:
    """Return cached train/holdout split for a given EPA rolling window size.

    On first access for a given window, rebuilds EPA features and runs
    _prepare_data; subsequent accesses return the cached result. Since
    _EPA_WINDOW_OPTIONS has at most 6 values, the cache is bounded and
    eliminates the dominant cost in the hyperparameter loop: repeated
    parquet loads for the same window.

    Args:
        cache: Mutable dict keyed by window size; populated in place.
        window: Rolling EPA window size to retrieve.
        df: Full raw modeling DataFrame (window=4 baseline).
        feature_fn: Feature engineering function passed to _prepare_data.
        repo: Repository root for _rebuild_features_with_window.

    Returns:
        ``WindowData`` named tuple with df_windowed, x_train, y_train,
        x_holdout, y_holdout, train_seasons, holdout_seasons.
    """
    if window not in cache:
        from gridiron_edge.models.game_prediction._features import _prepare_data

        df_w: DataFrame = _rebuild_features_with_window(df, window=window, repo=repo)
        x_tr, y_tr, x_ho, y_ho, tr_s, ho_s = _prepare_data(df_w, feature_fn)
        cache[window] = WindowData(df_w, x_tr, y_tr, x_ho, y_ho, tr_s, ho_s)
    return cache[window]
