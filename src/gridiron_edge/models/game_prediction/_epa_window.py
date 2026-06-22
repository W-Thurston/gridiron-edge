# src/gridiron_edge/models/game_prediction/_epa_window.py

"""EPA rolling window hyperparameter infrastructure for tree-based models.

The standard modeling file uses a fixed 4-game rolling window for EPA
features. This module provides the infrastructure to search over different
window sizes as a hyperparameter during training.

Public API
----------
_EPA_RAW_COLS       list[str]       - EPA column names from epa_by_game.parquet
_EPA_COL_MAP        dict[str, str]  - lowercase → uppercase EPA column mapping
_EPA_WINDOW_OPTIONS list[int]       - window sizes searched during tuning
WindowData          NamedTuple      - cached train/holdout split per window
_rebuild_features_with_window       - recompute EPA features with given window
_get_cached_window_data             - retrieve/populate per-window cache entry
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.team.epa import EPA_COLS as _EPA_COLS_RAW

if TYPE_CHECKING:
    pass

logger: Logger = logging.getLogger(__name__)

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
    """Recompute rolling EPA features with a configurable window size.

    The standard modeling file uses a fixed 4-game rolling window. This
    function loads the raw game-level EPA data and recomputes rolling
    averages with a different window, then splices the result back into
    the modeling DataFrame. Called during hyperparameter search when
    epa_window is a tunable parameter.

    Fast path: if window == 4, returns df unchanged (no disk read needed).

    Args:
        df: Full modeling DataFrame from load_modeling_file.
        window: Rolling window size (number of prior games to average).
        repo: Repository root (for loading epa_by_game.parquet).

    Returns:
        Modeling DataFrame with TEAM_A_* and TEAM_B_* EPA columns
        recomputed using the requested window. NaN rows from incomplete
        windows are retained; callers apply the NaN mask after feature
        engineering.
    """
    from gridiron_edge.datasets.loaders import load_epa_by_game

    if window == 4:
        return df

    epa_raw: pd.DataFrame = load_epa_by_game(repo)
    if epa_raw.empty:
        logger.warning("epa_by_game.parquet not found - returning df unchanged")
        return df

    epa_sorted: pd.DataFrame = epa_raw.sort_values(["season", "week", "team"]).copy()

    # Compute rolling mean per team with shift(1) to prevent lookahead
    rolled_parts: list[pd.DataFrame] = []
    for _team, grp in epa_sorted.groupby("team", sort=False):
        grp_sorted: DataFrame = grp.sort_values(["season", "week"]).copy()
        for col in _EPA_RAW_COLS:
            grp_sorted[f"{col}_roll"] = (
                grp_sorted[col].shift(1).rolling(window=window, min_periods=1).mean()
            )
        rolled_parts.append(grp_sorted)

    rolled: pd.DataFrame = pd.concat(rolled_parts, ignore_index=True)

    roll_cols: list[str] = [f"{c}_roll" for c in _EPA_RAW_COLS]
    lookup: DataFrame = rolled.loc[:, ["season", "week", "team", *roll_cols]].copy()

    # Build season int → YEAR string mapping from the modeling file itself
    # (epa_by_game uses int seasons like 2024; modeling file uses "2024-2025")
    year_to_season: dict[str, int] = {}
    for year_str in df["YEAR"].unique():
        with contextlib.suppress(ValueError, IndexError):
            year_to_season[year_str] = int(str(year_str).split("-")[0])

    lookup["YEAR"] = lookup["season"].map({v: k for k, v in year_to_season.items()})
    lookup = lookup.dropna(subset=["YEAR"])

    # Drop existing EPA columns before merging updated ones
    team_a_epa_cols: list[str] = [f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]
    team_b_epa_cols: list[str] = [f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]
    df = df.copy().drop(columns=team_a_epa_cols + team_b_epa_cols, errors="ignore")

    # Merge TEAM_A EPA
    team_a_merge: DataFrame = lookup.rename(
        columns={
            "team": "TEAM_A",
            "week": "WEEK_NUM",
            **{f"{c}_roll": f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS},
        }
    ).loc[:, ["TEAM_A", "YEAR", "WEEK_NUM", *[f"TEAM_A_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]]]
    df = df.merge(team_a_merge, on=["TEAM_A", "YEAR", "WEEK_NUM"], how="left")

    # Merge TEAM_B EPA
    team_b_merge: DataFrame = lookup.rename(
        columns={
            "team": "TEAM_B",
            "week": "WEEK_NUM",
            **{f"{c}_roll": f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS},
        }
    ).loc[:, ["TEAM_B", "YEAR", "WEEK_NUM", *[f"TEAM_B_{_EPA_COL_MAP[c]}" for c in _EPA_RAW_COLS]]]
    df = df.merge(team_b_merge, on=["TEAM_B", "YEAR", "WEEK_NUM"], how="left")

    return df


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
