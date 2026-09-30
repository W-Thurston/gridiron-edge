# src/gridiron_edge/evaluation/feature_sweep_rows.py

"""Row-matched train/holdout data for the consolidated feature sweep.

Every sweep step and model family is scored on identical games. The matched
row set is the rows with a complete value for the widest (final) expanded
feature set; because every step's columns are a subset of it, those rows are
valid at every step. The split is the canonical chronological one: holdout is
exactly ``HOLDOUT_SEASONS`` and training is strictly earlier seasons
(``_chronological_split_masks``). This module also reports the rows each
step would lose on its own, so training-row loss stays on record.

Public API
----------
MatchedRows             dataclass   - matched frame, masks, per-step loss
build_matched_rows      df, target -> MatchedRows
step_row_loss           per-step own-pool row counts
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.feature_sweep_steps import (
    FeatureStep,
    build_sweep_steps,
)
from gridiron_edge.models.game_prediction._features import (
    _chronological_split_masks,
    _make_expanded_features,
)

_SORT_KEYS: list[str] = ["YEAR", "WEEK_NUM", "GAME_DATE", "GAME_ID"]


@dataclass(frozen=True)
class StepRowCount:
    """Rows a step would keep on its own, before matching.

    Attributes:
        step: Step name.
        n_train: Training rows with complete values for the step's columns.
        n_holdout: Holdout rows with complete values for the step's columns.
    """

    step: str
    n_train: int
    n_holdout: int


@dataclass(frozen=True)
class MatchedRows:
    """Row-matched sweep data for one target.

    Attributes:
        frame: Sorted modeling rows with a non-null target, index reset.
        features: Full expanded feature matrix aligned with ``frame``.
        y: Target aligned with ``frame``.
        matched: Rows valid on the widest expanded feature set.
        train_mask: Matched training rows (seasons before the holdout).
        holdout_mask: Matched holdout rows.
        universe_train: Training rows before matching.
        universe_holdout: Holdout rows before matching.
        own_pool: Per-step rows kept without matching, baseline first.
    """

    frame: DataFrame
    features: DataFrame
    y: Series
    matched: Series
    train_mask: Series
    holdout_mask: Series
    universe_train: int
    universe_holdout: int
    own_pool: tuple[StepRowCount, ...]

    @property
    def n_train(self) -> int:
        """Matched training row count."""
        return int(self.train_mask.sum())

    @property
    def n_holdout(self) -> int:
        """Matched holdout row count."""
        return int(self.holdout_mask.sum())


def step_row_loss(
    features: DataFrame,
    steps: list[FeatureStep],
    train_universe: Series,
    holdout_universe: Series,
) -> tuple[StepRowCount, ...]:
    """Count the rows each step keeps on its own complete-case pool."""
    counts: list[StepRowCount] = []
    for step in steps:
        valid: Series = features.loc[:, list(step.columns)].notna().all(axis=1)
        counts.append(
            StepRowCount(
                step=step.name,
                n_train=int((valid & train_universe).sum()),
                n_holdout=int((valid & holdout_universe).sum()),
            )
        )
    return tuple(counts)


def build_matched_rows(df: DataFrame, target: str) -> MatchedRows:
    """Build row-matched sweep data for a target column.

    Args:
        df: Canonical one-row-per-game modeling frame.
        target: Target column (``HOME_WIN`` or ``ACTUAL_TOTAL``). Rows with a
            null target (for example ties, or unplayed games) are excluded.

    Returns:
        ``MatchedRows`` with identical train and holdout games for every step.

    Raises:
        ValueError: If the target is missing or either split is empty.
    """
    if target not in df.columns:
        msg: str = f"Modeling data is missing target column: {target}"
        raise ValueError(msg)

    frame: DataFrame = (
        df.dropna(subset=[target]).sort_values(_SORT_KEYS, kind="stable", ignore_index=True).copy()
    )
    features: DataFrame = _make_expanded_features(frame)
    train_universe, holdout_universe = _chronological_split_masks(frame["YEAR"], HOLDOUT_SEASONS)

    steps: list[FeatureStep] = build_sweep_steps("expanded")
    widest: FeatureStep = steps[-1]
    matched: Series = features.loc[:, list(widest.columns)].notna().all(axis=1)

    train_mask: Series = matched & train_universe
    holdout_mask: Series = matched & holdout_universe
    if not bool(train_mask.any()) or not bool(holdout_mask.any()):
        msg = f"Matched rows for {target} have an empty train or holdout split"
        raise ValueError(msg)

    return MatchedRows(
        frame=frame,
        features=features,
        y=pd.Series(frame[target]).astype(float),
        matched=matched,
        train_mask=train_mask,
        holdout_mask=holdout_mask,
        universe_train=int(train_universe.sum()),
        universe_holdout=int(holdout_universe.sum()),
        own_pool=step_row_loss(features, steps, train_universe, holdout_universe),
    )
