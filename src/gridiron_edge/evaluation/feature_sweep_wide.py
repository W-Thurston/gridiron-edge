# src/gridiron_edge/evaluation/feature_sweep_wide.py

"""Wide walk-forward evaluation of frozen feature subsets.

The 3-season holdout has too few games to resolve differences the size of the
feature effects, so the *selection procedure* is also validated on a wider
window. To keep that clean, the subset is selected using only seasons before
an early cutoff and scored walk-forward on the seasons from the cutoff
onward, which its selection never saw. The deployable subset is selected
separately on the full pre-holdout training rows and confirmed once on the
holdout by the ordinary sweep machinery.

Arms (for example baseline, full, selected) share one frozen set of
hyperparameters per arm and are refit expanding through each evaluation
season on the same matched games. Non-inferiority is judged on the paired
bootstrap: the arm is non-inferior to a reference when the lower end of its
improvement interval is above ``-margin``.

Public API
----------
WideArm                    named column set with frozen hyperparameters
WideComparison             one arm-vs-reference outcome
SELECTION_CUTOFF_START     early selection cutoff (season start year)
train_mask_before          matched rows before a season
wide_predictions           arms -> expanding predictions
compare_non_inferiority    arm vs reference on the wide window
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from pandas import Series

from gridiron_edge.evaluation.feature_sweep_ablation import REMOVABLE_HARM_LIMIT
from gridiron_edge.evaluation.feature_sweep_fit import (
    FamilySpec,
    FeatureProvider,
    SearchOutcome,
    expanding_holdout_predictions,
)
from gridiron_edge.evaluation.feature_sweep_metrics import (
    BootstrapDelta,
    absolute_losses,
    brier_losses,
    paired_bootstrap_improvement,
    season_improvements,
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows

#: Seasons before this start year are the only ones an early selection may see.
SELECTION_CUTOFF_START: Final[int] = 2013
#: Last season start year scored (the newest completed holdout season).
LAST_EVAL_START: Final[int] = 2025


@dataclass(frozen=True)
class WideArm:
    """A named column set with its frozen hyperparameters."""

    name: str
    columns: tuple[str, ...]
    outcome: SearchOutcome


@dataclass(frozen=True)
class WideComparison:
    """An arm scored against a reference on the wide window.

    Attributes:
        arm: Arm name.
        reference: Reference arm name.
        n_games: Paired games scored.
        delta: Bootstrap improvement of the arm over the reference (positive
            means the arm is better).
        season_deltas: Per-season improvement.
        margin: Non-inferiority margin (loss units).
        non_inferior: True when the interval's lower end is above ``-margin``.
    """

    arm: str
    reference: str
    n_games: int
    delta: BootstrapDelta
    season_deltas: dict[str, float]
    margin: float
    non_inferior: bool


def _season_start(rows: MatchedRows) -> Series:
    return rows.frame["YEAR"].astype(str).str[:4].astype(int)


def train_mask_before(rows: MatchedRows, start: int) -> Series:
    """Matched rows from seasons strictly before ``start``."""
    return rows.matched & (_season_start(rows) < start)


def eval_starts(first: int = SELECTION_CUTOFF_START, last: int = LAST_EVAL_START) -> list[int]:
    """Season start years scored on the wide window, ascending."""
    return list(range(first, last + 1))


def wide_predictions(
    spec: FamilySpec,
    rows: MatchedRows,
    provider: FeatureProvider,
    arms: list[WideArm],
    *,
    starts: list[int] | None = None,
) -> dict[str, Series]:
    """Refit each arm expanding through the wide window's seasons.

    Args:
        spec: Family.
        rows: Matched rows for the family's target.
        provider: EPA-window feature provider.
        arms: Arms to score; hyperparameters are frozen, never re-searched.
        starts: Evaluation season start years; defaults to the wide window.

    Returns:
        Arm name -> predictions indexed like the matched frame (NaN outside
        the evaluated seasons).
    """
    window = eval_starts() if starts is None else starts
    season = _season_start(rows)
    return {
        arm.name: expanding_holdout_predictions(
            spec,
            arm.outcome,
            x_all=provider(arm.outcome.epa_window).loc[:, list(arm.columns)],
            y=rows.y,
            season_start=season,
            matched=rows.matched,
            holdout_seasons=window,
        )
        for arm in arms
    }


def compare_non_inferiority(
    spec: FamilySpec,
    rows: MatchedRows,
    predictions: dict[str, Series],
    arm: str,
    reference: str,
    *,
    starts: list[int] | None = None,
    margin: float | None = None,
) -> WideComparison:
    """Score ``arm`` against ``reference`` on the wide window's matched games.

    Args:
        spec: Family.
        rows: Matched rows.
        predictions: Output of ``wide_predictions``.
        arm: Arm under test.
        reference: Reference arm.
        starts: Season start years scored; defaults to the wide window.
        margin: Non-inferiority margin; defaults to the ablation harm limit
            for the target (0.001 Brier, 0.045 MAE).
    """
    window = eval_starts() if starts is None else starts
    mask = rows.matched & _season_start(rows).isin(window)
    y = rows.y[mask].to_numpy(dtype=float)
    loss = brier_losses if spec.target == "win" else absolute_losses
    a = loss(predictions[arm][mask].to_numpy(dtype=float), y)
    r = loss(predictions[reference][mask].to_numpy(dtype=float), y)
    if np.isnan(a).any() or np.isnan(r).any():
        msg = "Wide-window predictions contain NaN"
        raise ValueError(msg)
    limit = REMOVABLE_HARM_LIMIT[spec.target] if margin is None else margin
    delta = paired_bootstrap_improvement(r, a)
    seasons = rows.frame.loc[mask, "YEAR"].astype(str)
    return WideComparison(
        arm=arm,
        reference=reference,
        n_games=int(mask.sum()),
        delta=delta,
        season_deltas=season_improvements(r, a, seasons),
        margin=limit,
        non_inferior=delta.lower > -limit,
    )
