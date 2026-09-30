# src/gridiron_edge/evaluation/feature_sweep_ablation.py

"""Column-group ablations for the consolidated feature sweep.

A batch can clear its bar while carrying dead weight, so named suspect
column groups are dropped from the full model and scored on a *pre-holdout*
screening window (seasons before ``HOLDOUT_SEASONS``), never on the holdout,
so a removal decision cannot select on the holdout. Hyperparameters are the
full model's frozen search outcome. Dropping a group "hurts" by the loss
increase it causes; a group is removable when dropping it does not hurt
beyond a tolerance and the paired-bootstrap interval excludes a meaningful
harm. The holdout reading, if wanted, is reported separately and never
enters the decision.

Public API
----------
AblationGroup            named suspect group
SUSPECT_GROUPS           the pre-declared suspects
AblationResult           one (family, group) outcome
run_ablation             frozen outcome + rows -> {group: AblationResult}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from pandas import Series

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
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows

#: Pre-holdout screening seasons (start years), each predicted expanding.
SCREEN_SEASON_STARTS: Final[tuple[int, ...]] = (2019, 2020, 2021, 2022)

#: Dropping a group may cost at most this much loss and still be "removable"
#: (a fraction of the corresponding adoption bar).
REMOVABLE_TOLERANCE: Final[dict[str, float]] = {"win": 0.0005, "total": 0.02}

#: The interval's upper end for the harm of dropping must stay below this.
REMOVABLE_HARM_LIMIT: Final[dict[str, float]] = {"win": 0.001, "total": 0.045}


@dataclass(frozen=True)
class AblationGroup:
    """A named suspect group of per-team metrics (all their columns)."""

    name: str
    metrics: tuple[str, ...]


SUSPECT_GROUPS: Final[tuple[AblationGroup, ...]] = (
    AblationGroup("form_trend", ("FORM_TREND",)),
    AblationGroup("glicko2_rd", ("GLICKO2_RD",)),
    AblationGroup("glicko2_rating", ("GLICKO2_RATING",)),
    AblationGroup("linear_ratings", ("MASSEY_RATING", "COLLEY_RATING", "KEENER_RATING")),
    AblationGroup("qb_elo", ("QB_ELO", "QB_ELO_STARTS")),
    AblationGroup(
        "qb_history",
        (
            "QB_EPA_PLAY_CAREER",
            "QB_EPA_PLAY_ROLLING",
            "QB_CPOE_ROLLING",
            "QB_CPOE_ROLLING_AVAILABLE",
            "QB_EXPERIENCE",
        ),
    ),
)


@dataclass(frozen=True)
class AblationResult:
    """Outcome of dropping one group from the full model.

    Attributes:
        group: Group name.
        n_columns_dropped: Columns removed from the model.
        harm: Point loss increase from dropping (positive = dropping hurts).
        delta: Paired-bootstrap interval of that harm.
        removable: True when the harm is within tolerance and its interval's
            upper end is below the harm limit.
    """

    group: str
    n_columns_dropped: int
    harm: float
    delta: BootstrapDelta
    removable: bool


def group_columns(group: AblationGroup, columns: list[str]) -> list[str]:
    """Columns of ``columns`` belonging to ``group`` (diff, Away and Home)."""
    wanted: set[str] = set()
    for metric in group.metrics:
        wanted.update({f"{metric}_DIFF", f"AWAY_{metric}", f"HOME_{metric}"})
    return [c for c in columns if c in wanted]


def _loss(target: str, pred: Series, y: Series, mask: Series) -> np.ndarray:
    p = pred[mask].to_numpy(dtype=float)
    truth = y[mask].to_numpy(dtype=float)
    return brier_losses(p, truth) if target == "win" else absolute_losses(p, truth)


def run_ablation(
    spec: FamilySpec,
    rows: MatchedRows,
    provider: FeatureProvider,
    outcome: SearchOutcome,
    columns: list[str],
    *,
    groups: tuple[AblationGroup, ...] = SUSPECT_GROUPS,
    screen_starts: tuple[int, ...] = SCREEN_SEASON_STARTS,
) -> dict[str, AblationResult]:
    """Score dropping each group from the full model on the screening window.

    Args:
        spec: Family.
        rows: Matched rows for the family's target.
        provider: EPA-window feature provider.
        outcome: The full model's frozen search outcome (never re-searched).
        columns: The full model's feature columns.
        groups: Suspect groups to drop, one at a time.
        screen_starts: Screening season start years (all before the holdout).

    Returns:
        Results for every group that has at least one column in ``columns``.
    """
    starts = rows.frame["YEAR"].astype(str).str[:4].astype(int)
    screen_mask = rows.matched & starts.isin(screen_starts)
    x_all = provider(outcome.epa_window)

    def predict(cols: list[str]) -> Series:
        return expanding_holdout_predictions(
            spec,
            outcome,
            x_all=x_all.loc[:, cols],
            y=rows.y,
            season_start=starts,
            matched=rows.matched,
            holdout_seasons=list(screen_starts),
        )

    full_loss = _loss(spec.target, predict(columns), rows.y, screen_mask)
    tolerance = REMOVABLE_TOLERANCE[spec.target]
    limit = REMOVABLE_HARM_LIMIT[spec.target]
    results: dict[str, AblationResult] = {}
    for group in groups:
        dropped = group_columns(group, columns)
        if not dropped:
            continue
        kept = [c for c in columns if c not in set(dropped)]
        dropped_loss = _loss(spec.target, predict(kept), rows.y, screen_mask)
        delta = paired_bootstrap_improvement(dropped_loss, full_loss)
        results[group.name] = AblationResult(
            group=group.name,
            n_columns_dropped=len(dropped),
            harm=delta.improvement,
            delta=delta,
            removable=delta.improvement <= tolerance and delta.upper < limit,
        )
    return results
