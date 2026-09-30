# src/gridiron_edge/evaluation/feature_sweep_gates.py

"""Adopt/reject bars for the consolidated feature sweep.

Pure decision logic over metrics and bootstrap deltas from
``feature_sweep_metrics``. The bars are recorded in ``PLAN.md`` and were
locked before any sweep result existed; thresholds live here as one
frozen dataclass per target so a test pins them. A family's verdict is
whether a step clears the bar against the step before it; a batch decision
combines the family verdicts (any pass and no harm) and the caller's
live-availability finding for scheduled games.

Public API
----------
WinBar, TotalBar                threshold dataclasses
FamilyVerdict                   one family's outcome for one batch
evaluate_win_family             Win bar
evaluate_total_family           Total bar
BatchDecision, decide_batch     cross-family attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from gridiron_edge.evaluation.feature_sweep_metrics import (
    BootstrapDelta,
    TotalMetrics,
    WinMetrics,
)


@dataclass(frozen=True)
class WinBar:
    """Win acceptance bar (Brier is a loss; deltas are improvements)."""

    min_brier_improvement: float = 0.002
    max_ece_worsening: float = 0.01
    min_seasons_improved: int = 2
    harm_brier: float = 0.001


@dataclass(frozen=True)
class TotalBar:
    """Total acceptance bar."""

    min_mae_improvement: float = 0.09
    max_bias_shift: float = 0.25
    min_seasons_improved: int = 2
    harm_mae: float = 0.05


DEFAULT_WIN_BAR: Final[WinBar] = WinBar()
DEFAULT_TOTAL_BAR: Final[TotalBar] = TotalBar()


@dataclass(frozen=True)
class FamilyVerdict:
    """One model family's outcome for one batch.

    Attributes:
        family: Model family label (for example ``"win/logistic"``).
        criteria: Per-criterion pass flags.
        passed: True when every criterion passed.
        harmed: True when the whole interval says the batch made it worse by
            more than the harm threshold.
        improvement: Point improvement on the primary loss.
        p_no_improvement: One-sided bootstrap p for Holm adjustment.
    """

    family: str
    criteria: dict[str, bool]
    passed: bool
    harmed: bool
    improvement: float
    p_no_improvement: float


def _seasons_improved(season_deltas: dict[str, float]) -> int:
    return sum(1 for value in season_deltas.values() if value > 0.0)


def evaluate_win_family(
    family: str,
    previous: WinMetrics,
    new: WinMetrics,
    delta: BootstrapDelta,
    season_deltas: dict[str, float],
    *,
    bar: WinBar = DEFAULT_WIN_BAR,
    noise_floor: float | None = None,
) -> FamilyVerdict:
    """Apply the Win bar.

    Args:
        family: Family label.
        previous: Metrics of the step before the batch.
        new: Metrics of the step including the batch.
        delta: Paired-bootstrap Brier improvement.
        season_deltas: Per-season Brier improvements.
        bar: Thresholds.
        noise_floor: Placebo/seed spread the improvement must exceed; ``None``
            skips the criterion.
    """
    criteria: dict[str, bool] = {
        "brier_improvement": delta.improvement >= bar.min_brier_improvement,
        "bootstrap_lower_above_zero": delta.lower > 0.0,
        "seasons_improved": _seasons_improved(season_deltas) >= bar.min_seasons_improved,
        "ece_not_worse": (new.ece - previous.ece) <= bar.max_ece_worsening,
        "log_loss_not_worse": new.log_loss <= previous.log_loss,
        "exceeds_noise_floor": noise_floor is None or delta.improvement > noise_floor,
    }
    return FamilyVerdict(
        family=family,
        criteria=criteria,
        passed=all(criteria.values()),
        harmed=delta.upper < -bar.harm_brier,
        improvement=delta.improvement,
        p_no_improvement=delta.p_no_improvement,
    )


def evaluate_total_family(
    family: str,
    previous: TotalMetrics,
    new: TotalMetrics,
    delta: BootstrapDelta,
    season_deltas: dict[str, float],
    *,
    bar: TotalBar = DEFAULT_TOTAL_BAR,
    noise_floor: float | None = None,
) -> FamilyVerdict:
    """Apply the Total bar (MAE is the primary loss; see ``evaluate_win_family``)."""
    criteria: dict[str, bool] = {
        "mae_improvement": delta.improvement >= bar.min_mae_improvement,
        "bootstrap_lower_above_zero": delta.lower > 0.0,
        "seasons_improved": _seasons_improved(season_deltas) >= bar.min_seasons_improved,
        "rmse_not_worse": new.rmse <= previous.rmse,
        "bias_shift_within_limit": abs(new.bias - previous.bias) <= bar.max_bias_shift,
        "exceeds_noise_floor": noise_floor is None or delta.improvement > noise_floor,
    }
    return FamilyVerdict(
        family=family,
        criteria=criteria,
        passed=all(criteria.values()),
        harmed=delta.upper < -bar.harm_mae,
        improvement=delta.improvement,
        p_no_improvement=delta.p_no_improvement,
    )


BatchStatus = Literal["adopt", "reject", "adopt_only_if_fixed"]


@dataclass(frozen=True)
class BatchDecision:
    """Cross-family call for one batch.

    Attributes:
        batch: Batch name.
        status: ``adopt``, ``reject`` or ``adopt_only_if_fixed``.
        reason: Human-readable explanation.
        passing: Families that cleared the bar.
        harmed: Families with significant harm.
    """

    batch: str
    status: BatchStatus
    reason: str
    passing: tuple[str, ...]
    harmed: tuple[str, ...]


def decide_batch(
    batch: str,
    verdicts: list[FamilyVerdict],
    *,
    live_complete: bool,
) -> BatchDecision:
    """Keep a batch if any family passes and none is significantly harmed.

    A batch that would be kept but leaves scheduled games without a value
    (any NaN disables the family for that week) is ``adopt_only_if_fixed``.
    """
    if not verdicts:
        msg = "decide_batch requires at least one family verdict"
        raise ValueError(msg)
    passing = tuple(v.family for v in verdicts if v.passed)
    harmed = tuple(v.family for v in verdicts if v.harmed)
    if not passing:
        return BatchDecision(batch, "reject", "no family cleared the bar", passing, harmed)
    if harmed:
        return BatchDecision(
            batch,
            "reject",
            f"cleared the bar for {', '.join(passing)} but significantly harmed "
            f"{', '.join(harmed)}",
            passing,
            harmed,
        )
    if not live_complete:
        return BatchDecision(
            batch,
            "adopt_only_if_fixed",
            f"cleared the bar for {', '.join(passing)} but leaves some scheduled games "
            "without a value (a NaN disables the family for that week)",
            passing,
            harmed,
        )
    return BatchDecision(
        batch, "adopt", f"cleared the bar for {', '.join(passing)}; no harm", passing, harmed
    )
