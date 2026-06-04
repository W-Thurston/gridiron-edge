# src/gridiron_edge/evaluation/champion.py

"""Champion/challenger model comparison and promotion logic.

Provides a gate-based promotion system: a challenger model must
meaningfully improve on Brier score while not degrading calibration
(ECE) or discrimination (AUC) beyond tolerance thresholds.

Usage::

    from gridiron_edge.evaluation.champion import compare_models, format_comparison

    result = compare_models(champion_meta, challenger_meta)
    print(format_comparison(result))
    if result.should_promote:
        store.save("random_forest", challenger_obj, metadata=challenger_meta)

Public API:

    PromotionCriteria   Gate thresholds for promotion decisions
    ComparisonResult    Full comparison outcome with per-gate results
    extract_metrics     Pull standardised metric dict from ModelMetadata
    compare_models      Run all gates and return ComparisonResult
    format_comparison   Human-readable comparison report string
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import ModelMetadata


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionCriteria:
    """Gate thresholds for champion/challenger promotion decisions.

    Attributes:
        min_brier_improvement: Challenger Brier must be at least this much
            lower than champion Brier to pass the primary gate.
        max_ece_degradation: Challenger ECE may be at most this much higher
            than champion ECE without failing the calibration gate.
        max_auc_degradation: Challenger AUC may be at most this much lower
            than champion AUC without failing the discrimination gate.
    """

    min_brier_improvement: float = 0.002
    max_ece_degradation: float = 0.01
    max_auc_degradation: float = 0.01


@dataclass(frozen=True)
class ComparisonResult:
    """Full outcome of a champion vs challenger comparison.

    Attributes:
        champion_metrics: Standardised metric dict for the champion.
        challenger_metrics: Standardised metric dict for the challenger.
        gates: Per-gate pass/fail results.
        should_promote: True if all gates passed.
        reason: Human-readable explanation of the verdict.
    """

    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    gates: dict[str, bool]
    should_promote: bool
    reason: str


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

_PARAM_KEYS: dict[str, str] = {
    "ece": "holdout_ece",
    "auc": "holdout_auc",
    "log_loss": "holdout_log_loss",
    "accuracy": "holdout_accuracy",
}


def extract_metrics(meta: ModelMetadata) -> dict[str, float]:
    """Pull a standardised metric dict from ModelMetadata.

    Brier comes from the top-level ``holdout_brier`` field.  All other
    metrics are read from the ``parameters`` dict and default to NaN
    if absent.

    Args:
        meta: Trained model metadata.

    Returns:
        Dict with keys ``brier``, ``ece``, ``auc``, ``log_loss``,
        ``accuracy``.
    """
    metrics: dict[str, float] = {"brier": meta.holdout_brier}
    for short_name, param_key in _PARAM_KEYS.items():
        metrics[short_name] = meta.parameters.get(param_key, float("nan"))
    return metrics


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_models(
    champion: ModelMetadata,
    challenger: ModelMetadata,
    criteria: PromotionCriteria | None = None,
) -> ComparisonResult:
    """Compare champion and challenger on all gates.

    Gate 1 (Brier): challenger must improve by at least
    ``criteria.min_brier_improvement``.

    Gate 2 (ECE): challenger ECE must not degrade by more than
    ``criteria.max_ece_degradation``.  Skipped (passes) if either
    model lacks ECE data.

    Gate 3 (AUC): challenger AUC must not degrade by more than
    ``criteria.max_auc_degradation``.  Skipped (passes) if either
    model lacks AUC data.

    Args:
        champion: Metadata for the current champion model.
        challenger: Metadata for the candidate challenger model.
        criteria: Gate thresholds.  Defaults to ``PromotionCriteria()``.

    Returns:
        ``ComparisonResult`` with per-gate outcomes and overall verdict.
    """
    if criteria is None:
        criteria = PromotionCriteria()

    champ: dict[str, float] = extract_metrics(champion)
    chall: dict[str, float] = extract_metrics(challenger)

    # Gate 1: Brier must meaningfully improve (lower is better)
    brier_delta: float = champ["brier"] - chall["brier"]
    brier_improved: bool = brier_delta >= criteria.min_brier_improvement

    # Gate 2: ECE must not degrade significantly (lower is better)
    if isnan(champ["ece"]) or isnan(chall["ece"]):
        ece_acceptable = True
    else:
        ece_acceptable: bool = chall["ece"] < champ["ece"] + criteria.max_ece_degradation

    # Gate 3: AUC must not degrade significantly (higher is better)
    if isnan(champ["auc"]) or isnan(chall["auc"]):
        auc_acceptable = True
    else:
        auc_acceptable: bool = chall["auc"] > champ["auc"] - criteria.max_auc_degradation

    gates: dict[str, bool] = {
        "brier_improved": brier_improved,
        "ece_acceptable": ece_acceptable,
        "auc_acceptable": auc_acceptable,
    }
    should_promote: bool = all(gates.values())

    # Build reason string
    if should_promote:
        reason: str = f"Challenger passes all gates (Brier improved by {brier_delta:.5f})"
    else:
        failed: list[str] = [name for name, passed in gates.items() if not passed]
        reason = f"Challenger failed {len(failed)} gate(s): {', '.join(failed)}"

    return ComparisonResult(
        champion_metrics=champ,
        challenger_metrics=chall,
        gates=gates,
        should_promote=should_promote,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_METRIC_LABELS: list[tuple[str, str, bool]] = [
    ("brier", "Brier", True),
    ("ece", "ECE", True),
    ("auc", "AUC", True),
    ("log_loss", "Log Loss", False),
    ("accuracy", "Accuracy", False),
]

_GATE_NAMES: dict[str, str] = {
    "brier": "brier_improved",
    "ece": "ece_acceptable",
    "auc": "auc_acceptable",
}


def format_comparison(result: ComparisonResult) -> str:
    """Build a human-readable comparison report.

    Args:
        result: Output of ``compare_models()``.

    Returns:
        Multi-line formatted string showing metric table and verdict.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("  === Champion vs Challenger ===")
    lines.append("")
    lines.append(
        f"  {'Metric':<14s}  {'Champion':>10s}  {'Challenger':>10s}  {'Delta':>10s}  {'Gate'}"
    )
    lines.append(f"  {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    for key, label, is_gated in _METRIC_LABELS:
        c_val: float = result.champion_metrics.get(key, float("nan"))
        r_val: float = result.challenger_metrics.get(key, float("nan"))

        if isnan(c_val) or isnan(r_val):
            c_str: str = "n/a" if isnan(c_val) else f"{c_val:.5f}"
            r_str: str = "n/a" if isnan(r_val) else f"{r_val:.5f}"
            d_str = "n/a"
            gate_str = "(skipped)" if is_gated else "(info only)"
        else:
            c_str = f"{c_val:.5f}"
            r_str = f"{r_val:.5f}"
            delta: float = r_val - c_val
            sign: Literal["", "+"] = "+" if delta >= 0 else ""
            d_str: str = f"{sign}{delta:.5f}"

            if is_gated:
                gate_key: str = _GATE_NAMES[key]
                passed: bool = result.gates[gate_key]
                gate_str = "✅" if passed else "❌"
            else:
                gate_str = "(info only)"

        lines.append(f"  {label:<14s}  {c_str:>10s}  {r_str:>10s}  {d_str:>10s}  {gate_str}")

    lines.append("")
    verdict: Literal["✅ PROMOTE", "❌ REJECT"] = (
        "✅ PROMOTE" if result.should_promote else "❌ REJECT"
    )
    lines.append(f"  VERDICT: {verdict} — {result.reason}")
    lines.append("")
    return "\n".join(lines)
