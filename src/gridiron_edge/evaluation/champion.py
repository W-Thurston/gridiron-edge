# src/gridiron_edge/evaluation/champion.py

"""Champion/challenger model comparison and promotion logic.

Classification:
    ClassificationPromotionGates           Gate thresholds (Brier/ECE/AUC)
    ClassificationComparisonResult         Comparison outcome
    extract_classification_metrics         Pull metrics from GameModelMetadata
    compare_classification_models          Run classification gates
    format_classification_comparison       Human-readable classification report

Regression:
    RegressionPromotionGates               Gate thresholds (R²/Coverage)
    RegressionModelResult                  Standardised regression metrics
    RegressionComparisonResult             Comparison outcome
    compare_regression_models              Run regression gates
    select_prop_champion                   Pick best model from multiple results
    format_regression_comparison           Human-readable regression report

Metric storage:
    Classification metrics (ECE, AUC, log_loss, accuracy) live on the
    ``metrics`` dict on :class:`BaseModelMetadata`, populated by
    ``GamesTrainer._build_classification_metadata``.
    ``extract_classification_metrics`` reads from that dict.

Usage::

    from gridiron_edge.evaluation.champion import (
        compare_classification_models,
        format_classification_comparison,
    )

    result = compare_classification_models(champion_meta, challenger_meta)
    print(format_classification_comparison(result))
    if result.should_promote:
        store.save(metadata=challenger_meta, model_obj=challenger_obj)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gridiron_edge.models.game_prediction.base import GameModelMetadata


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationPromotionGates:
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
class RegressionPromotionGates:
    """Gate thresholds for regression model promotion decisions.

    Attributes:
        min_r2: Minimum holdout R² to pass. Must beat mean baseline.
        min_coverage: Lower bound for 90% prediction interval coverage.
        max_coverage: Upper bound for 90% prediction interval coverage.
    """

    min_r2: float = 0.0
    min_coverage: float = 0.85
    max_coverage: float = 0.97


@dataclass(frozen=True)
class ClassificationComparisonResult:
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


def extract_classification_metrics(meta: GameModelMetadata) -> dict[str, float]:
    """Pull a standardised metric dict from GameModelMetadata.

    Reads the classification metrics from the ``metrics`` dict on
    :class:`BaseModelMetadata` (Unit 9). Any metric not recorded at
    training time surfaces as NaN so the comparator gates can treat
    them uniformly.

    Args:
        meta: Trained model metadata.

    Returns:
        Dict with keys ``brier``, ``ece``, ``auc``, ``log_loss``,
        ``accuracy``.
    """
    return {
        "brier": meta.metrics.get("brier", float("nan")),
        "ece": meta.metrics.get("ece", float("nan")),
        "auc": meta.metrics.get("auc", float("nan")),
        "log_loss": meta.metrics.get("log_loss", float("nan")),
        "accuracy": meta.metrics.get("accuracy", float("nan")),
    }


@dataclass(frozen=True)
class RegressionModelResult:
    """Standardised regression metrics for champion comparison.

    Attributes:
        model_type: Algorithm identifier (e.g. "elasticnet", "random_forest").
        mae: Mean absolute error on holdout.
        rmse: Root mean squared error on holdout.
        r2: R-squared on holdout.
        coverage: Actual coverage of 90% prediction interval, or NaN if unavailable.
    """

    model_type: str
    mae: float
    rmse: float
    r2: float
    coverage: float = float("nan")


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_classification_models(
    champion: GameModelMetadata,
    challenger: GameModelMetadata,
    criteria: ClassificationPromotionGates | None = None,
) -> ClassificationComparisonResult:
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
        criteria: Gate thresholds.  Defaults to ``ClassificationPromotionGates()``.

    Returns:
        ``ClassificationComparisonResult`` with per-gate outcomes and overall verdict.
    """
    if criteria is None:
        criteria = ClassificationPromotionGates()

    champ: dict[str, float] = extract_classification_metrics(champion)
    chall: dict[str, float] = extract_classification_metrics(challenger)

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

    return ClassificationComparisonResult(
        champion_metrics=champ,
        challenger_metrics=chall,
        gates=gates,
        should_promote=should_promote,
        reason=reason,
    )


@dataclass(frozen=True)
class RegressionComparisonResult:
    """Outcome of a regression champion vs challenger comparison.

    Attributes:
        champion: Metrics for the current champion.
        challenger: Metrics for the candidate challenger.
        gates: Per-gate pass/fail for the challenger.
        should_promote: True if challenger passes all gates AND beats champion MAE.
        reason: Human-readable explanation.
    """

    champion: RegressionModelResult
    challenger: RegressionModelResult
    gates: dict[str, bool]
    should_promote: bool
    reason: str


def compare_regression_models(
    champion: RegressionModelResult,
    challenger: RegressionModelResult,
    gates: RegressionPromotionGates | None = None,
) -> RegressionComparisonResult:
    """Compare champion and challenger regression models on all gates.

    Gate 1 (R²): challenger R² must exceed ``gates.min_r2``.
    Gate 2 (Coverage low): challenger coverage must be ≥ ``gates.min_coverage``.
        Skipped if coverage is NaN.
    Gate 3 (Coverage high): challenger coverage must be ≤ ``gates.max_coverage``.
        Skipped if coverage is NaN.
    Gate 4 (MAE): challenger MAE must be lower than champion MAE.

    Args:
        champion: Metrics for the current champion.
        challenger: Metrics for the candidate.
        gates: Gate thresholds. Defaults to ``RegressionPromotionGates()``.

    Returns:
        ``RegressionComparisonResult`` with per-gate outcomes and verdict.
    """
    if gates is None:
        gates = RegressionPromotionGates()

    # Gate 1: R² > min_r2
    r2_pass: bool = challenger.r2 > gates.min_r2

    # Gate 2 & 3: Coverage within [min, max] — skip if NaN
    if isnan(challenger.coverage):
        coverage_low_pass: bool = True
        coverage_high_pass: bool = True
    else:
        coverage_low_pass = challenger.coverage >= gates.min_coverage
        coverage_high_pass = challenger.coverage <= gates.max_coverage

    # Gate 4: MAE must beat champion (lower is better)
    mae_pass: bool = challenger.mae < champion.mae

    gate_results: dict[str, bool] = {
        "r2_above_zero": r2_pass,
        "coverage_in_range_low": coverage_low_pass,
        "coverage_in_range_high": coverage_high_pass,
        "mae_improved": mae_pass,
    }

    should_promote: bool = all(gate_results.values())

    if should_promote:
        delta: float = champion.mae - challenger.mae
        reason: str = (
            f"Challenger ({challenger.model_type}) passes all gates (MAE improved by {delta:.2f})"
        )
    else:
        failed: list[str] = [k for k, v in gate_results.items() if not v]
        reason = (
            f"Challenger ({challenger.model_type}) failed "
            f"{len(failed)} gate(s): {', '.join(failed)}"
        )

    return RegressionComparisonResult(
        champion=champion,
        challenger=challenger,
        gates=gate_results,
        should_promote=should_promote,
        reason=reason,
    )


def select_prop_champion(
    results: list[RegressionModelResult],
    gates: RegressionPromotionGates | None = None,
) -> tuple[RegressionModelResult, str]:
    """Select the best prop model from a list of trained results.

    Logic:
    1. Filter to models passing all guardrails (R² > 0, coverage in range).
    2. Among eligible models, select the one with lowest MAE.
    3. If no model passes guardrails, return the ElasticNet result as
       fallback (Decision #11: known stable baseline).

    Args:
        results: List of regression model results to compare.
        gates: Gate thresholds. Defaults to ``RegressionPromotionGates()``.

    Returns:
        Tuple of (champion result, summary string).

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        msg = "Cannot select champion from empty results list."
        raise ValueError(msg)

    if gates is None:
        gates = RegressionPromotionGates()

    # Evaluate guardrails for each model
    eligible: list[RegressionModelResult] = []
    for r in results:
        r2_ok: bool = r.r2 > gates.min_r2

        if isnan(r.coverage):
            coverage_ok: bool = True
        else:
            coverage_ok = gates.min_coverage <= r.coverage <= gates.max_coverage

        if r2_ok and coverage_ok:
            eligible.append(r)

    # Build summary table
    lines: list[str] = []
    lines.append("")
    lines.append("  === Prop Champion Selection ===")
    lines.append("")
    lines.append(
        f"  {'Model Type':<16s}  {'MAE':>8s}  {'RMSE':>8s}"
        f"  {'R²':>8s}  {'Coverage':>10s}  {'Eligible'}"
    )
    lines.append(f"  {'─' * 16}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}")

    for r in results:
        is_eligible: bool = r in eligible
        cov_str: str = f"{r.coverage:.1%}" if not isnan(r.coverage) else "n/a"
        lines.append(
            f"  {r.model_type:<16s}  {r.mae:>8.2f}  {r.rmse:>8.2f}"
            f"  {r.r2:>8.3f}  {cov_str:>10s}  {'✅' if is_eligible else '❌'}"
        )

    if eligible:
        champion: RegressionModelResult = min(eligible, key=lambda r: r.mae)
        lines.append("")
        lines.append(f"  🏆 Champion: {champion.model_type} (MAE={champion.mae:.2f})")
    else:
        # Fallback to ElasticNet
        elasticnet_results: list[RegressionModelResult] = [
            r for r in results if r.model_type == "elasticnet"
        ]
        if elasticnet_results:
            champion = elasticnet_results[0]
        else:
            # If no ElasticNet either, just pick lowest MAE as last resort
            champion = min(results, key=lambda r: r.mae)
        lines.append("")
        lines.append(
            f"  ⚠️  No model passed all guardrails. "
            f"Fallback: {champion.model_type} (MAE={champion.mae:.2f})"
        )

    lines.append("")
    summary: str = "\n".join(lines)

    return champion, summary


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


def format_classification_comparison(result: ClassificationComparisonResult) -> str:
    """Build a human-readable comparison report.

    Args:
        result: Output of ``compare_classification_models()``.

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


def format_regression_comparison(result: RegressionComparisonResult) -> str:
    """Build a human-readable regression comparison report.

    Args:
        result: Output of ``compare_regression_models()``.

    Returns:
        Multi-line formatted string showing metric table and verdict.
    """
    c: RegressionModelResult = result.champion
    r: RegressionModelResult = result.challenger

    lines: list[str] = []
    lines.append("")
    lines.append(f"  === {c.model_type} (champion) vs {r.model_type} (challenger) ===")
    lines.append("")
    lines.append(
        f"  {'Metric':<14s}  {'Champion':>10s}  {'Challenger':>10s}  {'Delta':>10s}  {'Gate'}"
    )
    lines.append(f"  {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    # MAE (lower is better)
    mae_delta: float = r.mae - c.mae
    mae_gate: str = "✅" if result.gates["mae_improved"] else "❌"
    lines.append(f"  {'MAE':<14s}  {c.mae:>10.2f}  {r.mae:>10.2f}  {mae_delta:>+10.2f}  {mae_gate}")

    # RMSE (info only)
    rmse_delta: float = r.rmse - c.rmse
    lines.append(
        f"  {'RMSE':<14s}  {c.rmse:>10.2f}  {r.rmse:>10.2f}  {rmse_delta:>+10.2f}  (info only)"
    )

    # R² (higher is better)
    r2_delta: float = r.r2 - c.r2
    r2_gate: str = "✅" if result.gates["r2_above_zero"] else "❌"
    lines.append(f"  {'R²':<14s}  {c.r2:>10.3f}  {r.r2:>10.3f}  {r2_delta:>+10.3f}  {r2_gate}")

    # Coverage
    c_cov: str = f"{c.coverage:.1%}" if not isnan(c.coverage) else "n/a"
    r_cov: str = f"{r.coverage:.1%}" if not isnan(r.coverage) else "n/a"
    if isnan(c.coverage) or isnan(r.coverage):
        cov_delta: str = "n/a"
        cov_gate = "(skipped)"
    else:
        cov_d: float = r.coverage - c.coverage
        cov_delta = f"{cov_d:>+.1%}"
        low_ok: bool = result.gates["coverage_in_range_low"]
        high_ok: bool = result.gates["coverage_in_range_high"]
        cov_gate = "✅" if (low_ok and high_ok) else "❌"
    lines.append(f"  {'Coverage':<14s}  {c_cov:>10s}  {r_cov:>10s}  {cov_delta:>10s}  {cov_gate}")

    lines.append("")
    verdict: str = "✅ PROMOTE" if result.should_promote else "❌ REJECT"
    lines.append(f"  VERDICT: {verdict} — {result.reason}")
    lines.append("")

    return "\n".join(lines)
