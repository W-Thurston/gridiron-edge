# src/gridiron_edge/evaluation/feature_sweep_stage3.py

"""Stage 3 of the feature sweep: per-batch attribution on the wide window.

The 3-season holdout is too small to resolve batch-sized effects, so each
batch is scored against the step before it walk-forward over the wide window
(2013-2025, about 3,300 matched games) using every step's own frozen sweep
hyperparameters (refits only, no new searches). The revised criterion was
recorded before results in ``PLAN.md``: point improvement at least half the
original floor, Benjamini-Hochberg adjusted one-sided p at most 0.10 across
all batch-by-family comparisons, and at least 60% of seasons improved; a
batch is rejected when nothing passes or any family is significantly harmed.
Read-only: nothing is persisted but local scratch checkpoints.

Public API
----------
BatchAttribution, FamilyAttribution   results
run_stage3                            runs -> per-family attribution
decide_stage3                         attribution -> batch decisions
build_stage3_report                   Markdown report
load_stage3                           checkpoint directory -> results
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import Path
import pickle
import time
from typing import Final

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_gates import (
    BatchDecision,
    FamilyVerdict,
    decide_batch,
)
from gridiron_edge.evaluation.feature_sweep_metrics import bh_adjust
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_run import FamilyRun, make_feature_provider
from gridiron_edge.evaluation.feature_sweep_steps import FEATURE_BATCHES, build_sweep_steps
from gridiron_edge.evaluation.feature_sweep_wide import (
    WideArm,
    WideComparison,
    compare_non_inferiority,
    eval_starts,
    wide_predictions,
)

STAGE3_SUBDIR: Final[str] = "stage3"
#: Half of the original per-batch floors (0.002 Brier, 0.09 MAE).
MIN_IMPROVEMENT: Final[dict[str, float]] = {"win": 0.001, "total": 0.045}
#: Harm: the whole 90% interval worse than this.
HARM: Final[dict[str, float]] = {"win": 0.001, "total": 0.05}
BH_Q: Final[float] = 0.10
MIN_SEASON_FRACTION: Final[float] = 0.60


@dataclass
class BatchAttribution:
    """One batch's wide-window comparison against the step before it."""

    batch: str
    comparison: WideComparison
    seasons_improved: int
    seasons_total: int


@dataclass
class FamilyAttribution:
    """Per-batch wide-window comparisons for one family."""

    family: str
    target: str
    batches: list[BatchAttribution] = field(default_factory=list)


def _path(checkpoint_dir: Path, label: str) -> Path:
    return checkpoint_dir / STAGE3_SUBDIR / f"{label.replace('/', '__')}.pkl"


def load_stage3(checkpoint_dir: Path) -> dict[str, FamilyAttribution]:
    """Load every stage 3 family result in ``checkpoint_dir``."""
    out: dict[str, FamilyAttribution] = {}
    for path in sorted((checkpoint_dir / STAGE3_SUBDIR).glob("*.pkl")):
        with path.open("rb") as handle:
            item: FamilyAttribution = pickle.load(handle)
        out[item.family] = item
    return out


def _family(run: FamilyRun, rows: MatchedRows, repo: Path, window: list[int]) -> FamilyAttribution:
    spec: FamilySpec = run.spec
    steps = build_sweep_steps(spec.feature_set_key)
    provider = make_feature_provider(rows, spec, repo)
    arms = [
        WideArm(step.name, step.columns, fit.outcome)
        for step, fit in zip(steps, run.steps, strict=True)
    ]
    preds = wide_predictions(spec, rows, provider, arms, starts=window)
    result = FamilyAttribution(spec.label, spec.target)
    for previous, current in pairwise(steps):
        cmp = compare_non_inferiority(
            spec, rows, preds, current.name, previous.name, starts=window, margin=HARM[spec.target]
        )
        improved = sum(1 for v in cmp.season_deltas.values() if v > 0.0)
        result.batches.append(BatchAttribution(current.name, cmp, improved, len(cmp.season_deltas)))
    return result


def run_stage3(
    runs: dict[str, FamilyRun],
    matched: dict[str, MatchedRows],
    repo: Path,
    checkpoint_dir: Path,
    *,
    window: list[int] | None = None,
    on_family: Callable[[str, float], None] | None = None,
) -> dict[str, FamilyAttribution]:
    """Run (or resume) stage 3 for every completed sweep family."""
    results = load_stage3(checkpoint_dir)
    for label, run in runs.items():
        if label in results:
            continue
        started = time.monotonic()
        results[label] = _family(
            run, matched[run.spec.target], repo, eval_starts() if window is None else window
        )
        path = _path(checkpoint_dir, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(results[label], handle)
        tmp.replace(path)
        if on_family is not None:
            on_family(label, time.monotonic() - started)
    return results


def decide_stage3(
    results: dict[str, FamilyAttribution], live_incomplete: dict[str, int]
) -> tuple[dict[str, BatchDecision], dict[str, float]]:
    """Apply the pre-declared criterion.

    Args:
        results: Per-family attribution.
        live_incomplete: Batch -> rows from the holdout onward with a NaN.

    Returns:
        Batch decisions and the BH-adjusted p-values keyed ``family|batch``.
    """
    raw = {
        f"{label}|{b.batch}": b.comparison.delta.p_no_improvement
        for label, fam in results.items()
        for b in fam.batches
    }
    adjusted = bh_adjust(raw)
    decisions: dict[str, BatchDecision] = {}
    for batch in FEATURE_BATCHES:
        verdicts: list[FamilyVerdict] = []
        for label, fam in results.items():
            item = next((b for b in fam.batches if b.batch == batch.name), None)
            if item is None:
                continue
            d = item.comparison.delta
            criteria = {
                "improvement_at_least_half_floor": d.improvement >= MIN_IMPROVEMENT[fam.target],
                "bh_p_within_q": adjusted[f"{label}|{batch.name}"] <= BH_Q,
                "seasons_improved": item.seasons_improved
                >= MIN_SEASON_FRACTION * item.seasons_total,
            }
            verdicts.append(
                FamilyVerdict(
                    family=label,
                    criteria=criteria,
                    passed=all(criteria.values()),
                    harmed=d.upper < -HARM[fam.target],
                    improvement=d.improvement,
                    p_no_improvement=d.p_no_improvement,
                )
            )
        if verdicts:
            decisions[batch.name] = decide_batch(
                batch.name, verdicts, live_complete=live_incomplete.get(batch.name, 0) == 0
            )
    return decisions, adjusted


def build_stage3_report(
    results: dict[str, FamilyAttribution],
    decisions: dict[str, BatchDecision],
    adjusted: dict[str, float],
    live_incomplete: dict[str, int],
    *,
    generated_on: date,
) -> str:
    """Render the stage 3 Markdown report."""
    n = sum(len(f.batches) for f in results.values())
    lines = [
        f"# Phase A per-batch attribution (wide window) - {generated_on.isoformat()}",
        "",
        "Stage 3 of the feature sweep (`PLAN.md`). Each batch is scored against the step "
        "before it, walk-forward over 2013-2025, using each step's own frozen sweep "
        "hyperparameters. The criterion was revised, before results, from the underpowered "
        "holdout bars: improvement at least half the original floor (0.001 Brier / 0.045 "
        f"MAE), Benjamini-Hochberg adjusted one-sided p at most {BH_Q} across all {n} "
        f"comparisons, and at least {MIN_SEASON_FRACTION:.0%} of seasons improved; a "
        "batch is rejected when no family passes or any family is significantly harmed. "
        "This is a data-informed revision motivated by power, not by any batch's result. "
        "Generated by `gridiron evaluate feature-attribution`; a dated record, not an input "
        "to any pipeline.",
        "",
        "## Batch decisions",
        "",
        "| Batch | Decision | Reason | Rows from holdout onward with a NaN |",
        "|---|---|---|---|",
    ]
    for name, d in decisions.items():
        lines.append(f"| {name} | **{d.status}** | {d.reason} | {live_incomplete.get(name, 0)} |")
    for label, fam in results.items():
        unit = "Brier" if fam.target == "win" else "MAE"
        lines += [
            "",
            f"## {label}",
            "",
            f"Improvement in {unit} over the previous step (positive is better).",
            "",
            "| Batch | Games | Improvement | 90% interval | Seasons improved | Raw p | BH p |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in fam.batches:
            d = b.comparison.delta
            lines.append(
                f"| {b.batch} | {b.comparison.n_games} | {d.improvement:+.5f} | "
                f"[{d.lower:+.5f}, {d.upper:+.5f}] | {b.seasons_improved}/{b.seasons_total} | "
                f"{d.p_no_improvement:.4f} | {adjusted[f'{label}|{b.batch}']:.4f} |"
            )
    return "\n".join(lines) + "\n"
