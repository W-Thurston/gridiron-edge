# src/gridiron_edge/evaluation/feature_sweep_confirm.py

"""Stage 4 of the feature sweep: confirm the final composition against baseline.

The final feature set (baseline plus the kept batch) is scored against
baseline directly, because the batch was attributed with rejected batches
present. To stay cheap it reuses the sweep's frozen hyperparameters (the
baseline step's for the baseline arm, the kept batch's sweep step for the
final arm) and does refits only, on the regenerated matched games, which now
include season openers. It reports the wide window (2013-2025), the holdout
(2023-2025) and a Week 1 slice. The criterion was recorded in ``PLAN.md``
before it ran: confirmed when no family is significantly worse than baseline
on either window and at least 4 of 5 wide-window point improvements are
positive. Read-only; nothing is persisted.

Public API
----------
FamilyConfirmation, ConfirmationVerdict
run_confirmation, judge_confirmation, build_confirmation_report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_run import FamilyRun, make_feature_provider
from gridiron_edge.evaluation.feature_sweep_stage3 import HARM
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.evaluation.feature_sweep_wide import (
    WideArm,
    WideComparison,
    compare_non_inferiority,
    eval_starts,
    wide_predictions,
)

KEPT_BATCH: Final[str] = "qb_and_coach"
MIN_POSITIVE_FAMILIES: Final[int] = 4


@dataclass
class FamilyConfirmation:
    """Final-composition versus baseline for one family."""

    family: str
    target: str
    n_baseline: int
    n_final: int
    wide: WideComparison
    holdout: WideComparison
    week1: WideComparison | None


@dataclass(frozen=True)
class ConfirmationVerdict:
    """Outcome of the pre-declared criterion."""

    confirmed: bool
    positive_families: int
    harmed: tuple[str, ...]
    reason: str


def _holdout_starts() -> list[int]:
    return sorted(int(s.split("-")[0]) for s in HOLDOUT_SEASONS)


def _family(run: FamilyRun, rows: MatchedRows, repo: Path) -> FamilyConfirmation:
    spec: FamilySpec = run.spec
    steps = build_sweep_steps(spec.feature_set_key)
    baseline, final = steps[0], steps[-1]
    old = {fit.step: fit for fit in run.steps}
    arms = [
        WideArm("baseline", baseline.columns, old["baseline"].outcome),
        WideArm("final", final.columns, old[KEPT_BATCH].outcome),
    ]
    provider = make_feature_provider(rows, spec, repo)
    window = eval_starts()
    hold = _holdout_starts()
    preds = wide_predictions(spec, rows, provider, arms, starts=sorted(set(window) | set(hold)))
    week1 = rows.frame["WEEK_NUM"] == 1
    week1_cmp: WideComparison | None = None
    if bool((rows.matched & week1).any()):
        week1_cmp = compare_non_inferiority(
            spec, rows, preds, "final", "baseline", starts=window, subset=week1
        )
    return FamilyConfirmation(
        family=spec.label,
        target=spec.target,
        n_baseline=len(baseline.columns),
        n_final=len(final.columns),
        wide=compare_non_inferiority(spec, rows, preds, "final", "baseline", starts=window),
        holdout=compare_non_inferiority(spec, rows, preds, "final", "baseline", starts=hold),
        week1=week1_cmp,
    )


def run_confirmation(
    runs: dict[str, FamilyRun], matched: dict[str, MatchedRows], repo: Path
) -> dict[str, FamilyConfirmation]:
    """Score the final composition against baseline for every sweep family."""
    return {label: _family(run, matched[run.spec.target], repo) for label, run in runs.items()}


def judge_confirmation(results: dict[str, FamilyConfirmation]) -> ConfirmationVerdict:
    """Apply the pre-declared criterion."""
    harmed = tuple(
        label
        for label, r in results.items()
        for c in (r.wide, r.holdout)
        if c.delta.upper < -HARM[r.target]
    )
    positive = sum(1 for r in results.values() if r.wide.delta.improvement > 0.0)
    confirmed = not harmed and positive >= MIN_POSITIVE_FAMILIES
    if harmed:
        reason = f"significantly worse than baseline: {', '.join(sorted(set(harmed)))}"
    elif positive < MIN_POSITIVE_FAMILIES:
        reason = f"only {positive} of {len(results)} wide-window improvements are positive"
    else:
        reason = (
            f"no family worse than baseline; {positive} of {len(results)} improvements positive"
        )
    return ConfirmationVerdict(confirmed, positive, tuple(sorted(set(harmed))), reason)


def _row(name: str, c: WideComparison | None) -> str:
    if c is None:
        return f"| {name} | n/a | n/a | n/a |"
    d = c.delta
    return f"| {name} | {c.n_games} | {d.improvement:+.5f} | [{d.lower:+.5f}, {d.upper:+.5f}] |"


def build_confirmation_report(
    results: dict[str, FamilyConfirmation], verdict: ConfirmationVerdict, *, generated_on: date
) -> str:
    """Render the stage 4 Markdown report."""
    lines = [
        f"# Phase A final-composition confirmation - {generated_on.isoformat()}",
        "",
        "Stage 4 of the feature sweep (`PLAN.md`). The final feature set (baseline plus "
        "the QB-and-coach batch, schema 13) is scored against baseline directly, reusing "
        "the sweep's frozen hyperparameters (refits only) on the regenerated matched "
        "games, which include season openers. Improvement is positive when the final "
        "set is better (Brier for Win, MAE for Total). Criterion, recorded before it ran: "
        "confirmed when no family is significantly worse than baseline on the wide "
        f"window (2013-2025) or the holdout and at least {MIN_POSITIVE_FAMILIES} of "
        f"{len(results)} wide-window improvements are positive. Generated by "
        "`gridiron evaluate feature-confirmation`; a dated record, not an input to any "
        "pipeline.",
        "",
        f"## Verdict: {'CONFIRMED' if verdict.confirmed else 'NOT CONFIRMED'}",
        "",
        verdict.reason + ".",
        "",
    ]
    for label, r in results.items():
        unit = "Brier" if r.target == "win" else "MAE"
        lines += [
            f"## {label}",
            "",
            f"Columns: baseline {r.n_baseline}, final {r.n_final}. Improvement in {unit}.",
            "",
            "| Window | Games | Improvement | 90% interval |",
            "|---|---|---|---|",
            _row("Wide 2013-2025", r.wide),
            _row("Holdout 2023-2025", r.holdout),
            _row("Week 1 games, wide window", r.week1),
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"
