# src/gridiron_edge/evaluation/feature_sweep_stage2.py

"""Stage 2 of the feature sweep: model-driven feature selection.

For each family this reuses the sweep's frozen full-set hyperparameters and
runs two selections on the full (final-step) column pool:

- an *early* selection on matched rows before ``early_cutoff`` (2013), scored
  walk-forward on the seasons from the cutoff onward, which it never saw, to
  validate the procedure on a wide window; and
- a *final* selection on all matched pre-holdout rows, the deployable subset,
  confirmed once on the holdout.

Each selected subset is compared with the full set and the baseline by paired
bootstrap non-inferiority. Read-only: nothing is persisted except local
scratch checkpoints in a caller-chosen directory, and no champion or
prediction artifact is touched.

Public API
----------
FamilyStage2, run_stage2, build_stage2_report, load_stage2
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import pickle
from typing import Any, Final

from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_run import FamilyRun, make_feature_provider
from gridiron_edge.evaluation.feature_sweep_selection import (
    SelectionResult,
    select_logistic,
    select_tree,
)
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.evaluation.feature_sweep_wide import (
    LAST_EVAL_START,
    SELECTION_CUTOFF_START,
    WideArm,
    WideComparison,
    compare_non_inferiority,
    eval_starts,
    train_mask_before,
    wide_predictions,
)

STAGE2_SUBDIR: Final[str] = "stage2"


@dataclass
class FamilyStage2:
    """Stage 2 results for one family.

    Attributes:
        family: Family label.
        n_baseline: Baseline column count.
        n_full: Full-set column count.
        early: Early-cutoff selection.
        final: Final (deployable) selection.
        wide: Comparisons of the early subset on the wide window, keyed
            ``selected_vs_full``, ``selected_vs_baseline``, ``full_vs_baseline``.
        holdout: Comparisons of the final subset on the holdout, keyed the
            same way.
    """

    family: str
    n_baseline: int
    n_full: int
    early: SelectionResult
    final: SelectionResult
    wide: dict[str, WideComparison] = field(default_factory=dict)
    holdout: dict[str, WideComparison] = field(default_factory=dict)


def _select(
    run: FamilyRun,
    x_pool: DataFrame,
    y: Series,
    mask: Series,
    *,
    threads: int | None,
    selection_kwargs: dict[str, Any],
) -> SelectionResult:
    spec = run.spec
    x_train, y_train = x_pool.loc[mask], y[mask]
    if spec.model_type.value == "logistic":
        return select_logistic(x_train, y_train, **selection_kwargs)
    outcome = run.steps[-1].outcome
    return select_tree(
        spec,
        outcome.params,
        x_train,
        y_train,
        xgb_recalibrate=outcome.xgb_recalibrate,
        threads=threads,
        **selection_kwargs,
    )


def _holdout_starts() -> list[int]:
    return sorted(int(s.split("-")[0]) for s in HOLDOUT_SEASONS)


def _family_stage2(
    run: FamilyRun,
    rows: MatchedRows,
    repo: Path,
    *,
    threads: int | None,
    early_cutoff: int,
    last_eval: int,
    selection_kwargs: dict[str, Any],
) -> FamilyStage2:
    spec = run.spec
    steps = build_sweep_steps(spec.feature_set_key)
    baseline, full = steps[0], steps[-1]
    base_out, full_out = run.steps[0].outcome, run.steps[-1].outcome
    provider = make_feature_provider(rows, spec, repo)
    x_pool = provider(full_out.epa_window).loc[:, list(full.columns)]

    early = _select(
        run,
        x_pool,
        rows.y,
        train_mask_before(rows, early_cutoff),
        threads=threads,
        selection_kwargs=selection_kwargs,
    )
    final = _select(
        run, x_pool, rows.y, rows.train_mask, threads=threads, selection_kwargs=selection_kwargs
    )
    result = FamilyStage2(spec.label, len(baseline.columns), len(full.columns), early, final)

    window = eval_starts(early_cutoff, last_eval)
    arms = [
        WideArm("baseline", baseline.columns, base_out),
        WideArm("full", full.columns, full_out),
    ]
    if early.selected:
        arms.append(WideArm("selected", tuple(early.selected), full_out))
    preds = wide_predictions(spec, rows, provider, arms, starts=window)
    result.wide["full_vs_baseline"] = compare_non_inferiority(
        spec, rows, preds, "full", "baseline", starts=window
    )
    if early.selected:
        for key, reference in (("selected_vs_full", "full"), ("selected_vs_baseline", "baseline")):
            result.wide[key] = compare_non_inferiority(
                spec, rows, preds, "selected", reference, starts=window
            )

    holdout_window = _holdout_starts()
    hold_preds = {"baseline": run.steps[0].predictions, "full": run.steps[-1].predictions}
    result.holdout["full_vs_baseline"] = compare_non_inferiority(
        spec, rows, hold_preds, "full", "baseline", starts=holdout_window
    )
    if final.selected:
        hold_preds.update(
            wide_predictions(
                spec,
                rows,
                provider,
                [WideArm("selected", tuple(final.selected), full_out)],
                starts=holdout_window,
            )
        )
        for key, reference in (("selected_vs_full", "full"), ("selected_vs_baseline", "baseline")):
            result.holdout[key] = compare_non_inferiority(
                spec, rows, hold_preds, "selected", reference, starts=holdout_window
            )
    return result


def _path(checkpoint_dir: Path, label: str) -> Path:
    return checkpoint_dir / STAGE2_SUBDIR / f"{label.replace('/', '__')}.pkl"


def load_stage2(checkpoint_dir: Path) -> dict[str, FamilyStage2]:
    """Load every stage 2 family result in ``checkpoint_dir``."""
    out: dict[str, FamilyStage2] = {}
    for path in sorted((checkpoint_dir / STAGE2_SUBDIR).glob("*.pkl")):
        with path.open("rb") as handle:
            item: FamilyStage2 = pickle.load(handle)
        out[item.family] = item
    return out


def run_stage2(
    runs: dict[str, FamilyRun],
    matched: dict[str, MatchedRows],
    repo: Path,
    checkpoint_dir: Path,
    *,
    threads: int | None = None,
    early_cutoff: int = SELECTION_CUTOFF_START,
    last_eval: int = LAST_EVAL_START,
    selection_kwargs: dict[str, Any] | None = None,
    on_family: Callable[[str, float], None] | None = None,
) -> dict[str, FamilyStage2]:
    """Run (or resume) stage 2 for every completed sweep family."""
    import time

    results = load_stage2(checkpoint_dir)
    for label, run in runs.items():
        if label in results:
            continue
        started = time.monotonic()
        results[label] = _family_stage2(
            run,
            matched[run.spec.target],
            repo,
            threads=threads,
            early_cutoff=early_cutoff,
            last_eval=last_eval,
            selection_kwargs=selection_kwargs or {},
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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _comparison_row(name: str, c: WideComparison | None) -> str:
    if c is None:
        return f"| {name} | n/a | n/a | n/a | n/a | no columns selected |"
    return (
        f"| {name} | {c.n_games} | {_fmt(c.delta.improvement, 5)} | "
        f"[{_fmt(c.delta.lower, 5)}, {_fmt(c.delta.upper, 5)}] | -{c.margin} | "
        f"{'yes' if c.non_inferior else 'NO'} |"
    )


def build_stage2_report(results: dict[str, FamilyStage2], *, generated_on: date) -> str:
    """Render the stage 2 Markdown report."""
    lines = [
        f"# Phase A feature selection - {generated_on.isoformat()}",
        "",
        "Stage 2 of the feature sweep (`PLAN.md`): each model selects its own columns "
        "from the wired pool on training rows only, with an in-procedure placebo "
        "control. Improvement is positive when the subset is better than the reference "
        "(Brier for Win, MAE for Total). A subset is non-inferior when the lower end of "
        "its 90% paired-bootstrap interval is above minus the margin. Generated by "
        "`gridiron evaluate feature-selection`; a dated record, not an input to any pipeline.",
        "",
        "- Early selection: matched rows before the 2013 season, scored walk-forward on "
        "2013 onward (never seen by that selection), fixed hyperparameters from the "
        "full-set search shared by all arms.",
        "- Final selection: all matched pre-holdout rows (the deployable subset), "
        "confirmed once on the holdout.",
        "",
        "## Selection summary",
        "",
        "| Family | Pool | Early kept | Early placebo kept | Final kept | Final placebo kept |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        lines.append(
            f"| {label} | {r.n_full} | {len(r.early.selected)} | "
            f"{r.early.placebo_selected}/{r.early.placebo_candidates} | {len(r.final.selected)} | "
            f"{r.final.placebo_selected}/{r.final.placebo_candidates} |"
        )
    for label, r in results.items():
        lines += ["", f"## {label}", ""]
        for title, comparisons in (
            ("Wide walk-forward (early subset)", r.wide),
            ("Holdout (final subset)", r.holdout),
        ):
            lines += [
                f"### {title}",
                "",
                "| Comparison | Games | Improvement | 90% interval | Margin | Non-inferior |",
                "|---|---|---|---|---|---|",
            ]
            for key in ("selected_vs_full", "selected_vs_baseline", "full_vs_baseline"):
                lines.append(_comparison_row(key, comparisons.get(key)))
            lines.append("")
        lines += ["### Final subset", "", f"Kept {len(r.final.selected)} of {r.final.n_pool}:", ""]
        stable = sorted(r.final.stability.items(), key=lambda kv: -kv[1])
        lines.append(", ".join(f"{c} ({v:.2f})" for c, v in stable if c in set(r.final.selected)))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
