# src/gridiron_edge/evaluation/feature_sweep_report.py

"""Consolidated Markdown report and per-game prediction payload for the sweep.

Pure formatting over ``FamilyRun``/``FamilyEvaluation`` results. Also derives
each batch's live-availability finding from the modeling frame (any NaN in a
batch's columns on rows from the holdout onward means a scheduled game could
lack a value, which disables the family for that week) and the Holm-adjusted
p-values across every batch-by-family comparison. Writes nothing itself.

Public API
----------
batch_live_complete     modeling frame -> {batch: bool}
build_report            runs, rows, frame -> Markdown text
predictions_payload     runs, rows -> JSON-serializable per-game predictions
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from pandas import DataFrame

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.feature_sweep_columns import ColumnLevel
from gridiron_edge.evaluation.feature_sweep_gates import (
    DEFAULT_TOTAL_BAR,
    DEFAULT_WIN_BAR,
    BatchDecision,
    FamilyVerdict,
    decide_batch,
)
from gridiron_edge.evaluation.feature_sweep_metrics import (
    DEFAULT_BOOTSTRAP_DRAWS,
    TotalMetrics,
    WinMetrics,
    holm_adjust,
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_run import (
    EARLY_SEASON_MAX_WEEK,
    PLACEBO_DRAWS,
    TREE_SEEDS,
    FamilyEvaluation,
    FamilyRun,
    evaluate_family_run,
)
from gridiron_edge.evaluation.feature_sweep_steps import (
    FEATURE_BATCHES,
    batch_columns,
)
from gridiron_edge.models.game_prediction._columns import _EXPANDED_FEATURES
from gridiron_edge.models.game_prediction._features import _make_expanded_features


def batch_live_complete(df: DataFrame) -> dict[str, dict[str, int]]:
    """Count NaN rows per batch among rows from the holdout onward.

    Args:
        df: Canonical modeling frame (including the current, unplayed season).

    Returns:
        ``{batch: {"rows": n, "rows_with_nan": k, "week1_rows_with_nan": w}}``.
    """
    earliest = min(int(s.split("-")[0]) for s in HOLDOUT_SEASONS)
    recent = df.loc[df["YEAR"].astype(str).str[:4].astype(int) >= earliest]
    features = _make_expanded_features(recent)
    out: dict[str, dict[str, int]] = {}
    for batch in FEATURE_BATCHES:
        cols = batch_columns(batch, list(_EXPANDED_FEATURES))
        missing = features.loc[:, cols].isna().any(axis=1)
        out[batch.name] = {
            "rows": len(recent),
            "rows_with_nan": int(missing.sum()),
            "week1_rows_with_nan": int((missing & (recent["WEEK_NUM"] == 1)).sum()),
        }
    return out


def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def _metric_row(name: str, m: WinMetrics | TotalMetrics) -> str:
    if isinstance(m, WinMetrics):
        return (
            f"| {name} | {m.n} | {_fmt(m.brier, 5)} | {_fmt(m.log_loss, 5)} | {_fmt(m.ece)} "
            f"| {_fmt(m.auc)} | {_fmt(m.calibration_slope, 3)} | "
            f"{_fmt(m.calibration_intercept, 3)} |"
        )
    return (
        f"| {name} | {m.n} | {_fmt(m.mae, 3)} | {_fmt(m.rmse, 3)} | {_fmt(m.bias, 3)} "
        f"| {_fmt(m.median_ae, 3)} |"
    )


def _metric_header(target: str) -> str:
    if target == "win":
        return (
            "| Step | n | Brier | Log loss | ECE | AUC | Cal. slope | Cal. intercept |\n"
            "|---|---|---|---|---|---|---|---|"
        )
    return "| Step | n | MAE | RMSE | Bias | Median AE |\n|---|---|---|---|---|---|"


def _family_section(ev: FamilyEvaluation, run: FamilyRun, target: str) -> list[str]:
    unit = "Brier" if target == "win" else "MAE"
    lines = [f"### {ev.family}", "", _metric_header(target)]
    for name, m in ev.step_metrics.items():
        lines.append(_metric_row(name, m))
    for name, m in ev.references.items():
        lines.append(_metric_row(f"*reference: {name}*", m))
    lines += [
        "",
        "Chosen hyperparameters per step (EPA window; XGBoost isotonic wrap decided from "
        "training out-of-fold ECE):",
        "",
    ]
    for sf in run.steps:
        wrap = ""
        if sf.outcome.oof_ece is not None:
            wrap = f", oof ECE {sf.outcome.oof_ece:.4f}, wrap={sf.outcome.xgb_recalibrate}"
        lines.append(
            f"- {sf.step}: window {sf.outcome.epa_window}, CV {unit} "
            f"{sf.outcome.cv_score:.4f}{wrap}, params {sf.outcome.params} "
            f"({sf.seconds / 60:.1f} min)"
        )
    lines += [
        "",
        f"Per-batch improvement over the previous step (positive = better; {unit}). "
        f"Interval is the paired-bootstrap 90% interval ({DEFAULT_BOOTSTRAP_DRAWS} draws).",
        "",
        "| Batch | Improvement | 90% interval | Season deltas | Fixed-HP | Placebo max "
        "| Seed improvements | Noise floor | Verdict | Failed criteria |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in ev.batches:
        failed = ", ".join(k for k, ok in b.verdict.criteria.items() if not ok) or "-"
        seasons = ", ".join(f"{_fmt(v, 4)}" for v in b.season_deltas.values())
        seeds = ", ".join(f"{k}:{_fmt(v, 4)}" for k, v in b.seed_improvements.items()) or "n/a"
        verdict = "PASS" if b.verdict.passed else ("HARM" if b.verdict.harmed else "fail")
        lines.append(
            f"| {b.batch} | {_fmt(b.delta.improvement, 4)} | "
            f"[{_fmt(b.delta.lower, 4)}, {_fmt(b.delta.upper, 4)}] | {seasons} | "
            f"{_fmt(b.fixed_hp_improvement, 4)} | {_fmt(b.placebo_max_improvement, 4)} | "
            f"{seeds} | {_fmt(b.noise_floor, 4)} | {verdict} | {failed} |"
        )
    lines += ["", "Early-season slice (weeks 1-3 vs later, matched games only):", ""]
    lines.append(_metric_header(target).replace("| Step ", "| Step / slice ", 1))
    for name, (early, late) in ev.early_season_metrics.items():
        lines.append(_metric_row(f"{name} (weeks <= {EARLY_SEASON_MAX_WEEK})", early))
        lines.append(_metric_row(f"{name} (later weeks)", late))
    lines.append("")
    return lines


def _column_level_section(cl: ColumnLevel) -> list[str]:
    lines = [
        "## Column-level findings",
        "",
        "Ablations drop a suspect group from the full model and score it on a pre-holdout "
        "screening window (2019-2022, expanding refits, the full model's frozen "
        "hyperparameters); the holdout never enters these decisions. Harm is the loss "
        "increase from dropping the group (positive = dropping hurts; Brier for Win, MAE "
        "for Total). A group is removable when the harm is within tolerance "
        "(0.0005 Brier / 0.02 MAE) and the interval's upper end is below 0.001 Brier / "
        "0.045 MAE.",
        "",
        "| Family | Group | Columns | Harm from dropping | 90% interval | Removable |",
        "|---|---|---|---|---|---|",
    ]
    for label, groups in cl.ablations.items():
        for name, r in groups.items():
            lines.append(
                f"| {label} | {name} | {r.n_columns_dropped} | {_fmt(r.harm, 5)} | "
                f"[{_fmt(r.delta.lower, 5)}, {_fmt(r.delta.upper, 5)}] | "
                f"{'yes' if r.removable else 'no'} |"
            )
    lines += ["", "### Suspect correlations (matched training rows)", ""]
    for target, table in cl.correlations.items():
        lines.append(f"**{target}**")
        lines.append("")
        lines.append("| Column | Partner | Correlation |")
        lines.append("|---|---|---|")
        for column, partners in table.items():
            for partner, corr in partners.items():
                lines.append(f"| {column} | {partner} | {_fmt(corr, 3)} |")
        lines.append("")
    lines += ["### Leakage smell test (new columns, training rows)", ""]
    for target, flags in cl.leakage.items():
        text = (
            "; ".join(
                f"{f.column} (auc {_fmt(f.auc, 3)}, |r| {_fmt(f.correlation, 3)})" for f in flags
            )
            or "no column flagged"
        )
        lines.append(f"- {target}: {text}")
    lines += ["", "### Train-versus-holdout drift (PSI above alert level)", ""]
    for target, drift in cl.drift.items():
        text = (
            "; ".join(f"{c} ({psi:.3f})" for c, psi in drift[:15]) or "none above the alert level"
        )
        lines.append(f"- {target}: {text}")
    lines += ["", "### Logistic collinearity (final step, training rows)", ""]
    for label, rep in cl.collinearity.items():
        top = ", ".join(f"{c} ({v:.1f})" for c, v in list(rep.vif.items())[:12])
        lines.append(f"- {label}: condition number {rep.condition_number:.1f}; highest VIF: {top}")
    lines.append("")
    return lines


def build_report(
    runs: dict[str, FamilyRun],
    matched: dict[str, MatchedRows],
    df: DataFrame,
    *,
    generated_on: date,
    column_level: ColumnLevel | None = None,
) -> tuple[str, dict[str, BatchDecision]]:
    """Build the consolidated Markdown report and the per-batch decisions."""
    evaluations: dict[str, FamilyEvaluation] = {
        label: evaluate_family_run(run, matched[run.spec.target]) for label, run in runs.items()
    }
    live = batch_live_complete(df)

    p_values = {
        f"{label}|{b.batch}": b.delta.p_no_improvement
        for label, ev in evaluations.items()
        for b in ev.batches
    }
    holm = holm_adjust(p_values)

    decisions: dict[str, BatchDecision] = {}
    for batch in FEATURE_BATCHES:
        verdicts: list[FamilyVerdict] = [
            b.verdict for ev in evaluations.values() for b in ev.batches if b.batch == batch.name
        ]
        if not verdicts:
            continue
        decisions[batch.name] = decide_batch(
            batch.name, verdicts, live_complete=live[batch.name]["rows_with_nan"] == 0
        )

    win = next((m for t, m in matched.items() if t == "win"), None)
    total = next((m for t, m in matched.items() if t == "total"), None)
    lines: list[str] = [
        f"# Phase A feature sweep - {generated_on.isoformat()}",
        "",
        "Consolidated incremental walk-forward sweep of the wired feature batches "
        "(`DECISIONS.md` D62). Generated by `gridiron evaluate feature-sweep`; a dated "
        "record, not an input to any pipeline.",
        "",
        "## Protocol",
        "",
        "- Steps: baseline (schema 5 composition), then each batch cumulatively in wiring "
        "order; each batch is scored against the step before it.",
        "- Rows: matched. Every step and family is scored on identical games: rows complete "
        "on the widest feature set. This excludes season-opener games whose adjusted-EPA and "
        "some QB/coach features are undefined, so the sweep cannot speak to Week 1.",
    ]
    for label, rows in (("Win", win), ("Total", total)):
        if rows is not None:
            lines.append(
                f"- {label} matched rows: {rows.n_train} train of {rows.universe_train}, "
                f"{rows.n_holdout} holdout of {rows.universe_holdout}."
            )
    lines += [
        f"- Hyperparameters: each step runs its own training-only search (production grids, "
        f"budgets and seeded combos); frozen, then refit expanding through each holdout "
        f"season. Trees also rerun with seeds {list(TREE_SEEDS)}; {PLACEBO_DRAWS} placebo "
        f"(random-noise) draws per batch; a fixed baseline-hyperparameter reading.",
        f"- Win bar: Brier improvement >= {DEFAULT_WIN_BAR.min_brier_improvement}, bootstrap "
        f"lower bound > 0, >= {DEFAULT_WIN_BAR.min_seasons_improved} of 3 seasons improved, "
        f"ECE worsening <= {DEFAULT_WIN_BAR.max_ece_worsening}, log loss not worse.",
        f"- Total bar: MAE improvement >= {DEFAULT_TOTAL_BAR.min_mae_improvement}, bootstrap "
        f"lower bound > 0, >= {DEFAULT_TOTAL_BAR.min_seasons_improved} of 3 seasons, RMSE not "
        f"worse, bias shift <= {DEFAULT_TOTAL_BAR.max_bias_shift}.",
        "- A pass must also exceed the noise floor: the larger of the largest placebo "
        "improvement and the seed range of the batch's improvement.",
        "- Very short EPA windows (1, 2) leave some rates undefined; those columns are "
        "imputed with training medians so matched games stay identical.",
        "- The Elo reference is an `ELO_DIFF`-only logistic; the simulated Elo model is not "
        "re-run.",
        f"- Comparisons made: {len(p_values)} (batch x family). Holm-adjusted p-values are "
        "listed below; the holdout was also used earlier for champion selection.",
        "",
        "## Batch decisions",
        "",
        "| Batch | Decision | Reason | Live availability (rows from holdout onward) |",
        "|---|---|---|---|",
    ]
    for name, d in decisions.items():
        info = live[name]
        lines.append(
            f"| {name} | **{d.status}** | {d.reason} | {info['rows_with_nan']} of {info['rows']} "
            f"rows with a NaN ({info['week1_rows_with_nan']} in Week 1) |"
        )
    lines += [
        "",
        "## Multiple-comparison view",
        "",
        "| Comparison | Raw p | Holm p |",
        "|---|---|---|",
    ]
    for key in sorted(p_values, key=lambda k: p_values[k]):
        lines.append(f"| {key} | {p_values[key]:.4f} | {holm[key]:.4f} |")
    lines += ["", "## Results by family", ""]
    for label, ev in evaluations.items():
        lines += _family_section(ev, runs[label], runs[label].spec.target)
    if column_level is not None:
        lines += _column_level_section(column_level)
    return "\n".join(lines).rstrip("\n") + "\n", decisions


def predictions_payload(
    runs: dict[str, FamilyRun], matched: dict[str, MatchedRows]
) -> dict[str, Any]:
    """Return per-game holdout predictions (primary reading) as plain JSON data."""
    payload: dict[str, Any] = {}
    for label, run in runs.items():
        rows = matched[run.spec.target]
        mask = rows.holdout_mask
        payload[label] = {
            "game_ids": rows.frame.loc[mask, "GAME_ID"].tolist(),
            "actual": [round(v, 4) for v in rows.y[mask].tolist()],
            "steps": {
                sf.step: [
                    round(v, 5) for v in sf.predictions.reindex(rows.frame.index)[mask].tolist()
                ]
                for sf in run.steps
            },
        }
    return payload
