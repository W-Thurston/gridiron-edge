# src/gridiron_edge/evaluation/starting_qb_projection_report.py

"""Markdown report for the projected starting-QB backtest.

Public API
----------
build_report    scored rows + timing -> Markdown text
"""

from __future__ import annotations

from datetime import date

from pandas import DataFrame

from gridiron_edge.evaluation.starting_qb_projection import (
    CARRY_FORWARD,
    DEPTH_QB1,
    LEAKAGE_GAP,
    MIN_GAIN,
    MIN_SEASONS_WON,
    PRIMARY,
    VARIANT,
    BarVerdict,
    RuleEvaluation,
    error_breakdown,
    evaluate_rules,
)

_LABELS = {
    CARRY_FORWARD: "Carry-forward (previous starter, D65)",
    DEPTH_QB1: "Depth chart QB1",
    PRIMARY: "Primary: QB1 to QB3, skipping Out and Doubtful",
    VARIANT: "Variant: also skipping Questionable (information only)",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _points(value: float) -> str:
    return f"{value * 100:+.2f}"


def _rule_table(results: list[RuleEvaluation]) -> list[str]:
    lines = [
        "| Rule | Accuracy | Gain vs carry-forward (points) | 90% interval | Seasons won |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.rule == CARRY_FORWARD:
            lines.append(f"| {_LABELS[r.rule]} | {_pct(r.accuracy)} | baseline | | |")
            continue
        lines.append(
            f"| {_LABELS[r.rule]} | {_pct(r.accuracy)} | {_points(r.gain)} | "
            f"[{_points(r.gain_low)}, {_points(r.gain_high)}] | {_pct(r.seasons_won)} |"
        )
    return lines


def build_report(
    scored: DataFrame,
    verdict: BarVerdict,
    timing: dict[str, DataFrame],
    *,
    generated_on: date,
) -> str:
    """Render the backtest as Markdown.

    Args:
        scored: Output of ``score_rules``.
        verdict: Output of ``bar_verdict``.
        timing: Snapshot-timing tables keyed by label (``all weeks``, ...).
        generated_on: Date recorded in the report.

    Returns:
        The report text.
    """
    seasons = sorted(scored["season"].unique())
    later = scored[scored["week"] > 1]
    week_1 = scored[scored["week"] == 1]
    lines: list[str] = [
        "# Projected starting-QB backtest",
        "",
        f"Generated {generated_on.isoformat()}. Read-only evaluation (`PLAN.md` Unit 9); "
        "no model, schema or `data/` artifact changed.",
        "",
        "## Question",
        "",
        "Can a rules-based projected starter, built from the depth chart and the "
        "weekly availability status, name the quarterback who actually starts more often "
        "than carrying forward each team's previous starter (`DECISIONS.md` D65)?",
        "",
        f"Scored: {len(scored):,} regular-season team-games, {seasons[0]}-{seasons[-1]}, "
        "with a depth chart and a previous starter. The injury report is the last report "
        "of the week, so every reading below is the state at the final weekly report.",
        "",
        "## Result on every week",
        "",
        *_rule_table(evaluate_rules(scored)),
        "",
        "## Leakage guard",
        "",
        f"Week 1 accuracy of the primary rule is {_pct(week_1[f'{PRIMARY}_correct'].mean())} "
        f"({len(week_1):,} games) against {_pct(later[f'{PRIMARY}_correct'].mean())} "
        f"afterwards: a gap of {verdict.week_1_gap * 100:.1f} points. "
        f"The guard fires above {LEAKAGE_GAP * 100:.0f} points, "
        f"so it **{'fired' if verdict.flagged else 'did not fire'}**."
        + (
            " Week 1 is excluded from the decision and both readings are reported."
            if verdict.flagged
            else ""
        ),
        "",
        "Carry-forward is weak in week 1 only because offseason moves change starters "
        f"({_pct(week_1[f'{CARRY_FORWARD}_correct'].mean())} correct), so a large week-1 "
        "gain is expected even from a genuine pre-game chart. The 2025 snapshot test below "
        "checks week 1 directly.",
        "",
        "### Weeks after week 1 (the decision reading when the guard fires)",
        "",
        *_rule_table(evaluate_rules(later)),
        "",
        "## Bar",
        "",
        f"Pre-declared: the primary rule's accuracy exceeds carry-forward by at least "
        f"{MIN_GAIN * 100:.1f} points, the 90% interval excludes zero, and it wins in at "
        f"least {MIN_SEASONS_WON * 100:.0f}% of seasons.",
        "",
        f"- Every week: gain {_points(verdict.all_weeks.gain)} points, interval "
        f"[{_points(verdict.all_weeks.gain_low)}, {_points(verdict.all_weeks.gain_high)}], "
        f"seasons won {_pct(verdict.all_weeks.seasons_won)}.",
        f"- Weeks after week 1: gain {_points(verdict.later_weeks.gain)} points, interval "
        f"[{_points(verdict.later_weeks.gain_low)}, {_points(verdict.later_weeks.gain_high)}], "
        f"seasons won {_pct(verdict.later_weeks.seasons_won)}.",
        f"- Decision reading: {'weeks after week 1' if verdict.flagged else 'every week'}.",
        "",
        f"**Bar {'cleared' if verdict.passes else 'not cleared'}.**",
        "",
        "## Remaining errors of the primary rule",
        "",
    ]
    errors = error_breakdown(scored)
    lines += [
        f"- Misses: {int(errors['misses']):,}",
        f"- The actual starter was in the top three of the chart: "
        f"{_pct(errors['share_actual_in_top_three'])}",
        f"- The actual starter was the carried-forward quarterback the rule skipped: "
        f"{_pct(errors['share_actual_was_carry_forward'])}",
        f"- Misses in week 1: {_pct(errors['share_week_1'])}",
        "",
        "## By season (primary rule, gain over carry-forward in points)",
        "",
        "| Season | Games | Carry-forward | Primary | Gain |",
        "|---|---|---|---|---|",
    ]
    by_season = scored.groupby("season")
    for season, group in by_season:
        cf = group[f"{CARRY_FORWARD}_correct"].mean()
        pr = group[f"{PRIMARY}_correct"].mean()
        lines.append(f"| {season} | {len(group)} | {_pct(cf)} | {_pct(pr)} | {_points(pr - cf)} |")
    lines += [
        "",
        "## Timing evidence (2025, descriptive only)",
        "",
        "Latest depth chart snapshot on or before the lead time. `Chart only` is fully "
        "as-of. `Chart and report` adds the week's final injury report; the 2025 injury "
        "file has no modification time, so that column is optimistic about what was known "
        "at the lead time. No bar applies.",
        "",
    ]
    for label, table in timing.items():
        lines += [
            f"### {label}",
            "",
            "| Lead (hours) | Team-games | Chart only | Chart and report |",
            "|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {int(row['lead_hours'])} | {int(row['scored'])} | "
                f"{_pct(float(row['chart_only_accuracy']))} | "
                f"{_pct(float(row['chart_and_report_accuracy']))} |"
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Depth charts before 2025 carry no timestamp, so pre-kickoff capture is "
        "inferred from accuracy, not verified.",
        "- Injury reports are the last report of the week (mostly Friday). The rule "
        "therefore needs information that exists late in the week.",
        "- Starters change for reasons an injury report never shows (benching, rest, "
        "late scratches on game day).",
        "",
    ]
    return "\n".join(lines)
