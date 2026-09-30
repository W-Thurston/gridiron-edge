# src/gridiron_edge/evaluation/feature_sweep_columns.py

"""Column-level stage of the consolidated feature sweep.

Runs after the batch sweep, on the full (final-step) model: suspect-group
ablations on a pre-holdout screening window, plus leakage, drift and
collinearity evidence. Evidence only; ablation results inform removal but
nothing here reads a holdout label.

Public API
----------
SUSPECT_PARTNERS        named correlation checks
ColumnLevel             stage results
run_column_level        runs + matched rows -> ColumnLevel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from gridiron_edge.evaluation.feature_sweep_ablation import AblationResult, run_ablation
from gridiron_edge.evaluation.feature_sweep_diagnostics import (
    PSI_ALERT,
    CollinearityReport,
    LeakageFlag,
    collinearity_report,
    leakage_smell_test,
    population_stability,
    suspect_correlations,
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_run import FamilyRun, make_feature_provider
from gridiron_edge.evaluation.feature_sweep_steps import (
    FEATURE_BATCHES,
    batch_columns,
    build_sweep_steps,
)
from gridiron_edge.models.game_prediction._columns import _EXPANDED_FEATURES

#: Suspect columns and the columns they are checked against.
SUSPECT_PARTNERS: Final[dict[str, list[str]]] = {
    "GLICKO2_RD_DIFF": ["WEEK_NUM", "ELO_DIFF"],
    "AWAY_GLICKO2_RD": ["WEEK_NUM"],
    "HOME_GLICKO2_RD": ["WEEK_NUM"],
    "GLICKO2_RATING_DIFF": ["ELO_DIFF"],
    "MASSEY_RATING_DIFF": ["ELO_DIFF", "GLICKO2_RATING_DIFF"],
    "COLLEY_RATING_DIFF": ["ELO_DIFF", "MASSEY_RATING_DIFF"],
    "KEENER_RATING_DIFF": ["ELO_DIFF", "MASSEY_RATING_DIFF"],
    "QB_ELO_DIFF": ["ELO_DIFF"],
    "FORM_TREND_DIFF": ["WEEK_NUM"],
}


@dataclass
class ColumnLevel:
    """Results of the column-level stage.

    Attributes:
        ablations: Family label -> group name -> result.
        leakage: Target -> flagged columns among batch columns.
        drift: Target -> batch columns with PSI above the alert level.
        collinearity: Family label -> report (Logistic only).
        correlations: Target -> suspect -> partner -> Pearson correlation on
            matched training rows.
    """

    ablations: dict[str, dict[str, AblationResult]] = field(default_factory=dict)
    leakage: dict[str, list[LeakageFlag]] = field(default_factory=dict)
    drift: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    collinearity: dict[str, CollinearityReport] = field(default_factory=dict)
    correlations: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def run_column_level(
    runs: dict[str, FamilyRun], matched: dict[str, MatchedRows], repo: Path
) -> ColumnLevel:
    """Run the column-level stage for every completed family."""
    result = ColumnLevel()
    batch_cols: list[str] = []
    for batch in FEATURE_BATCHES:
        batch_cols += batch_columns(batch, list(_EXPANDED_FEATURES))

    for label, run in runs.items():
        spec = run.spec
        rows = matched[spec.target]
        final = build_sweep_steps(spec.feature_set_key)[-1]
        provider = make_feature_provider(rows, spec, repo)
        result.ablations[label] = run_ablation(
            spec, rows, provider, run.steps[-1].outcome, list(final.columns)
        )
        if spec.model_type.value == "logistic":
            x_train = provider(run.steps[-1].outcome.epa_window).loc[
                rows.train_mask, list(final.columns)
            ]
            result.collinearity[label] = collinearity_report(x_train)

    for target, rows in matched.items():
        x_train = rows.features.loc[rows.train_mask]
        x_hold = rows.features.loc[rows.holdout_mask]
        y_train = rows.y[rows.train_mask]
        result.leakage[target] = leakage_smell_test(
            x_train, y_train, batch_cols, binary_target=(target == "win")
        )
        result.drift[target] = sorted(
            (
                (c, psi)
                for c in batch_cols
                if (psi := population_stability(x_train[c], x_hold[c])) > PSI_ALERT
            ),
            key=lambda item: -item[1],
        )
        result.correlations[target] = suspect_correlations(x_train, SUSPECT_PARTNERS)
    return result
