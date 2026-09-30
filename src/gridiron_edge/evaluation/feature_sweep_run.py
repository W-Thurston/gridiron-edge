# src/gridiron_edge/evaluation/feature_sweep_run.py

"""Orchestration of one family's incremental feature sweep.

For a (target, model family) this runs every cumulative step through a
training-only search and expanding holdout refits (``feature_sweep_fit``),
plus the noise-floor evidence: fixed-hyperparameter refits, tree seed
reruns, and placebo (random-noise) batches. It then scores each batch
against the step before it under the recorded bars
(``feature_sweep_gates``). Read-only: it persists nothing and never touches
champion or prediction artifacts.

Public API
----------
FamilyRun, StepFit          raw predictions per step
run_family                  spec, rows, steps -> FamilyRun
reference_predictions       naive and Elo-only references on the matched holdout
BatchEvaluation, FamilyEvaluation
evaluate_family_run         FamilyRun -> per-batch verdicts and metrics
make_feature_provider       EPA-window feature matrices for the matched frame
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.feature_sweep_fit import (
    DEFAULT_SEED,
    FamilySpec,
    FeatureProvider,
    SearchOutcome,
    expanding_holdout_predictions,
    search_hyperparameters,
)
from gridiron_edge.evaluation.feature_sweep_gates import (
    DEFAULT_TOTAL_BAR,
    DEFAULT_WIN_BAR,
    FamilyVerdict,
    evaluate_total_family,
    evaluate_win_family,
)
from gridiron_edge.evaluation.feature_sweep_metrics import (
    BootstrapDelta,
    TotalMetrics,
    WinMetrics,
    absolute_losses,
    brier_losses,
    paired_bootstrap_improvement,
    season_improvements,
    total_metrics,
    win_metrics,
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows
from gridiron_edge.evaluation.feature_sweep_steps import FeatureStep
from gridiron_edge.models.game_prediction._epa_window import _rebuild_features_with_window
from gridiron_edge.models.game_prediction._features import _make_expanded_features

logger = logging.getLogger(__name__)

TREE_SEEDS: tuple[int, ...] = (42, 7, 123)
PLACEBO_DRAWS = 5
PLACEBO_SEED = 424242
EARLY_SEASON_MAX_WEEK = 3

StepCallback = Callable[[str, str, float], None]


# ---------------------------------------------------------------------------
# Feature provider
# ---------------------------------------------------------------------------


def make_feature_provider(rows: MatchedRows, spec: FamilySpec, repo: Path) -> FeatureProvider:
    """Return a cached per-window feature-matrix provider for the matched frame.

    Regression always uses the on-disk four-game window. Other windows
    rebuild the canonical EPA columns exactly as production's search does,
    aligned to the matched frame by ``GAME_ID``.

    Raises:
        ValueError: If a rebuilt window still leaves a matched row incomplete
            after training-median imputation.
    """
    cache: dict[int, DataFrame] = {4: rows.features}

    def provider(window: int) -> DataFrame:
        effective = 4 if spec.task == "regression" else window
        if effective not in cache:
            rebuilt = _rebuild_features_with_window(rows.frame, window=effective, repo=repo)
            feats = _make_expanded_features(rebuilt)
            feats.index = pd.Index(rebuilt["GAME_ID"].to_numpy())
            aligned = feats.reindex(pd.Index(rows.frame["GAME_ID"].to_numpy()))
            aligned.index = rows.frame.index
            incomplete = aligned.loc[rows.matched].isna().any()
            if incomplete.any():
                # Very short windows can leave a rate undefined (no attempts).
                # Impute with the matched *training* median so every step keeps
                # the same matched games; recorded in the sweep report.
                medians = aligned.loc[rows.train_mask, incomplete[incomplete].index].median()
                aligned = aligned.fillna(medians)
                if aligned.loc[rows.matched].isna().any().any():
                    msg = f"EPA window {effective} leaves matched rows incomplete"
                    raise ValueError(msg)
                logger.info(
                    "EPA window %d: imputed %d columns with training medians",
                    effective,
                    int(incomplete.sum()),
                )
            cache[effective] = aligned
        return cache[effective]

    return provider


# ---------------------------------------------------------------------------
# Raw run
# ---------------------------------------------------------------------------


@dataclass
class StepFit:
    """Primary search outcome and holdout predictions for one step."""

    step: str
    outcome: SearchOutcome
    predictions: Series
    seconds: float


@dataclass
class FamilyRun:
    """Every prediction set one family's sweep produced.

    Attributes:
        spec: The family.
        steps: Primary fits in step order.
        fixed_hp: Step name -> predictions refit with the baseline's
            hyperparameters (sensitivity reading).
        seeds: Seed -> step name -> predictions (trees only).
        placebo: Step name -> predictions of the previous step plus noise
            columns as wide as the batch, one Series per draw.
    """

    spec: FamilySpec
    steps: list[StepFit] = field(default_factory=list)
    fixed_hp: dict[str, Series] = field(default_factory=dict)
    seeds: dict[int, dict[str, Series]] = field(default_factory=dict)
    placebo: dict[str, list[Series]] = field(default_factory=dict)


def _holdout_starts() -> list[int]:
    return sorted(int(s.split("-")[0]) for s in HOLDOUT_SEASONS)


def _season_start(rows: MatchedRows) -> Series:
    return rows.frame["YEAR"].astype(str).str[:4].astype(int)


def run_family(
    spec: FamilySpec,
    rows: MatchedRows,
    steps: list[FeatureStep],
    provider: FeatureProvider,
    *,
    seeds: tuple[int, ...] = TREE_SEEDS,
    placebo_draws: int = PLACEBO_DRAWS,
    on_step: StepCallback | None = None,
    resume: FamilyRun | None = None,
    checkpoint: Callable[[FamilyRun], None] | None = None,
) -> FamilyRun:
    """Run every step for one family.

    Args:
        spec: Family.
        rows: Matched rows for the family's target.
        steps: Cumulative steps for the family's feature set, baseline first.
        provider: EPA-window feature provider.
        seeds: Estimator seeds for trees; the first is the primary seed.
        placebo_draws: Random-noise draws per batch.
        on_step: Optional ``(family, step, seconds)`` progress callback.
        resume: A partial run for the same family; its completed steps are
            kept and skipped.
        checkpoint: Called with the run after every completed step.
    """
    run = resume if resume is not None else FamilyRun(spec=spec)
    y = rows.y
    starts = _season_start(rows)
    holdout_starts = _holdout_starts()
    use_seeds = spec.model_type.value != "logistic"
    extra_seeds = [s for s in seeds if s != DEFAULT_SEED] if use_seeds else []
    for s in extra_seeds:
        run.seeds.setdefault(s, {})

    def refit(outcome: SearchOutcome, columns: list[str], seed: int = DEFAULT_SEED) -> Series:
        x_all = provider(outcome.epa_window).loc[:, columns]
        return expanding_holdout_predictions(
            spec,
            outcome,
            x_all=x_all,
            y=y,
            season_start=starts,
            matched=rows.matched,
            holdout_seasons=holdout_starts,
            seed=seed,
        )

    for i, step in enumerate(steps):
        if i < len(run.steps):
            continue
        started = time.monotonic()
        columns = list(step.columns)
        outcome = search_hyperparameters(spec, provider, columns, y, rows.train_mask)
        preds = refit(outcome, columns)
        run.steps.append(StepFit(step.name, outcome, preds, time.monotonic() - started))
        for s in extra_seeds:
            run.seeds[s][step.name] = refit(outcome, columns, s)
        if i > 0:
            run.fixed_hp[step.name] = refit(run.steps[0].outcome, columns)
            prev = run.steps[i - 1]
            prev_cols = list(steps[i - 1].columns)
            width = len(columns) - len(prev_cols)
            draws: list[Series] = []
            for d in range(placebo_draws):
                noise_rng = np.random.default_rng([PLACEBO_SEED, i, d])
                x_prev = provider(prev.outcome.epa_window).loc[:, prev_cols]
                noise = pd.DataFrame(
                    noise_rng.normal(size=(len(x_prev), width)),
                    index=x_prev.index,
                    columns=[f"__noise_{j}" for j in range(width)],
                )
                x_all = pd.concat([x_prev, noise], axis=1)
                draws.append(
                    expanding_holdout_predictions(
                        spec,
                        prev.outcome,
                        x_all=x_all,
                        y=y,
                        season_start=starts,
                        matched=rows.matched,
                        holdout_seasons=holdout_starts,
                    )
                )
            run.placebo[step.name] = draws
        if checkpoint is not None:
            checkpoint(run)
        if on_step is not None:
            on_step(spec.label, step.name, time.monotonic() - started)
    # The primary seed is the DEFAULT_SEED run already stored in ``steps``.
    if use_seeds:
        run.seeds[DEFAULT_SEED] = {s.step: s.predictions for s in run.steps}
    return run


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def reference_predictions(rows: MatchedRows, target: str) -> dict[str, Series]:
    """Naive references on the matched holdout, refit expanding like the models.

    Win: the training home-win rate, and a logistic on ``ELO_DIFF`` alone.
    Total: the training mean total. Returns predictions indexed like the frame.
    """
    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegression

    starts = _season_start(rows)
    y = rows.y
    const = pd.Series(np.nan, index=rows.frame.index, dtype=float)
    elo = pd.Series(np.nan, index=rows.frame.index, dtype=float)
    for start in _holdout_starts():
        train = rows.matched & (starts < start)
        hold = rows.matched & (starts == start)
        const.loc[hold] = float(y[train].mean())
        if target == "win":
            x_tr = rows.features.loc[train, ["ELO_DIFF"]].to_numpy()
            model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            model.fit(x_tr, y[train].to_numpy())
            elo.loc[hold] = model.predict_proba(rows.features.loc[hold, ["ELO_DIFF"]].to_numpy())[
                :, 1
            ]
    if target == "win":
        return {"home_win_rate": const, "elo_diff_logistic": elo}
    return {"league_mean_total": const}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchEvaluation:
    """Verdict and evidence for one batch in one family.

    Attributes:
        batch: Batch (step) name.
        verdict: Primary reading against the previous step.
        delta: Bootstrap improvement of the primary reading.
        season_deltas: Per-season improvement of the primary reading.
        fixed_hp_improvement: Point improvement under baseline hyperparameters.
        placebo_max_improvement: Largest spurious improvement over placebo draws.
        seed_improvements: Improvement per seed (trees only).
        noise_floor: The floor the improvement had to exceed.
    """

    batch: str
    verdict: FamilyVerdict
    delta: BootstrapDelta
    season_deltas: dict[str, float]
    fixed_hp_improvement: float
    placebo_max_improvement: float
    seed_improvements: dict[int, float]
    noise_floor: float


@dataclass(frozen=True)
class FamilyEvaluation:
    """Scored sweep for one family.

    Attributes:
        family: Family label.
        step_metrics: Step name -> metrics on the matched holdout.
        early_season_metrics: Step name -> (weeks <= 3, later weeks) metrics.
        batches: Per-batch evaluations in wiring order.
        references: Reference name -> metrics.
    """

    family: str
    step_metrics: dict[str, WinMetrics | TotalMetrics]
    early_season_metrics: dict[str, tuple[WinMetrics | TotalMetrics, WinMetrics | TotalMetrics]]
    batches: list[BatchEvaluation]
    references: dict[str, WinMetrics | TotalMetrics]


def _arrays(rows: MatchedRows, preds: Series) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mask = rows.holdout_mask
    # Checkpointed predictions may predate rows appended to the frame since.
    p = preds.reindex(rows.frame.index)[mask].to_numpy(dtype=float)
    if np.isnan(p).any():
        msg = "Holdout predictions contain NaN"
        raise ValueError(msg)
    return p, rows.y[mask].to_numpy(dtype=float)


def _metrics(
    target: str, p: NDArray[np.float64], y: NDArray[np.float64]
) -> WinMetrics | TotalMetrics:
    return win_metrics(p, y) if target == "win" else total_metrics(p, y)


def _losses(target: str, p: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    return brier_losses(p, y) if target == "win" else absolute_losses(p, y)


def evaluate_family_run(run: FamilyRun, rows: MatchedRows) -> FamilyEvaluation:
    """Score a family's run: metrics per step and a verdict per batch."""
    target = run.spec.target
    seasons = rows.frame.loc[rows.holdout_mask, "YEAR"].astype(str).to_numpy()
    weeks = rows.frame.loc[rows.holdout_mask, "WEEK_NUM"].to_numpy()
    early = weeks <= EARLY_SEASON_MAX_WEEK

    holdout: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
        s.step: _arrays(rows, s.predictions) for s in run.steps
    }
    step_metrics = {name: _metrics(target, p, y) for name, (p, y) in holdout.items()}
    early_metrics = {
        name: (_metrics(target, p[early], y[early]), _metrics(target, p[~early], y[~early]))
        for name, (p, y) in holdout.items()
    }

    batches: list[BatchEvaluation] = []
    for i in range(1, len(run.steps)):
        prev, cur = run.steps[i - 1], run.steps[i]
        p_prev, y = holdout[prev.step]
        p_cur, _ = holdout[cur.step]
        loss_prev, loss_cur = _losses(target, p_prev, y), _losses(target, p_cur, y)
        delta = paired_bootstrap_improvement(loss_prev, loss_cur)
        season_deltas = season_improvements(loss_prev, loss_cur, seasons)

        fixed_p, _ = _arrays(rows, run.fixed_hp[cur.step])
        fixed_prev_p, _ = (
            holdout[run.steps[0].step] if i == 1 else _arrays(rows, run.fixed_hp[prev.step])
        )
        fixed_improvement = float(
            np.mean(_losses(target, fixed_prev_p, y)) - np.mean(_losses(target, fixed_p, y))
        )

        placebo_impr = [
            float(np.mean(loss_prev) - np.mean(_losses(target, _arrays(rows, draw)[0], y)))
            for draw in run.placebo[cur.step]
        ]
        placebo_max = max([0.0, *placebo_impr])

        seed_impr: dict[int, float] = {}
        for seed, by_step in run.seeds.items():
            sp_prev = _arrays(rows, by_step[prev.step])[0]
            sp_cur = _arrays(rows, by_step[cur.step])[0]
            seed_impr[seed] = float(
                np.mean(_losses(target, sp_prev, y)) - np.mean(_losses(target, sp_cur, y))
            )
        seed_range = (max(seed_impr.values()) - min(seed_impr.values())) if seed_impr else 0.0
        floor = max(placebo_max, seed_range)

        if target == "win":
            verdict = evaluate_win_family(
                run.spec.label,
                step_metrics[prev.step],  # type: ignore[arg-type]
                step_metrics[cur.step],  # type: ignore[arg-type]
                delta,
                season_deltas,
                bar=DEFAULT_WIN_BAR,
                noise_floor=floor,
            )
        else:
            verdict = evaluate_total_family(
                run.spec.label,
                step_metrics[prev.step],  # type: ignore[arg-type]
                step_metrics[cur.step],  # type: ignore[arg-type]
                delta,
                season_deltas,
                bar=DEFAULT_TOTAL_BAR,
                noise_floor=floor,
            )
        batches.append(
            BatchEvaluation(
                batch=cur.step,
                verdict=verdict,
                delta=delta,
                season_deltas=season_deltas,
                fixed_hp_improvement=fixed_improvement,
                placebo_max_improvement=placebo_max,
                seed_improvements=seed_impr,
                noise_floor=floor,
            )
        )

    references = {
        name: _metrics(target, *_arrays(rows, preds))
        for name, preds in reference_predictions(rows, target).items()
    }
    return FamilyEvaluation(run.spec.label, step_metrics, early_metrics, batches, references)
