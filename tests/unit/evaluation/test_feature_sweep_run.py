# tests/unit/evaluation/test_feature_sweep_run.py

"""Tests for the sweep's search, expanding refits and family orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation import feature_sweep_fit as fit_mod
from gridiron_edge.evaluation.feature_sweep_fit import (
    FamilySpec,
    expanding_holdout_predictions,
    fit_predict,
    search_hyperparameters,
)
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows, build_matched_rows
from gridiron_edge.evaluation.feature_sweep_run import (
    FamilyRun,
    evaluate_family_run,
    make_feature_provider,
    reference_predictions,
    run_family,
)
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.models.game_prediction._columns import (
    _GAME_FEATURES,
    _RAW_FEATURES,
    _TEAM_FEATURES,
)
from gridiron_edge.models.game_prediction.base import GameModelType

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
PER_SEASON = 120


def _frame() -> DataFrame:
    rng = np.random.default_rng(3)
    n = PER_SEASON * len(SEASONS)
    values = {c: rng.normal(size=n) for c in [*_RAW_FEATURES, *_GAME_FEATURES, *_TEAM_FEATURES]}
    values["HOME_ELO"] = 1500 + 60 * rng.normal(size=n)
    values["AWAY_ELO"] = 1500 + 60 * rng.normal(size=n)
    structure = []
    for s, season in enumerate(SEASONS):
        for i in range(PER_SEASON):
            structure.append(
                {
                    "GAME_ID": f"{season}_{i:03d}",
                    "YEAR": season,
                    "GAME_DATE": pd.Timestamp("2021-09-01") + pd.Timedelta(days=365 * s + i),
                }
            )
    df = pd.concat([pd.DataFrame(values), pd.DataFrame(structure)], axis=1)
    df["WEEK_NUM"] = [2 + i % 16 for i in range(PER_SEASON)] * len(SEASONS)
    edge = (df["HOME_ELO"] - df["AWAY_ELO"]) / 400.0
    df["HOME_WIN"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-3 * edge))).astype(float)
    df["ACTUAL_TOTAL"] = 45 + 4 * df["AWAY_OFF_EPA_PER_PLAY"] + rng.normal(0, 3, size=n)
    return df


@pytest.fixture(autouse=True)
def _tiny_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    def grid(model_type: GameModelType, task: str) -> list[dict[str, Any]]:
        if model_type == GameModelType.LOGISTIC:
            return [{"epa_window": 4}]
        if model_type == GameModelType.RANDOM_FOREST:
            base = {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5}
        else:
            base = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}
        rows = [{**base, "max_depth": d} for d in (2, 3)]
        return [{**r, "epa_window": 4} for r in rows] if task == "classification" else rows

    monkeypatch.setattr(fit_mod, "_get_param_grid", grid)
    monkeypatch.setattr(fit_mod, "_n_iter_for", lambda *_: 2)


def _spec(target: str, model: GameModelType, key: str = "expanded") -> FamilySpec:
    return FamilySpec(target, model, key)


def _rows(target: str) -> MatchedRows:
    return build_matched_rows(_frame(), "HOME_WIN" if target == "win" else "ACTUAL_TOTAL")


@pytest.mark.unit
def test_fit_predict_returns_probabilities_and_is_deterministic() -> None:
    rows = _rows("win")
    spec = _spec("win", GameModelType.LOGISTIC, "combined")
    cols = list(build_sweep_steps("combined")[0].columns)
    x = rows.features.loc[:, cols]
    args = (spec, {}, x[rows.train_mask], rows.y[rows.train_mask], x[rows.holdout_mask])
    a, b = fit_predict(*args), fit_predict(*args)
    assert a.shape == (rows.n_holdout,)
    assert np.all((a >= 0) & (a <= 1))
    assert np.allclose(a, b)


@pytest.mark.unit
def test_search_never_reads_holdout_labels() -> None:
    rows = _rows("win")
    y = rows.y.copy()
    y[~rows.train_mask] = np.nan  # a search touching any non-training label would fail
    spec = _spec("win", GameModelType.RANDOM_FOREST)
    cols = list(build_sweep_steps("expanded")[0].columns)
    outcome = search_hyperparameters(spec, lambda _w: rows.features, cols, y, rows.train_mask)
    assert outcome.epa_window == 4
    assert outcome.params["max_depth"] in (2, 3)
    assert outcome.cv_score == pytest.approx(outcome.cv_score) and outcome.cv_score < 0.5


@pytest.mark.unit
def test_xgboost_recalibration_follows_training_oof_ece() -> None:
    rows = _rows("win")
    spec = _spec("win", GameModelType.XGBOOST)
    cols = list(build_sweep_steps("expanded")[0].columns)
    outcome = search_hyperparameters(spec, lambda _w: rows.features, cols, rows.y, rows.train_mask)
    assert outcome.oof_ece is not None
    assert outcome.xgb_recalibrate == (outcome.oof_ece > 0.025)


@pytest.mark.unit
def test_expanding_refit_uses_only_earlier_seasons() -> None:
    rows = _rows("win")
    spec = _spec("win", GameModelType.RANDOM_FOREST)
    cols = list(build_sweep_steps("expanded")[0].columns)
    outcome = search_hyperparameters(spec, lambda _w: rows.features, cols, rows.y, rows.train_mask)
    starts = rows.frame["YEAR"].str[:4].astype(int)
    x_all = rows.features.loc[:, cols]
    base = expanding_holdout_predictions(
        spec,
        outcome,
        x_all=x_all,
        y=rows.y,
        season_start=starts,
        matched=rows.matched,
        holdout_seasons=[2023, 2024, 2025],
    )
    scrambled = rows.y.copy()
    middle = starts == 2024
    scrambled[middle] = 1.0 - scrambled[middle]
    changed = expanding_holdout_predictions(
        spec,
        outcome,
        x_all=x_all,
        y=scrambled,
        season_start=starts,
        matched=rows.matched,
        holdout_seasons=[2023, 2024, 2025],
    )
    # A season's own labels never influence its predictions; later seasons
    # train on them.
    assert np.allclose(
        base[rows.matched & (starts <= 2024) & (starts >= 2023)],
        changed[rows.matched & (starts <= 2024) & (starts >= 2023)],
    )
    last = rows.matched & (starts == 2025)
    assert not np.allclose(base[last], changed[last])
    assert base[rows.matched & (starts >= 2023)].notna().all()
    assert base[rows.matched & (starts < 2023)].isna().all()


@pytest.mark.unit
def test_references_are_expanding_and_target_appropriate() -> None:
    win = reference_predictions(_rows("win"), "win")
    assert set(win) == {"home_win_rate", "elo_diff_logistic"}
    rows = _rows("win")
    assert win["elo_diff_logistic"][rows.holdout_mask].notna().all()
    assert win["home_win_rate"][rows.holdout_mask].between(0, 1).all()
    total = reference_predictions(_rows("total"), "total")
    assert set(total) == {"league_mean_total"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "model", "key"),
    [
        ("win", GameModelType.XGBOOST, "expanded"),
        ("total", GameModelType.XGBOOST, "expanded"),
    ],
)
def test_run_family_end_to_end_scores_every_batch(
    target: str, model: GameModelType, key: str
) -> None:
    rows = _rows(target)
    spec = _spec(target, model, key)
    steps = build_sweep_steps(key)
    provider = make_feature_provider(rows, spec, Path("."))
    seen: list[tuple[str, str]] = []
    run = run_family(
        spec,
        rows,
        steps,
        provider,
        placebo_draws=1,
        on_step=lambda fam, step, _sec: seen.append((fam, step)),
    )
    assert [s for _, s in seen] == [s.name for s in steps]
    ev = evaluate_family_run(run, rows)
    assert [b.batch for b in ev.batches] == [s.name for s in steps[1:]]
    assert set(ev.step_metrics) == {s.name for s in steps}
    for b in ev.batches:
        assert b.noise_floor >= 0.0
        assert set(b.verdict.criteria) >= {"exceeds_noise_floor", "bootstrap_lower_above_zero"}
        assert b.delta.n_games == rows.n_holdout
    assert ev.references
    for early, late in ev.early_season_metrics.values():
        assert early.n + late.n == rows.n_holdout


@pytest.mark.unit
def test_parallel_search_matches_serial_search() -> None:
    rows = _rows("total")
    spec = _spec("total", GameModelType.XGBOOST)
    cols = list(build_sweep_steps("expanded")[0].columns)
    serial = search_hyperparameters(
        spec, lambda _w: rows.features, cols, rows.y, rows.train_mask, n_workers=1
    )
    parallel = search_hyperparameters(
        spec, lambda _w: rows.features, cols, rows.y, rows.train_mask, n_workers=2
    )
    assert parallel.params == serial.params
    assert parallel.cv_score == pytest.approx(serial.cv_score)


@pytest.mark.unit
def test_short_window_gaps_are_imputed_with_training_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gridiron_edge.evaluation import feature_sweep_run as run_mod

    rows = _rows("win")
    spec = _spec("win", GameModelType.LOGISTIC, "combined")

    def fake_rebuild(frame: DataFrame, *, window: int, repo: Path) -> DataFrame:
        out = frame.copy()
        out.loc[out.index[::7], "HOME_OFF_REDZONE_TD_PCT"] = np.nan
        return out

    monkeypatch.setattr(run_mod, "_rebuild_features_with_window", fake_rebuild)
    provider = make_feature_provider(rows, spec, Path("."))
    out = provider(1)
    assert out.loc[rows.matched].notna().all().all()
    gap_train = rows.train_mask & (rows.frame.index % 7 == 0)
    train_median = rows.frame.loc[rows.train_mask & ~gap_train, "HOME_OFF_REDZONE_TD_PCT"].median()
    assert np.allclose(out.loc[gap_train, "HOME_OFF_REDZONE_TD_PCT"], train_median)
    assert provider(4) is rows.features


class _StopError(Exception):
    """Simulated interruption after a checkpoint."""


@pytest.mark.unit
def test_run_family_resumes_from_partial_checkpoint() -> None:
    rows = _rows("total")
    spec = _spec("total", GameModelType.XGBOOST)
    steps = build_sweep_steps("expanded")
    provider = make_feature_provider(rows, spec, Path("."))
    saved: list[FamilyRun] = []

    def stop_after_two(run: FamilyRun) -> None:
        saved.append(run)
        if len(run.steps) == 2:
            raise _StopError

    with pytest.raises(_StopError):
        run_family(spec, rows, steps, provider, placebo_draws=1, checkpoint=stop_after_two)
    assert len(saved[-1].steps) == 2

    seen: list[str] = []
    resumed = run_family(
        spec,
        rows,
        steps,
        provider,
        placebo_draws=1,
        resume=saved[-1],
        on_step=lambda _f, step, _s: seen.append(step),
    )
    assert seen == [s.name for s in steps[2:]]
    assert [s.step for s in resumed.steps] == [s.name for s in steps]
    assert set(resumed.placebo) == {s.name for s in steps[1:]}
    assert evaluate_family_run(resumed, rows).batches


@pytest.mark.unit
def test_ablation_scores_on_the_screening_window_and_flags_dead_weight() -> None:
    from gridiron_edge.evaluation.feature_sweep_ablation import (
        AblationGroup,
        group_columns,
        run_ablation,
    )

    rows = _rows("total")
    spec = _spec("total", GameModelType.XGBOOST)
    cols = list(build_sweep_steps("expanded")[-1].columns)
    outcome = search_hyperparameters(spec, lambda _w: rows.features, cols, rows.y, rows.train_mask)
    groups = (
        AblationGroup("noise_group", ("GLICKO2_RD",)),
        AblationGroup("signal_group", ("OFF_EPA_PER_PLAY",)),
        AblationGroup("absent_group", ("NOT_A_METRIC",)),
    )
    assert len(group_columns(groups[0], cols)) == 3
    assert group_columns(groups[2], cols) == []
    out = run_ablation(
        spec,
        rows,
        lambda _w: rows.features,
        outcome,
        cols,
        groups=groups,
        screen_starts=(2022,),
    )
    assert set(out) == {"noise_group", "signal_group"}
    assert out["noise_group"].n_columns_dropped == 3
    assert out["noise_group"].delta.n_games == int(
        (rows.matched & (rows.frame["YEAR"] == "2022-2023")).sum()
    )
    # ACTUAL_TOTAL is driven by AWAY_OFF_EPA_PER_PLAY: dropping it must hurt more.
    assert out["signal_group"].harm > out["noise_group"].harm
    assert not out["signal_group"].removable
