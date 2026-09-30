# tests/unit/evaluation/test_feature_sweep_rows.py

"""Tests for row-matched sweep data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.feature_sweep_rows import build_matched_rows
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.models.game_prediction._columns import (
    _GAME_FEATURES,
    _RAW_FEATURES,
    _TEAM_FEATURES,
)

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"]


def _frame(rows_per_season: int = 20) -> DataFrame:
    rng = np.random.default_rng(0)
    n = rows_per_season * len(SEASONS)
    values = {col: rng.normal(size=n) for col in [*_RAW_FEATURES, *_GAME_FEATURES, *_TEAM_FEATURES]}
    structure: list[dict[str, object]] = []
    for s, season in enumerate(SEASONS):
        for i in range(rows_per_season):
            structure.append(
                {
                    "GAME_ID": f"{season}_{i:03d}",
                    "YEAR": season,
                    "GAME_DATE": pd.Timestamp("2021-09-01") + pd.Timedelta(days=365 * s + i),
                    "HOME_WIN": float(i % 2),
                    "ACTUAL_TOTAL": 40.0 + i,
                }
            )
    df = pd.concat([DataFrame(values), DataFrame(structure)], axis=1)
    df["WEEK_NUM"] = [1 + i % 18 for i in range(rows_per_season)] * len(SEASONS)
    return df


@pytest.mark.unit
def test_matched_rows_drop_rows_missing_a_late_batch_feature() -> None:
    df = _frame()
    week1 = df["WEEK_NUM"] == 1
    df.loc[week1, "HOME_QB_EPA_PLAY_CAREER"] = np.nan  # a column of the kept batch
    out = build_matched_rows(df, "HOME_WIN")
    frame = out.frame
    assert not frame.loc[out.matched, "WEEK_NUM"].eq(1).any()
    steps = {c.step: c for c in out.own_pool}
    assert steps["baseline"].n_holdout > steps["qb_and_coach"].n_holdout
    assert out.n_holdout == steps["qb_and_coach"].n_holdout


@pytest.mark.unit
def test_holdout_is_exactly_holdout_seasons_and_train_precedes_it() -> None:
    out = build_matched_rows(_frame(), "ACTUAL_TOTAL")
    hold_years = set(out.frame.loc[out.holdout_mask, "YEAR"])
    train_years = set(out.frame.loc[out.train_mask, "YEAR"])
    assert hold_years == {"2023-2024", "2024-2025", "2025-2026"}
    assert train_years == {"2021-2022", "2022-2023"}
    assert "2026-2027" not in hold_years | train_years
    assert not (out.train_mask & out.holdout_mask).any()


@pytest.mark.unit
def test_every_step_is_complete_on_matched_rows() -> None:
    df = _frame()
    df.loc[df["WEEK_NUM"] <= 2, "AWAY_QB_EPA_PLAY_CAREER"] = np.nan
    out = build_matched_rows(df, "HOME_WIN")
    for step in build_sweep_steps("expanded"):
        assert out.features.loc[out.matched, list(step.columns)].notna().all().all()
    assert out.universe_holdout > out.n_holdout


@pytest.mark.unit
def test_null_target_rows_are_excluded() -> None:
    df = _frame()
    df.loc[df["GAME_ID"].str.endswith("_005"), "HOME_WIN"] = np.nan
    out = build_matched_rows(df, "HOME_WIN")
    assert not out.frame["GAME_ID"].str.endswith("_005").any()
    assert out.y.notna().all()


@pytest.mark.unit
def test_matched_rows_are_sorted_chronologically() -> None:
    df = _frame().sample(frac=1.0, random_state=1)
    out = build_matched_rows(df, "HOME_WIN")
    keys = out.frame[["YEAR", "WEEK_NUM", "GAME_DATE", "GAME_ID"]]
    assert keys.equals(keys.sort_values(list(keys.columns), kind="stable"))


@pytest.mark.unit
def test_missing_target_or_empty_split_raises() -> None:
    with pytest.raises(ValueError, match="missing target"):
        build_matched_rows(_frame().drop(columns=["HOME_WIN"]), "HOME_WIN")
    df = _frame()
    df.loc[df["YEAR"].isin(["2023-2024", "2024-2025", "2025-2026"]), "HOME_ELO"] = np.nan
    with pytest.raises(ValueError, match="empty train or holdout"):
        build_matched_rows(df, "HOME_WIN")
