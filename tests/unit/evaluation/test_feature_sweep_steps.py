# tests/unit/evaluation/test_feature_sweep_steps.py

"""Tests for the incremental feature-sweep step definitions."""

from __future__ import annotations

from itertools import pairwise

import pytest

from gridiron_edge.evaluation.feature_sweep_steps import (
    BASELINE_STEP_NAME,
    FEATURE_BATCHES,
    FeatureBatch,
    batch_columns,
    build_sweep_steps,
)
from gridiron_edge.models.game_prediction._columns import (
    _COMBINED_FEATURES,
    _EXPANDED_FEATURES,
)

EXPANDED_COUNTS = [152, 167, 186, 211, 239, 260]
COMBINED_COUNTS = [111, 126, 144, 165, 186, 207]


@pytest.mark.unit
def test_expanded_step_counts_match_wired_compositions() -> None:
    steps = build_sweep_steps("expanded")
    assert [len(s.columns) for s in steps] == EXPANDED_COUNTS
    assert steps[0].name == BASELINE_STEP_NAME
    assert [s.name for s in steps[1:]] == [b.name for b in FEATURE_BATCHES]


@pytest.mark.unit
def test_combined_step_counts_match_wired_compositions() -> None:
    steps = build_sweep_steps("combined")
    assert [len(s.columns) for s in steps] == COMBINED_COUNTS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "declared"), [("expanded", _EXPANDED_FEATURES), ("combined", _COMBINED_FEATURES)]
)
def test_steps_are_nested_ordered_and_end_at_current_declaration(
    key: str, declared: list[str]
) -> None:
    steps = build_sweep_steps(key)
    for earlier, later in pairwise(steps):
        assert set(earlier.columns) < set(later.columns)
    assert list(steps[-1].columns) == declared
    for step in steps:
        assert list(step.columns) == [c for c in declared if c in set(step.columns)]
        assert len(set(step.columns)) == len(step.columns)


@pytest.mark.unit
def test_batches_are_disjoint_and_named_cumulatively() -> None:
    steps = build_sweep_steps("expanded")
    assert steps[-1].batches == tuple(b.name for b in FEATURE_BATCHES)
    owned = [set(batch_columns(b, list(_EXPANDED_FEATURES))) for b in FEATURE_BATCHES]
    for i, a in enumerate(owned):
        for b in owned[i + 1 :]:
            assert not a & b


@pytest.mark.unit
def test_baseline_excludes_every_batch_column() -> None:
    baseline = set(build_sweep_steps("expanded")[0].columns)
    for batch in FEATURE_BATCHES:
        assert not baseline & set(batch_columns(batch, list(_EXPANDED_FEATURES)))
    assert {"AWAY_ELO", "ELO_DIFF", "WEEK_NUM", "IS_DOME"} & baseline >= {
        "AWAY_ELO",
        "ELO_DIFF",
        "WEEK_NUM",
    }


@pytest.mark.unit
def test_batch_columns_expands_diff_and_sides() -> None:
    batch = FeatureBatch("x", metrics=("GLICKO2_RD",), single_columns=("WEEK_NUM",))
    cols = batch_columns(batch, list(_EXPANDED_FEATURES))
    assert set(cols) == {"GLICKO2_RD_DIFF", "AWAY_GLICKO2_RD", "HOME_GLICKO2_RD", "WEEK_NUM"}


@pytest.mark.unit
def test_empty_batch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from gridiron_edge.evaluation import feature_sweep_steps as mod

    monkeypatch.setattr(mod, "FEATURE_BATCHES", [FeatureBatch("ghost", metrics=("NOPE",))])
    with pytest.raises(ValueError, match="contributes no columns"):
        mod.build_sweep_steps("expanded")


@pytest.mark.unit
def test_overlapping_batches_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from gridiron_edge.evaluation import feature_sweep_steps as mod

    monkeypatch.setattr(
        mod,
        "FEATURE_BATCHES",
        [
            FeatureBatch("a", metrics=("GLICKO2_RD",)),
            FeatureBatch("b", metrics=("GLICKO2_RD",)),
        ],
    )
    with pytest.raises(ValueError, match="overlaps"):
        mod.build_sweep_steps("expanded")


@pytest.mark.unit
def test_unsupported_feature_set_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_sweep_steps("raw")
