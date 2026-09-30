# src/gridiron_edge/evaluation/feature_sweep_steps.py

"""Incremental feature-batch steps for the consolidated feature sweep.

Pure-data leaf - no I/O, no training. Defines the ordered feature batches
that were wired into the canonical Win/Total feature set after the baseline
composition, and derives each cumulative step's exact column list from the
current column declarations (``DECISIONS.md`` D62). Batches are identified
by the per-team metric each one contributes; every metric expands to its
Home-minus-Away differential and its Away and Home raw columns when those
exist in the current declaration.

Public API
----------
FeatureBatch            dataclass   - one wired batch and its metrics
FeatureStep             dataclass   - one cumulative sweep step
FEATURE_BATCHES         list        - wiring-ordered batches
BASELINE_STEP_NAME      str         - name of the pre-batch step
batch_columns           batch -> ordered current columns it contributes
build_sweep_steps       feature-set key -> ordered cumulative steps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from gridiron_edge.models.game_prediction._columns import (
    _COMBINED_FEATURES,
    _EXPANDED_FEATURES,
)

BASELINE_STEP_NAME: Final[str] = "baseline"

#: Batches the sweep rejected and a forward schema bump removed from the
#: declared set (``DECISIONS.md`` D72). They contribute no current columns, so
#: ``build_sweep_steps`` skips them instead of treating them as stale.
RETIRED_BATCHES: Final[frozenset[str]] = frozenset(
    {
        "construction_fixes",
        "opponent_adjusted_epa",
        "situational_context",
        "rating_ensemble",
    }
)


@dataclass(frozen=True)
class FeatureBatch:
    """One wired feature batch.

    Attributes:
        name: Descriptive slug used in reports.
        metrics: Per-team metric names (differential/side suffixes removed).
        single_columns: Matchup-level columns that exist once per game.
    """

    name: str
    metrics: tuple[str, ...]
    single_columns: tuple[str, ...] = ()


#: Wiring-ordered batches. Each batch is the exact column delta of its
#: wiring commit, verified against the schema-5..12 declarations.
FEATURE_BATCHES: Final[list[FeatureBatch]] = [
    FeatureBatch(
        name="construction_fixes",
        metrics=(
            "OFF_CPOE",
            "OFF_CPOE_AVAILABLE",
            "DEF_CPOE",
            "DEF_CPOE_AVAILABLE",
            "OFF_FUMBLE_RATE",
        ),
    ),
    FeatureBatch(
        name="opponent_adjusted_epa",
        metrics=(
            "OFF_PASS_EPA_ADJ",
            "OFF_RUSH_EPA_ADJ",
            "DEF_PASS_EPA_ADJ",
            "DEF_RUSH_EPA_ADJ",
            "PASS_MATCHUP_EPA",
            "RUSH_MATCHUP_EPA",
        ),
        single_columns=("PROJECTED_TOTAL_PLAYS",),
    ),
    FeatureBatch(
        name="qb_and_coach",
        metrics=(
            "QB_EPA_PLAY_CAREER",
            "QB_EPA_PLAY_ROLLING",
            "QB_CPOE_ROLLING",
            "QB_CPOE_ROLLING_AVAILABLE",
            "QB_EXPERIENCE",
            "QB_CHANGE_FLAG",
            "HC_WIN_PCT_CAREER",
            "HC_TENURE",
            "NEW_COACHING_STAFF_FLAG",
        ),
    ),
    FeatureBatch(
        name="situational_context",
        metrics=(
            "BACK_TO_BACK_ROAD",
            "BODY_CLOCK_KICKOFF_HOUR",
            "FORM_TREND",
            "ONE_SCORE_GAMES",
            "ONE_SCORE_WIN_PCT",
            "PUNT_EPA_PER_PLAY",
            "PYTHAG_LUCK_GAP",
            "PYTHAG_WIN_PCT",
            "ST_EPA_PER_PLAY",
        ),
        single_columns=("IS_ARTIFICIAL_TURF", "LEAGUE_PPG_TO_DATE", "WIND_SPEED_BIN"),
    ),
    FeatureBatch(
        name="rating_ensemble",
        metrics=(
            "COLLEY_RATING",
            "GLICKO2_RATING",
            "GLICKO2_RD",
            "KEENER_RATING",
            "MASSEY_RATING",
            "QB_ELO",
            "QB_ELO_STARTS",
        ),
    ),
]


@dataclass(frozen=True)
class FeatureStep:
    """One cumulative sweep step.

    Attributes:
        name: ``"baseline"`` or the name of the newest batch in the step.
        batches: Names of every batch included, in wiring order.
        columns: Ordered feature columns for the step.
    """

    name: str
    batches: tuple[str, ...]
    columns: tuple[str, ...]


def _declared_columns(feature_set_key: str) -> list[str]:
    if feature_set_key == "combined":
        return list(_COMBINED_FEATURES)
    if feature_set_key == "expanded":
        return list(_EXPANDED_FEATURES)
    msg: str = f"Unsupported feature set for the sweep: {feature_set_key!r}"
    raise ValueError(msg)


def batch_columns(batch: FeatureBatch, declared: list[str]) -> list[str]:
    """Return the ordered columns of ``declared`` that ``batch`` contributes.

    Args:
        batch: The batch to expand.
        declared: Current ordered column declaration (combined or expanded).

    Returns:
        Columns of ``declared`` belonging to ``batch``, in declaration order.
    """
    wanted: set[str] = set(batch.single_columns)
    for metric in batch.metrics:
        wanted.update({f"{metric}_DIFF", f"AWAY_{metric}", f"HOME_{metric}"})
    return [column for column in declared if column in wanted]


def build_sweep_steps(feature_set_key: str) -> list[FeatureStep]:
    """Build the baseline and cumulative steps for a feature set.

    The baseline is every declared column that no batch contributes, so it
    reproduces the pre-batch composition exactly. Raises if a batch names a
    column that is not declared (a stale definition), so a schema change can
    never silently shrink a step.

    Args:
        feature_set_key: ``"combined"`` (Logistic) or ``"expanded"`` (trees,
            Total).

    Returns:
        Steps in wiring order, baseline first.

    Raises:
        ValueError: If the set is unsupported, a batch contributes no columns
            or overlaps another batch.
    """
    declared: list[str] = _declared_columns(feature_set_key)
    owned: dict[str, set[str]] = {}
    seen: set[str] = set()
    for batch in FEATURE_BATCHES:
        cols: list[str] = batch_columns(batch, declared)
        if not cols and batch.name in RETIRED_BATCHES:
            continue
        if not cols:
            msg = f"Batch {batch.name!r} contributes no columns to {feature_set_key!r}"
            raise ValueError(msg)
        missing_single = [c for c in batch.single_columns if c not in declared]
        if feature_set_key == "expanded" and missing_single:
            msg = f"Batch {batch.name!r} names undeclared columns: {missing_single}"
            raise ValueError(msg)
        overlap: set[str] = seen & set(cols)
        if overlap:
            msg = f"Batch {batch.name!r} overlaps earlier batches: {sorted(overlap)}"
            raise ValueError(msg)
        seen.update(cols)
        owned[batch.name] = set(cols)

    baseline_cols: tuple[str, ...] = tuple(c for c in declared if c not in seen)
    steps: list[FeatureStep] = [FeatureStep(BASELINE_STEP_NAME, (), baseline_cols)]
    included: set[str] = set()
    included_names: list[str] = []
    for batch in FEATURE_BATCHES:
        if batch.name not in owned:
            continue
        included |= owned[batch.name]
        included_names.append(batch.name)
        columns = tuple(c for c in declared if c in included or c in set(baseline_cols))
        steps.append(FeatureStep(batch.name, tuple(included_names), columns))
    return steps
