"""Domain heuristics for the evaluate report CLI surface.

Pure functions over evaluation DataFrames. The CLI layer
(``cli/evaluate.py``) calls into these helpers and is
responsible only for rendering.

Each heuristic returns either ``None`` (no notable signal)
or a small dataclass carrying the values needed to render
the corresponding warning or summary line. Thresholds are
declared at module scope so they can be reviewed in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass

from pandas import DataFrame, Series

# ---------------------------------------------------------------------------
# Thresholds — kept at module scope for discoverability.
# ---------------------------------------------------------------------------

#: Confidence threshold above which we look for overconfidence
#: behaviour. Predicted probability bins at or above this value are
#: considered the high-confidence regime.
HIGH_CONFIDENCE_THRESHOLD: float = 0.75

#: Absolute calibration gap above which a high-confidence tier is
#: flagged as miscalibrated.
HIGH_CONFIDENCE_GAP_THRESHOLD: float = 0.03

#: Week number at or below which a miss counts as "early season".
EARLY_SEASON_WEEK_THRESHOLD: int = 3

#: Minimum number of worst misses falling in early-season weeks
#: required to surface an early-season-instability flag.
EARLY_SEASON_MISS_COUNT_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HighConfidenceFlag:
    """A high-confidence tier whose calibration gap exceeds the threshold."""

    confidence_tier: str
    predicted_avg: float
    actual_win_rate: float
    calibration_gap: float
    direction: str  # "overconfident" or "underconfident"


@dataclass(frozen=True)
class DriftFlag:
    """A season whose Brier deviates far enough from the long-run mean."""

    season: str
    delta_vs_mean: float


@dataclass(frozen=True)
class EarlySeasonMissFlag:
    """The early-season slice of the top-N misses."""

    n_early: int
    top_misses: int


@dataclass(frozen=True)
class OverconfidenceMissFlag:
    """The overconfidence pattern of losses-as-predicted-favorite among misses."""

    n_losses: int
    top_misses: int


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------


def find_high_confidence_warning(
    df_tiers: DataFrame,
) -> HighConfidenceFlag | None:
    """Surface an overconfidence flag for the worst high-confidence tier.

    Returns ``None`` if there are no predictions at or above
    :data:`HIGH_CONFIDENCE_THRESHOLD` predicted probability, or if the
    worst calibration gap is below :data:`HIGH_CONFIDENCE_GAP_THRESHOLD`.
    """
    high_conf: DataFrame = df_tiers.loc[
        df_tiers["predicted_avg"] >= HIGH_CONFIDENCE_THRESHOLD,
        :,
    ]
    if high_conf.empty:
        return None

    worst_gap: float = float(high_conf["calibration_gap"].abs().max())
    if worst_gap < HIGH_CONFIDENCE_GAP_THRESHOLD:
        return None

    worst_idx = high_conf["calibration_gap"].abs().idxmax()
    worst_row: Series = high_conf.loc[[worst_idx]].iloc[0]
    direction: str = "overconfident" if worst_row["calibration_gap"] > 0 else "underconfident"
    return HighConfidenceFlag(
        confidence_tier=str(worst_row["confidence_tier"]),
        predicted_avg=float(worst_row["predicted_avg"]),
        actual_win_rate=float(worst_row["actual_win_rate"]),
        calibration_gap=float(worst_row["calibration_gap"]),
        direction=direction,
    )


def find_season_drift_warning(
    df_seasons: DataFrame,
) -> DriftFlag | None:
    """Surface the worst season whose Brier deviates from the long-run mean.

    Returns ``None`` if no season is flagged as a drift candidate.
    """
    warn_seasons: DataFrame = df_seasons.loc[
        df_seasons["trend"] == "⚠",
        :,
    ]
    if warn_seasons.empty:
        return None

    worst: Series = warn_seasons.sort_values(
        "delta_vs_mean",
        ascending=False,
    ).iloc[0]
    return DriftFlag(
        season=str(worst["season"]),
        delta_vs_mean=float(worst["delta_vs_mean"]),
    )


def find_early_season_miss_pattern(
    df_misses: DataFrame,
    top_misses: int,
) -> EarlySeasonMissFlag | None:
    """Surface the early-season instability pattern among the worst misses."""
    early_mask: Series = df_misses["week"] <= EARLY_SEASON_WEEK_THRESHOLD
    n_early: int = int(early_mask.sum())
    if n_early < EARLY_SEASON_MISS_COUNT_THRESHOLD:
        return None
    return EarlySeasonMissFlag(n_early=n_early, top_misses=top_misses)


def find_overconfidence_miss_pattern(
    df_misses: DataFrame,
    top_misses: int,
) -> OverconfidenceMissFlag | None:
    """Surface the overconfidence pattern of losses-as-favorite among misses."""
    loss_misses: DataFrame = df_misses.loc[
        df_misses["actual_result"] == "LOSS",
        :,
    ]
    n_losses: int = len(loss_misses)
    if n_losses < (top_misses // 2):
        return None
    return OverconfidenceMissFlag(n_losses=n_losses, top_misses=top_misses)
