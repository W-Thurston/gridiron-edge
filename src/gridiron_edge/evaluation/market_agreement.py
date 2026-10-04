# src/gridiron_edge/evaluation/market_agreement.py

"""Test whether the model beats the closing market where it disagrees (``DECISIONS.md`` D85).

A raw model-versus-market EV treats the model's probability or total as the truth. This module
tests that on walk-forward backfilled forecasts against the closing lines:

* the model's score against the market's (Brier for win probability, MAE for totals);
* the *calibration slope* of the outcome on the model-minus-market gap: the outcome moves by
  ``slope`` times the gap, so 0 means the market is right and the disagreement is noise, and 1
  means the model is right. The slope is fitted through the origin with a robust standard error;
* realized outcomes by gap bin.

Raw model EV is trustworthy only if the slope's 95% interval excludes 0 and includes 1. The rule
was recorded before it was applied; the thresholds in the bins and the ``VEGAS_LINE`` reading
were chosen after seeing the data.

Public API
----------
agreement_tables    games, forecast events -> FamilyAgreement
build_report        families -> Markdown text
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from math import erf, sqrt

import numpy as np
from pandas import DataFrame

FIRST_SEASON = 2013
#: NFL game-margin standard deviation; only turns the closing spread into a win probability.
MARGIN_STD = 13.5
Z_95 = 1.96
WIN_GAP_EDGES = (-1.0, -0.10, -0.05, 0.05, 0.10, 1.0)
TOTAL_GAP_EDGES = (-np.inf, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, np.inf)
_GAME_COLUMNS = ("GAME_ID", "YEAR", "AWAY_SCORE", "HOME_SCORE", "VEGAS_LINE", "OVER_UNDER")


@dataclass(frozen=True, slots=True)
class GapBin:
    """Realized outcomes for games whose model-minus-market gap falls in one bin."""

    label: str
    games: int
    mean_model: float
    mean_market: float
    realized: float


@dataclass(frozen=True, slots=True)
class FamilyAgreement:
    """How one forecast family compares with the closing market."""

    name: str
    kind: str
    games: int
    model_score: float
    market_score: float
    score_label: str
    slope: float
    slope_low: float
    slope_high: float
    bins: tuple[GapBin, ...]

    @property
    def validated(self) -> bool:
        """Whether the slope's 95% interval excludes 0 and includes 1."""
        return self.slope_low > 0.0 and self.slope_low <= 1.0 <= self.slope_high


def _phi(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _slope(gap: np.ndarray, outcome: np.ndarray) -> tuple[float, float, float]:
    """Fit ``outcome = slope * gap`` through the origin; return slope and its 95% interval."""
    denominator = float(np.sum(gap**2))
    if denominator == 0.0:
        raise ValueError("The model never disagrees with the market; the slope is undefined.")
    slope = float(np.sum(gap * outcome)) / denominator
    residual = outcome - slope * gap
    standard_error = sqrt(float(np.sum((gap * residual) ** 2))) / denominator
    return slope, slope - Z_95 * standard_error, slope + Z_95 * standard_error


def _label(low: float, high: float) -> str:
    def fmt(value: float) -> str:
        return "∞" if np.isinf(value) else f"{value:+g}"

    return f"{fmt(low)} to {fmt(high)}"


def _joined(games: DataFrame, events: DataFrame, column: str) -> DataFrame:
    missing = sorted(set(_GAME_COLUMNS) - set(games.columns))
    if missing:
        raise ValueError("Games are missing columns: " + ", ".join(missing))
    scored = games.loc[
        games["AWAY_SCORE"].notna() & games["HOME_SCORE"].notna() & games["VEGAS_LINE"].notna()
    ].copy()
    scored = scored.loc[scored["YEAR"].astype(str).str[:4].astype(int) >= FIRST_SEASON]
    joined = events.loc[:, ["game_id", column]].merge(
        scored, left_on="game_id", right_on="GAME_ID", validate="one_to_one"
    )
    joined["margin"] = joined["HOME_SCORE"] - joined["AWAY_SCORE"]
    joined["total"] = joined["HOME_SCORE"] + joined["AWAY_SCORE"]
    return joined


def win_agreement(name: str, games: DataFrame, events: DataFrame) -> FamilyAgreement:
    """Compare a win-probability family with the closing spread's implied probability.

    ``VEGAS_LINE`` is the expected home margin, so the market probability is the normal
    probability of a positive margin with that mean.
    """
    joined = _joined(games, events, "home_win_prob")
    joined = joined.loc[joined["margin"] != 0]
    market = np.array([_phi(line / MARGIN_STD) for line in joined["VEGAS_LINE"]])
    model = joined["home_win_prob"].astype(float).to_numpy()
    won = (joined["margin"] > 0).to_numpy().astype(float)
    slope, low, high = _slope(model - market, won - market)
    gap = model - market
    bins = []
    for lower, upper in pairwise(WIN_GAP_EDGES):
        mask = (gap > lower) & (gap <= upper)
        if mask.any():
            bins.append(
                GapBin(
                    _label(lower, upper),
                    int(mask.sum()),
                    float(model[mask].mean()),
                    float(market[mask].mean()),
                    float(won[mask].mean()),
                )
            )
    return FamilyAgreement(
        name=name,
        kind="win",
        games=len(joined),
        model_score=float(np.mean((model - won) ** 2)),
        market_score=float(np.mean((market - won) ** 2)),
        score_label="Brier",
        slope=slope,
        slope_low=low,
        slope_high=high,
        bins=tuple(bins),
    )


def total_agreement(name: str, games: DataFrame, events: DataFrame) -> FamilyAgreement:
    """Compare a total family with the closing over/under line."""
    joined = _joined(games, events, "model_total")
    joined = joined.loc[joined["OVER_UNDER"].notna()]
    line = joined["OVER_UNDER"].astype(float).to_numpy()
    model = joined["model_total"].astype(float).to_numpy()
    actual = joined["total"].astype(float).to_numpy()
    gap = model - line
    slope, low, high = _slope(gap, actual - line)
    bins = []
    for lower, upper in pairwise(TOTAL_GAP_EDGES):
        mask = (gap > lower) & (gap <= upper)
        if mask.any():
            bins.append(
                GapBin(
                    _label(lower, upper),
                    int(mask.sum()),
                    float(gap[mask].mean()),
                    0.0,
                    float((actual[mask] - line[mask]).mean()),
                )
            )
    return FamilyAgreement(
        name=name,
        kind="total",
        games=len(joined),
        model_score=float(np.mean(np.abs(model - actual))),
        market_score=float(np.mean(np.abs(line - actual))),
        score_label="MAE",
        slope=slope,
        slope_low=low,
        slope_high=high,
        bins=tuple(bins),
    )


def agreement_tables(
    games: DataFrame, families: dict[str, tuple[str, DataFrame]]
) -> list[FamilyAgreement]:
    """Score each family; ``families`` maps a display name to ``(kind, events)``."""
    results: list[FamilyAgreement] = []
    for name, (kind, events) in families.items():
        if kind == "win":
            results.append(win_agreement(name, games, events))
        elif kind == "total":
            results.append(total_agreement(name, games, events))
        else:
            raise ValueError(f"Unknown family kind {kind!r}.")
    return results


def build_report(families: list[FamilyAgreement], *, generated_on: date | None = None) -> str:
    """Render the agreement tables and the validation verdicts as Markdown."""
    day = (generated_on or date.today()).isoformat()
    lines = [
        "# Does the model beat the closing market where they disagree?",
        "",
        f"Generated {day} by `gridiron evaluate market-agreement` (`DECISIONS.md` D85).",
        "",
        "## Measure",
        "",
        "Walk-forward backfilled forecasts, "
        f"{FIRST_SEASON} onward, against the closing lines in the cleaned games file. For win "
        "probability the market is the closing spread's implied probability (`VEGAS_LINE` is the "
        "expected home margin; a normal margin with standard deviation "
        f"{MARGIN_STD:g}); for totals it is the closing over/under. The *gap* is model minus "
        "market. The *slope* is how much of the gap shows up in the outcome (fitted through "
        "the origin, robust standard error): 0 means the market is right and the disagreement is "
        "noise, 1 means the model is right.",
        "",
        "**Rule (recorded before it was applied):** raw model EV is trustworthy only if the "
        "slope's 95% interval excludes 0 and includes 1. The gap bins and the reading of "
        "`VEGAS_LINE` were chosen after seeing the data. Ties and pushes are excluded.",
        "",
        "## Summary",
        "",
        "| Family | Games | Model | Market | Slope | 95% interval | Validated |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in families:
        lines.append(
            f"| {item.name} | {item.games} | {item.score_label} {item.model_score:.4f} | "
            f"{item.score_label} {item.market_score:.4f} | {item.slope:.3f} | "
            f"{item.slope_low:.3f} to {item.slope_high:.3f} | {'yes' if item.validated else 'no'} |"
        )
    for item in families:
        lines += ["", f"## {item.name}", ""]
        if item.kind == "win":
            lines += [
                "| Model minus market | Games | Mean model | Mean market | Realized home win |",
                "|---|---|---|---|---|",
            ]
            lines += [
                f"| {b.label} | {b.games} | {b.mean_model:.3f} | {b.mean_market:.3f} | "
                f"{b.realized:.3f} |"
                for b in item.bins
            ]
        else:
            lines += [
                "| Model minus line (points) | Games | Mean gap | Mean actual minus line |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| {b.label} | {b.games} | {b.mean_model:+.2f} | {b.realized:+.2f} |"
                for b in item.bins
            ]
    passed = [item.name for item in families if item.validated]
    lines += [
        "",
        "## Verdict",
        "",
        (
            "Families that pass the rule: " + ", ".join(passed) + "."
            if passed
            else "No family passes the rule: where the model disagrees with the closing market, "
            "outcomes follow the market. Raw model EV is not an edge."
        ),
        "",
    ]
    return "\n".join(lines)


def prediction_frame(events: DataFrame, column: str) -> DataFrame:
    """Reduce a family's backfilled events to one row per game."""
    if events.empty:
        raise ValueError("No forecast events.")
    return (
        events.sort_values("generated_at")
        .drop_duplicates("game_id", keep="last")
        .reset_index(drop=True)[["game_id", column]]
        .pipe(lambda frame: frame.astype({column: float}))
    )


__all__ = [
    "FamilyAgreement",
    "GapBin",
    "agreement_tables",
    "build_report",
    "prediction_frame",
    "total_agreement",
    "win_agreement",
]
