# src/gridiron_edge/evaluation/quote_movement.py

"""How far a quote drifts as it ages, and the market age limit that follows (``DECISIONS.md`` D80).

The measure is the sportsbooks' de-vigged moneyline probability for the home team. For each
game, the consensus at one moment is the median across books. Two quantities are compared:

* *disagreement*: how far a single book sits from that consensus at one moment (the noise
  floor a bettor already faces when choosing among books);
* *drift*: how far the consensus moves between two snapshots of the same game, by the age
  between them.

The age limit is the lower edge of the first age bin whose median drift exceeds the median
disagreement: a quote older than that has moved by more than the books differ among
themselves. The rule was chosen after seeing the first weeks of data; the report says so.

Public API
----------
movement_tables     ledger rows -> QuoteMovement
build_report        QuoteMovement -> Markdown text
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import pandas as pd
from pandas import DataFrame

LAG_BIN_EDGES_HOURS: tuple[float, ...] = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0, np.inf)
MIN_PAIRS_PER_BIN = 30
# A bin whose median drift is within this many points of the disagreement is too close to call;
# a borderline result takes the lower edge, because blocking early only delays an edge while
# passing a stale quote can publish a false one. Chosen after seeing the first weeks' data.
BORDERLINE_MARGIN_POINTS = 0.05
_SNAPSHOT_KEYS = ["season", "week", "fetched_at", "game_id"]


@dataclass(frozen=True, slots=True)
class LagBin:
    """Drift statistics for pairs of snapshots separated by an age range."""

    lower_hours: float
    upper_hours: float
    pairs: int
    median_points: float
    p90_points: float


@dataclass(frozen=True, slots=True)
class QuoteMovement:
    """Disagreement, drift by age, and the age limit they imply."""

    games: int
    snapshots_by_week: dict[int, int]
    disagreement_median_points: float
    disagreement_p90_points: float
    cross_book_range_median_points: float
    bins: tuple[LagBin, ...]
    max_age_hours: float | None
    conservative_hours: float | None
    limit_reason: str


def _implied(odds: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0)), index=odds.index
    )


def _book_probabilities(ledger: DataFrame) -> DataFrame:
    """One de-vigged home probability per (snapshot, game, sportsbook)."""
    moneyline = ledger.loc[(ledger["market"] == "moneyline") & (~ledger["is_live"])].copy()
    if moneyline.empty:
        return DataFrame()
    moneyline["implied"] = _implied(moneyline["odds"])
    wide = moneyline.pivot_table(
        index=[*_SNAPSHOT_KEYS, "sportsbook", "commence_time"],
        columns="side",
        values="implied",
    ).reset_index()
    for side in ("home", "away"):
        if side not in wide.columns:
            wide[side] = np.nan
    wide = wide.dropna(subset=["home", "away"])
    wide["fair_home"] = wide["home"] / (wide["home"] + wide["away"])
    return wide


def _drift_pairs(consensus: DataFrame) -> DataFrame:
    rows: list[tuple[float, float]] = []
    for _, game in consensus.groupby("game_id"):
        ordered = game.sort_values("fetched_at").reset_index(drop=True)
        fetched = ordered["fetched_at"].to_numpy()
        ahead = (ordered["commence_time"] - ordered["fetched_at"]).to_numpy() / np.timedelta64(
            1, "h"
        )
        value = ordered["consensus"].to_numpy()
        for later in range(1, len(ordered)):
            if ahead[later] <= 0:
                continue
            for earlier in range(later):
                lag = (fetched[later] - fetched[earlier]) / np.timedelta64(1, "h")
                rows.append((float(lag), abs(float(value[later] - value[earlier]))))
    return DataFrame(rows, columns=["lag_hours", "drift"])


def _bins(pairs: DataFrame) -> tuple[LagBin, ...]:
    out: list[LagBin] = []
    for lower, upper in pairwise(LAG_BIN_EDGES_HOURS):
        chosen = pairs.loc[(pairs["lag_hours"] > lower) & (pairs["lag_hours"] <= upper), "drift"]
        out.append(
            LagBin(
                lower_hours=lower,
                upper_hours=upper,
                pairs=len(chosen),
                median_points=100.0 * float(chosen.median()) if len(chosen) else float("nan"),
                p90_points=100.0 * float(chosen.quantile(0.9)) if len(chosen) else float("nan"),
            )
        )
    return tuple(out)


def age_limit(
    bins: tuple[LagBin, ...], disagreement_median_points: float
) -> tuple[float | None, str]:
    """Return the age limit in hours and why, from the first bin that drifts past the books."""
    for item in bins:
        if item.pairs < MIN_PAIRS_PER_BIN:
            return (
                None,
                f"no limit: fewer than {MIN_PAIRS_PER_BIN} snapshot pairs at ages above "
                f"{item.lower_hours:g} h",
            )
        if item.median_points > disagreement_median_points:
            return (
                item.lower_hours,
                f"median drift {item.median_points:.2f} points at {item.lower_hours:g}-"
                f"{item.upper_hours:g} h exceeds the {disagreement_median_points:.2f}-point "
                "disagreement between books",
            )
    return None, "no limit: drift never exceeded the disagreement between books"


def conservative_limit(
    bins: tuple[LagBin, ...], limit: float | None, disagreement_median_points: float
) -> float | None:
    """Lower a borderline limit to the previous bin edge (see ``BORDERLINE_MARGIN_POINTS``)."""
    if limit is None:
        return None
    for previous, item in pairwise(bins):
        if (
            item.lower_hours == limit
            and previous.lower_hours < limit
            and previous.pairs >= MIN_PAIRS_PER_BIN
            and not np.isnan(previous.median_points)
            and previous.median_points >= disagreement_median_points - BORDERLINE_MARGIN_POINTS
        ):
            return previous.lower_hours
    return limit


def movement_tables(ledger: DataFrame) -> QuoteMovement:
    """Measure book disagreement and drift by age from the quote ledger."""
    books = _book_probabilities(ledger)
    if books.empty:
        raise ValueError("The quote ledger has no complete pregame moneyline pairs.")
    consensus = (
        books.groupby([*_SNAPSHOT_KEYS, "commence_time"])["fair_home"]
        .median()
        .rename("consensus")
        .reset_index()
    )
    scored = books.merge(consensus, on=[*_SNAPSHOT_KEYS, "commence_time"])
    deviation = (scored["fair_home"] - scored["consensus"]).abs()
    spread = scored.groupby(_SNAPSHOT_KEYS)["fair_home"].agg(lambda x: x.max() - x.min())
    bins = _bins(_drift_pairs(consensus))
    disagreement = 100.0 * float(deviation.median())
    limit, reason = age_limit(bins, disagreement)
    conservative = conservative_limit(bins, limit, disagreement)
    return QuoteMovement(
        games=int(consensus["game_id"].nunique()),
        snapshots_by_week={
            int(str(week)): int(count)
            for week, count in consensus.groupby("week")["fetched_at"].nunique().items()
        },
        disagreement_median_points=disagreement,
        disagreement_p90_points=100.0 * float(deviation.quantile(0.9)),
        cross_book_range_median_points=100.0 * float(spread.median()),
        bins=bins,
        max_age_hours=limit,
        conservative_hours=conservative,
        limit_reason=reason,
    )


def _upper(hours: float) -> str:
    return "∞" if np.isinf(hours) else f"{hours:g}"


def build_report(
    movement: QuoteMovement, *, configured_hours: float, generated_on: date | None = None
) -> str:
    """Render the movement tables and the age limit as Markdown."""
    day = (generated_on or date.today()).isoformat()
    derived = movement.max_age_hours
    recommended = movement.conservative_hours
    agree = recommended is not None and abs(recommended - configured_hours) < 1e-9
    lines = [
        "# Quote movement and the market age limit",
        "",
        f"Generated {day} by `gridiron evaluate quote-movement` (`DECISIONS.md` D80).",
        "",
        "## Measure",
        "",
        "The sportsbooks' de-vigged moneyline probability for the home team, from the quote",
        "ledger (pregame quotes only). The consensus is the median across books at one",
        "moment. *Disagreement* is how far one book sits from the consensus at one moment.",
        "*Drift* is how far the consensus moves between two snapshots of the same game, by the",
        "age between them. Both are in percentage points of win probability; at even odds an",
        "expected-value change is twice this.",
        "",
        "## Data",
        "",
        f"{movement.games} games; snapshots per week: "
        + ", ".join(f"week {w}: {n}" for w, n in sorted(movement.snapshots_by_week.items()))
        + ". Pairs of snapshots of one game overlap, so they are not independent observations.",
        "",
        "## Disagreement between books at one moment",
        "",
        f"Median distance from the consensus {movement.disagreement_median_points:.2f} points "
        f"(90th percentile {movement.disagreement_p90_points:.2f}); median spread between the "
        f"highest and lowest book {movement.cross_book_range_median_points:.2f} points.",
        "",
        "## Drift by age",
        "",
        "| Age between snapshots (h) | Pairs | Median drift (points) | 90th percentile |",
        "|---|---|---|---|",
    ]
    for item in movement.bins:
        lines.append(
            f"| {item.lower_hours:g}-{_upper(item.upper_hours)} | {item.pairs} | "
            f"{item.median_points:.2f} | {item.p90_points:.2f} |"
        )
    lines += [
        "",
        "## Age limit",
        "",
        "Rule: the limit is the lower edge of the first age bin whose median drift exceeds the",
        "median disagreement between books, requiring at least "
        f"{MIN_PAIRS_PER_BIN} pairs per bin. A quote older than that has moved by more than the",
        "books differ among themselves.",
        "",
        f"**Derived by the rule:** {'none' if derived is None else f'{derived:g} hours'} "
        f"({movement.limit_reason}).",
        (
            "**Taken (borderline rule):** "
            + ("none" if recommended is None else f"{recommended:g} hours")
            + (
                f". The bin below the derived limit has a median within {BORDERLINE_MARGIN_POINTS} "
                "points of the disagreement, so the lower bin edge is taken: blocking early only "
                "delays an edge, while passing a stale quote can publish a false one."
                if recommended != derived
                else "."
            )
        ),
        f"**Configured (`MAX_MARKET_AGE`):** {configured_hours:g} hours"
        + (" (matches)." if agree else " (**does not match the limit taken above**)."),
        "",
        "## How this limit was chosen, and what it is not",
        "",
        "The rule was written after reading the first dense weeks of the ledger (week 1 from",
        "the original collector, then weeks 3 and 4), so it is a post-hoc criterion, not a",
        "pre-declared one; the weeks collected from now on are the out-of-sample check, and this",
        "command should be rerun as they arrive. The bins are coarse: the derived limit is a bin",
        "edge, and the median drift just above it is close to the disagreement. It is a single",
        "age for every market, derived from moneylines; spreads and totals move less.",
        "",
    ]
    return "\n".join(lines)
