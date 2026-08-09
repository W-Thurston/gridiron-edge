"""Pure Closing Line Value calculations and summaries.

This module does not select opening or closing quotes from historical market
observations. Callers must supply CLV values produced by a separately validated,
same-source, sportsbook-specific, pre-kickoff quote-selection policy.
"""

from __future__ import annotations

import math

import pandas as pd
from pandas import DataFrame, Series


def closing_line_value(
    bet_fair_prob: float,
    close_fair_prob: float,
) -> float:
    """Return probability-based CLV from validated bet and close probabilities."""
    _validate_prob(bet_fair_prob, "bet_fair_prob")
    _validate_prob(close_fair_prob, "close_fair_prob")
    return (close_fair_prob - bet_fair_prob) / bet_fair_prob


def spread_clv(
    bet_spread: float,
    close_spread: float,
    side: str,
) -> float:
    """Return point movement for a spread wager from validated observations."""
    if side == "home":
        return bet_spread - close_spread
    if side == "away":
        return close_spread - bet_spread
    raise ValueError(f"side must be 'home' or 'away', got {side!r}")


def total_clv(
    bet_total: float,
    close_total: float,
    side: str,
) -> float:
    """Return point movement for a total wager from validated observations."""
    if side == "over":
        return close_total - bet_total
    if side == "under":
        return bet_total - close_total
    raise ValueError(f"side must be 'over' or 'under', got {side!r}")


def summarize_clv(clv_report_df: DataFrame) -> dict[str, float]:
    """Summarize CLV values already produced by a validated closeout policy."""
    if "clv" not in clv_report_df.columns:
        return _empty_clv_summary()

    clv_col: Series = clv_report_df.loc[:, "clv"]
    numeric: Series = clv_col.apply(lambda value: pd.to_numeric(value, errors="coerce"))
    valid: Series = numeric.dropna()
    if valid.empty:
        return _empty_clv_summary()

    return {
        "mean_clv": float(valid.mean()),
        "median_clv": float(valid.median()),
        "pct_positive_clv": float((valid > 0).mean()),
        "n_edges": float(len(valid)),
    }


def _empty_summary() -> dict[str, float]:
    """Return an unavailable CLV summary."""
    return {
        "mean_clv": math.nan,
        "median_clv": math.nan,
        "pct_positive_clv": math.nan,
        "n_edges": 0.0,
    }


def _empty_clv_summary() -> dict[str, float]:
    """Return an unavailable summary when no validated CLV values exist."""
    return {
        "mean_clv": float("nan"),
        "median_clv": float("nan"),
        "pct_positive_clv": float("nan"),
        "n_edges": 0.0,
    }


def _validate_prob(probability: float, name: str) -> None:
    """Require one probability strictly between zero and one."""
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {probability}")
