# src/gridiron_edge/betting/performance.py
"""Betting performance analytics — pure DataFrame-in, results-out.

All functions accept a bets DataFrame (from ``load_bets()``) and return
dicts or DataFrames.  No file I/O — the caller is responsible for loading
the ledger.

Public API::

    record(bets)            W-L-P counts, overall or by split
    roi(bets)               ROI %, overall or by split
    clv_summary(bets)       Mean/median CLV, % positive
    ev_analysis(bets)       Model EV vs actual performance
    streak_analysis(bets)   Current streak, longest W/L streaks
    summary(bets)           Combined dashboard dict
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SETTLED: frozenset[str] = frozenset({"won", "lost", "push"})


def _settled(bets: pd.DataFrame) -> pd.DataFrame:
    """Return only settled bets."""
    if bets.empty:
        return bets
    return bets.loc[bets["status"].isin(_SETTLED), :].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def record(
    bets: pd.DataFrame,
    *,
    split_by: str | None = None,
) -> pd.DataFrame:
    """Count wins, losses, and pushes.

    Args:
        bets: Bet ledger DataFrame.
        split_by: Optional column name to group by (e.g. ``"market_type"``).

    Returns:
        DataFrame with columns ``wins``, ``losses``, ``pushes``, ``total``,
        ``win_pct``.  One row per group (or one row if *split_by* is None).
        ``win_pct`` is ``wins / (wins + losses)`` — pushes are excluded from
        the denominator.
    """
    settled: DataFrame = _settled(bets)
    if settled.empty:
        cols: list[str] = ["wins", "losses", "pushes", "total", "win_pct"]
        if split_by:
            cols = [split_by, *cols]
        return pd.DataFrame(columns=cols)

    def _agg(df: pd.DataFrame) -> dict[str, Any]:
        wins: int = (df["status"] == "won").sum()
        losses: int = (df["status"] == "lost").sum()
        pushes: int = (df["status"] == "push").sum()
        denom: int = wins + losses
        return {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total": wins + losses + pushes,
            "win_pct": wins / denom if denom > 0 else float("nan"),
        }

    if split_by is None:
        return pd.DataFrame([_agg(settled)])

    rows: list[dict[str, Any]] = []
    for name, group in settled.groupby(split_by):
        row: dict[str, Any] = _agg(group)
        row[split_by] = name
        rows.append(row)
    cols = [split_by, "wins", "losses", "pushes", "total", "win_pct"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


def roi(
    bets: pd.DataFrame,
    *,
    split_by: str | None = None,
) -> pd.DataFrame:
    """Compute return on investment.

    Args:
        bets: Bet ledger DataFrame.
        split_by: Optional column name to group by.

    Returns:
        DataFrame with columns ``total_staked``, ``total_pnl``, ``roi_pct``.
    """
    settled: DataFrame = _settled(bets)
    if settled.empty:
        cols: list[str] = ["total_staked", "total_pnl", "roi_pct"]
        if split_by:
            cols = [split_by, *cols]
        return pd.DataFrame(columns=cols)

    def _agg(df: pd.DataFrame) -> dict[str, Any]:
        staked = float(df["stake"].sum())
        pnl = float(df["pnl"].sum())
        return {
            "total_staked": staked,
            "total_pnl": pnl,
            "roi_pct": (pnl / staked * 100) if staked > 0 else float("nan"),
        }

    if split_by is None:
        return pd.DataFrame([_agg(settled)])

    rows: list[dict[str, Any]] = []
    for name, group in settled.groupby(split_by):
        row: dict[str, Any] = _agg(group)
        row[split_by] = name
        rows.append(row)
    cols = [split_by, "total_staked", "total_pnl", "roi_pct"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CLV summary
# ---------------------------------------------------------------------------


def clv_summary(bets: pd.DataFrame) -> dict[str, float]:
    """Summarise closing line value across settled bets.

    Returns:
        Dict with keys ``mean_clv``, ``median_clv``, ``pct_positive_clv``,
        ``n_bets``.  Values are ``NaN`` when no CLV data is available.
    """
    settled: DataFrame = _settled(bets)
    if settled.empty:
        return _empty_clv()

    clv_col: Series = settled["clv"].dropna()
    if clv_col.empty:
        return _empty_clv()

    return {
        "mean_clv": clv_col.mean(),
        "median_clv": clv_col.median(),
        "pct_positive_clv": (clv_col > 0).sum() / len(clv_col) * 100,
        "n_bets": len(clv_col),
    }


def _empty_clv() -> dict[str, float]:
    return {
        "mean_clv": float("nan"),
        "median_clv": float("nan"),
        "pct_positive_clv": float("nan"),
        "n_bets": 0,
    }


# ---------------------------------------------------------------------------
# EV analysis
# ---------------------------------------------------------------------------


def ev_analysis(bets: pd.DataFrame) -> dict[str, float]:
    """Compare model EV at bet time to actual performance.

    Returns:
        Dict with keys ``mean_ev_at_bet``, ``mean_actual_roi``,
        ``ev_vs_actual_gap``, ``n_model_bets``.
    """
    settled: DataFrame = _settled(bets)
    if settled.empty:
        return _empty_ev()

    # Only bets with model EV populated
    model_bets: DataFrame = settled.dropna(subset=["model_ev"])
    if model_bets.empty:
        return _empty_ev()

    mean_ev: float = model_bets["model_ev"].mean()
    staked: Series = model_bets["stake"].astype(float)
    pnl: Series = model_bets["pnl"].astype(float)
    actual_roi: float = float((pnl / staked).mean()) if (staked > 0).all() else float("nan")
    gap: float = actual_roi - mean_ev if not math.isnan(actual_roi) else float("nan")

    return {
        "mean_ev_at_bet": mean_ev,
        "mean_actual_roi": actual_roi,
        "ev_vs_actual_gap": gap,
        "n_model_bets": len(model_bets),
    }


def _empty_ev() -> dict[str, float]:
    return {
        "mean_ev_at_bet": float("nan"),
        "mean_actual_roi": float("nan"),
        "ev_vs_actual_gap": float("nan"),
        "n_model_bets": 0,
    }


# ---------------------------------------------------------------------------
# Calibration health
# ---------------------------------------------------------------------------

# When the absolute EV-vs-actual gap is at or below this magnitude, the
# model is considered well-calibrated. Half a percent of ROI is real
# signal; less is noise. See ``performance/H1`` from audit_2026_06_18.md.
_CALIBRATION_GAP_TOLERANCE: float = 0.005


def _calibration_health(
    *,
    gap: float,
    n_model_bets: int,
) -> str:
    """Classify EV-vs-actual-roi gap into a health signal.

    Args:
        gap: The signed gap (actual_roi - mean_ev_at_bet) from
            :func:`ev_analysis`.
        n_model_bets: Number of bets with model EV populated.

    Returns:
        One of ``"healthy"``, ``"degraded"``, or ``"unknown"``.

        - ``"unknown"``: no model bets, or the gap is NaN.
        - ``"degraded"``: the model claimed positive EV that did not
          materialize, by more than the tolerance.
        - ``"healthy"``: the gap is within tolerance (or positive, i.e.
          actual exceeded the claimed EV).
    """
    if n_model_bets == 0 or math.isnan(gap):
        return "unknown"
    if gap < -_CALIBRATION_GAP_TOLERANCE:
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# Streak analysis
# ---------------------------------------------------------------------------


def streak_analysis(bets: pd.DataFrame) -> dict[str, Any]:
    """Analyse win/loss streaks.

    Pushes break streaks.  Bets are sorted by ``placed_at``.

    Returns:
        Dict with keys ``current_streak``, ``current_streak_type``,
        ``longest_win_streak``, ``longest_loss_streak``.
    """
    settled: DataFrame = _settled(bets)
    if settled.empty:
        return _empty_streak()

    sorted_bets: DataFrame = settled.sort_values("placed_at").reset_index(drop=True)
    streak, longest_w, longest_l = _walk_streaks(sorted_bets["status"])

    if streak > 0:
        current_type = "W"
    elif streak < 0:
        current_type = "L"
    else:
        current_type = ""

    return {
        "current_streak": streak,
        "current_streak_type": current_type,
        "longest_win_streak": longest_w,
        "longest_loss_streak": longest_l,
    }


def _walk_streaks(statuses: pd.Series) -> tuple[int, int, int]:
    """Walk a series of statuses and return (current, longest_w, longest_l)."""
    longest_w = 0
    longest_l = 0
    streak = 0

    for status in statuses:
        if status == "won":
            streak: int = streak + 1 if streak > 0 else 1
        elif status == "lost":
            streak = streak - 1 if streak < 0 else -1
        else:
            streak = 0

        if streak > 0:
            longest_w: int = max(longest_w, streak)
        elif streak < 0:
            longest_l: int = max(longest_l, abs(streak))

    return streak, longest_w, longest_l


def _empty_streak() -> dict[str, Any]:
    return {
        "current_streak": 0,
        "current_streak_type": "",
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
    }


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------


def summary(bets: pd.DataFrame) -> dict[str, Any]:
    """Build a combined performance summary.

    Calls :func:`record`, :func:`roi`, :func:`clv_summary`,
    :func:`ev_analysis`, and :func:`streak_analysis` and merges the
    results into a single flat dictionary.
    """
    rec: DataFrame = record(bets)
    r: DataFrame = roi(bets)
    clv: dict[str, float] = clv_summary(bets)
    ev: dict[str, float] = ev_analysis(bets)
    streaks: dict[str, Any] = streak_analysis(bets)

    result: dict[str, Any] = {}

    # Record
    if not rec.empty:
        row: Series = rec.iloc[0]
        result["wins"] = int(row["wins"])
        result["losses"] = int(row["losses"])
        result["pushes"] = int(row["pushes"])
        result["total"] = int(row["total"])
        result["win_pct"] = float(row["win_pct"])
    else:
        result.update({"wins": 0, "losses": 0, "pushes": 0, "total": 0, "win_pct": float("nan")})

    # ROI
    if not r.empty:
        row = r.iloc[0]
        result["total_staked"] = float(row["total_staked"])
        result["total_pnl"] = float(row["total_pnl"])
        result["roi_pct"] = float(row["roi_pct"])
    else:
        result.update({"total_staked": 0.0, "total_pnl": 0.0, "roi_pct": float("nan")})

    # CLV
    result["mean_clv"] = clv["mean_clv"]
    result["pct_positive_clv"] = clv["pct_positive_clv"]
    result["n_clv_bets"] = clv["n_bets"]

    # EV
    result["mean_ev_at_bet"] = ev["mean_ev_at_bet"]
    result["n_model_bets"] = ev["n_model_bets"]
    result["ev_vs_actual_gap"] = ev["ev_vs_actual_gap"]
    result["calibration_health"] = _calibration_health(
        gap=ev["ev_vs_actual_gap"],
        n_model_bets=int(ev["n_model_bets"]),
    )

    # Streaks
    result["current_streak"] = streaks["current_streak"]
    result["current_streak_type"] = streaks["current_streak_type"]
    result["longest_win_streak"] = streaks["longest_win_streak"]
    result["longest_loss_streak"] = streaks["longest_loss_streak"]

    return result
