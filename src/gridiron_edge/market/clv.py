# src/gridiron_edge/market/clv.py
"""Closing Line Value (CLV) analysis.

CLV measures whether the market moved toward or away from your position
between bet placement and game close.  Consistently positive CLV is the
gold-standard validation that a model is finding real edges — not just
getting lucky.

Pure scalar helpers (no I/O):
    closing_line_value     Generic probability-based CLV
    spread_clv             Point-based CLV for spread bets
    total_clv              Point-based CLV for total bets

Ledger extraction (pandas):
    extract_opening_odds   First pull per (game_id, market, side)
    extract_closing_odds   Last pull per (game_id, market, side)

Reporting (pandas):
    build_clv_report       Augment an edge report with CLV columns
    summarize_clv          Aggregate CLV stats
"""

from __future__ import annotations

import logging
from logging import Logger

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.market.odds_math import no_vig
from gridiron_edge.market.recommendations import pivot_odds_to_wide

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure scalar CLV helpers
# ---------------------------------------------------------------------------


def closing_line_value(bet_fair_prob: float, close_fair_prob: float) -> float:
    """Return probability-based CLV for a moneyline wager.

    CLV = (close_fair_prob - bet_fair_prob) / bet_fair_prob

    A positive value means the market moved **toward** your position
    (the closing line was shorter than the line you bet), indicating
    genuine edge.

    Parameters
    ----------
    bet_fair_prob : float
        No-vig fair probability at the time of the bet.  Must be in (0, 1).
    close_fair_prob : float
        No-vig fair probability at close.  Must be in (0, 1).

    Returns:
    -------
    float
        CLV as a decimal (e.g. 0.05 = +5% CLV).

    Raises:
    ------
    ValueError
        If either probability is not in (0, 1).
    """
    _validate_prob(bet_fair_prob, "bet_fair_prob")
    _validate_prob(close_fair_prob, "close_fair_prob")
    return (close_fair_prob - bet_fair_prob) / bet_fair_prob


def spread_clv(
    bet_spread: float,
    close_spread: float,
    side: str,
) -> float:
    """Return point-based CLV for a spread wager.

    Measures how many points of value you gained (or lost) from spread
    movement between bet placement and close.

    Convention: spread is negative when the home team is favored.

    - ``side="home"``: you bet home to cover.  If the line moves from
      -3 to -7 (home got stronger), you locked in a better number.
      ``CLV = bet_spread - close_spread = -3 - (-7) = +4``.
    - ``side="away"``: mirror of home.
      ``CLV = close_spread - bet_spread``.

    Parameters
    ----------
    bet_spread : float
        Spread at time of bet (negative = home favored).
    close_spread : float
        Closing spread (negative = home favored).
    side : str
        ``"home"`` or ``"away"``.

    Returns:
    -------
    float
        Points of value (positive = you got a better number).

    Raises:
    ------
    ValueError
        If *side* is not ``"home"`` or ``"away"``.
    """
    if side == "home":
        return bet_spread - close_spread
    if side == "away":
        return close_spread - bet_spread
    raise ValueError(f"side must be 'home' or 'away', got '{side}'")


def total_clv(
    bet_total: float,
    close_total: float,
    side: str,
) -> float:
    """Return point-based CLV for a total (over/under) wager.

    Measures how many points the total moved in your favor.

    - ``side="over"``: you bet over.  If the line moves from 42 to 45,
      the market agrees more with the over -> ``CLV = close - bet = +3``.
    - ``side="under"``: mirror of over.
      ``CLV = bet - close``.

    Parameters
    ----------
    bet_total : float
        Total at time of bet.
    close_total : float
        Closing total.
    side : str
        ``"over"`` or ``"under"``.

    Returns:
    -------
    float
        Points of value (positive = market moved your way).

    Raises:
    ------
    ValueError
        If *side* is not ``"over"`` or ``"under"``.
    """
    if side == "over":
        return close_total - bet_total
    if side == "under":
        return bet_total - close_total
    raise ValueError(f"side must be 'over' or 'under', got '{side}'")


# ---------------------------------------------------------------------------
# Ledger extraction
# ---------------------------------------------------------------------------


def extract_opening_odds(
    odds_ledger: pd.DataFrame,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Extract the earliest pull per (game_id, market, side).

    Parameters
    ----------
    odds_ledger : pd.DataFrame
        Full odds ledger in long format (must include ``fetched_at``).
    game_ids : list[str] | None
        If provided, filter to these game_ids first.

    Returns:
    -------
    pd.DataFrame
        Long-format DataFrame with only the opening rows.
    """
    if odds_ledger.empty:
        return odds_ledger.copy()
    df: DataFrame = odds_ledger.copy()
    if game_ids is not None:
        df = df.loc[df["game_id"].isin(game_ids), :]
    return (
        df.sort_values("fetched_at", ascending=True, na_position="last")
        .groupby(["game_id", "market", "side"], sort=False)
        .first()
        .reset_index()
    )


def extract_closing_odds(
    odds_ledger: pd.DataFrame,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Extract the latest pull per (game_id, market, side).

    Parameters
    ----------
    odds_ledger : pd.DataFrame
        Full odds ledger in long format (must include ``fetched_at``).
    game_ids : list[str] | None
        If provided, filter to these game_ids first.

    Returns:
    -------
    pd.DataFrame
        Long-format DataFrame with only the closing rows.
    """
    if odds_ledger.empty:
        return odds_ledger.copy()
    df: DataFrame = odds_ledger.copy()
    if game_ids is not None:
        df = df.loc[df["game_id"].isin(game_ids), :]
    return (
        df.sort_values("fetched_at", ascending=True, na_position="first")
        .groupby(["game_id", "market", "side"], sort=False)
        .last()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# CLV report builder
# ---------------------------------------------------------------------------


def build_clv_report(
    edge_report_df: pd.DataFrame,
    odds_ledger_df: pd.DataFrame,
) -> pd.DataFrame:
    """Augment an edge report with CLV columns.

    For each row in the edge report, finds the opening and closing odds
    from the ledger and computes the appropriate CLV metric. Vectorized
    per market_type (spread and total are fully vectorized; moneyline
    uses a small per-row de-vigging step inside its subset). See
    ``clv/M1``.

    Parameters
    ----------
    edge_report_df : pd.DataFrame
        Output of :func:`~gridiron_edge.market.recommendations.build_edge_report`.
        Must include ``game_id``, ``market_type``, ``side``.
    odds_ledger_df : pd.DataFrame
        Full odds ledger in long format.

    Returns:
    -------
    pd.DataFrame
        Copy of *edge_report_df* with added columns: ``opening_value``,
        ``closing_value``, ``clv``.
    """
    if edge_report_df.empty:
        result: DataFrame = edge_report_df.copy()
        result["opening_value"] = pd.Series(dtype=float)
        result["closing_value"] = pd.Series(dtype=float)
        result["clv"] = pd.Series(dtype=float)
        return result

    game_ids = edge_report_df["game_id"].unique().tolist()

    opening_long: DataFrame = extract_opening_odds(odds_ledger_df, game_ids=game_ids)
    closing_long: DataFrame = extract_closing_odds(odds_ledger_df, game_ids=game_ids)

    opening_wide: DataFrame = _pivot_and_suffix(opening_long, "open")
    closing_wide: DataFrame = _pivot_and_suffix(closing_long, "close")

    merged = edge_report_df.copy()
    merged = merged.merge(opening_wide, on="game_id", how="left")
    merged = merged.merge(closing_wide, on="game_id", how="left")

    # Dispatch by market_type. Each handler returns a DataFrame with
    # opening_value, closing_value, clv columns appended. Rows whose
    # market_type is unrecognized fall through to the NaN bucket.
    handlers = {
        "moneyline": _vectorized_ml_clv,
        "spread": _vectorized_spread_clv,
        "total": _vectorized_total_clv,
    }

    parts: list[pd.DataFrame] = []
    known_mask = pd.Series(False, index=merged.index)
    for market_type, handler in handlers.items():
        mask = merged["market_type"] == market_type
        known_mask = known_mask | mask
        if mask.any():
            parts.append(handler(merged.loc[mask, :]))

    # Unknown market_type rows (or rows where market_type is NaN) get
    # NaN CLV columns to match the original fallback semantics.
    unknown = merged.loc[~known_mask, :].copy()
    if not unknown.empty:
        unknown["opening_value"] = float("nan")
        unknown["closing_value"] = float("nan")
        unknown["clv"] = float("nan")
        parts.append(unknown)

    result = (
        pd.concat(parts, axis=0).sort_index()
        if parts
        else merged.assign(
            opening_value=float("nan"),
            closing_value=float("nan"),
            clv=float("nan"),
        )
    )

    # Drop the intermediate wide columns to keep output clean.
    drop_cols: list[str] = [
        c for c in result.columns if c.endswith("_open") or c.endswith("_close")
    ]
    result = result.drop(columns=drop_cols, errors="ignore")

    return result


def summarize_clv(clv_report_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate CLV statistics from a CLV report.

    Parameters
    ----------
    clv_report_df : pd.DataFrame
        Output of :func:`build_clv_report`.

    Returns:
    -------
    dict[str, float]
        Keys: ``mean_clv``, ``median_clv``, ``pct_positive_clv``,
        ``n_edges``.  NaN for empty reports.
    """
    clv_col: Series | None = clv_report_df.get("clv")
    if clv_col is None or clv_col.dropna().empty:
        return {
            "mean_clv": float("nan"),
            "median_clv": float("nan"),
            "pct_positive_clv": float("nan"),
            "n_edges": 0.0,
        }

    valid: Series = clv_col.dropna()
    return {
        "mean_clv": valid.mean(),
        "median_clv": valid.median(),
        "pct_positive_clv": (valid > 0).mean(),
        "n_edges": float(len(valid)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_prob(p: float, name: str = "probability") -> None:
    """Raise ``ValueError`` if *p* is not strictly between 0 and 1."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {p}")


def _pivot_and_suffix(long_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Pivot long odds to wide and add a column suffix."""
    wide: DataFrame = pivot_odds_to_wide(long_df)
    rename_map: dict[str, str] = {c: f"{c}_{suffix}" for c in wide.columns if c != "game_id"}
    return wide.rename(columns=rename_map)


def _compute_row_clv(row: pd.Series) -> tuple[float, float, float]:
    """Compute CLV for a single edge report row.

    Returns (opening_value, closing_value, clv).
    """
    mkt: str = row.get("market_type", "")
    side: str = row.get("side", "")
    nan = float("nan")

    try:
        if mkt == "moneyline":
            return _ml_clv(row, side)
        if mkt == "spread":
            return _spread_row_clv(row, side)
        if mkt == "total":
            return _total_row_clv(row, side)
    except (ValueError, TypeError, KeyError):
        pass

    return nan, nan, nan


def _ml_clv(row: pd.Series, side: str) -> tuple[float, float, float]:
    """Moneyline CLV using no-vig probabilities."""
    nan = float("nan")

    ml_home_open = row.get("ml_home_open", nan)
    ml_away_open = row.get("ml_away_open", nan)
    ml_home_close = row.get("ml_home_close", nan)
    ml_away_close = row.get("ml_away_close", nan)

    if any(_isnan(v) for v in [ml_home_open, ml_away_open, ml_home_close, ml_away_close]):
        return nan, nan, nan

    open_home, open_away = no_vig(int(ml_home_open), int(ml_away_open))
    close_home, close_away = no_vig(int(ml_home_close), int(ml_away_close))

    if side == "home":
        open_prob, close_prob = open_home, close_home
    else:
        open_prob, close_prob = open_away, close_away

    clv_val: float = closing_line_value(open_prob, close_prob)
    return open_prob, close_prob, clv_val


def _spread_row_clv(row: pd.Series, side: str) -> tuple[float, float, float]:
    """Spread CLV using point movement."""
    nan = float("nan")

    spread_open = row.get("spread_line_home_open", nan)
    spread_close = row.get("spread_line_home_close", nan)

    if _isnan(spread_open) or _isnan(spread_close):
        return nan, nan, nan

    clv_val: float = spread_clv(spread_open, spread_close, side)
    return float(spread_open), float(spread_close), clv_val


def _total_row_clv(row: pd.Series, side: str) -> tuple[float, float, float]:
    """Total CLV using point movement."""
    nan = float("nan")

    total_open = row.get("total_line_open", nan)
    total_close = row.get("total_line_close", nan)

    if _isnan(total_open) or _isnan(total_close):
        return nan, nan, nan

    clv_val: float = total_clv(total_open, total_close, side)
    return float(total_open), float(total_close), clv_val


def _isnan(val: object) -> bool:
    """Return True if *val* is NaN or None."""
    if val is None:
        return True
    try:
        # pyrefly: ignore [no-matching-overload]
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Vectorized CLV computation (clv/M1)
# ---------------------------------------------------------------------------
#
# These helpers process whole DataFrames per market_type in one pass,
# replacing the row-by-row iterrows() path used by the original
# build_clv_report. The per-market scalar helpers above remain the
# public scalar API; these are internal to the report builder.


def _vectorized_ml_clv(group: pd.DataFrame) -> pd.DataFrame:
    """Compute moneyline CLV for a group of edge-report rows.

    Vectorized counterpart to ``_ml_clv``. Returns the input DataFrame
    with ``opening_value``, ``closing_value``, and ``clv`` columns
    appended. NaN inputs propagate to NaN outputs.

    Notes:
    -----
    The ``no_vig`` helper is per-row scalar (takes two ints, returns a
    pair). We call it via a row-wise apply over the subset because the
    moneyline conversion involves a non-linear de-vigging step that is
    not natively vectorized in ``market.odds_math``. The hot path for
    CLV reports is spread + total (point-based), where the entire
    computation is vectorized below. Moneyline rows are typically a
    small minority and the apply over a subset is acceptable.
    """
    out = group.copy()
    nan = float("nan")

    required = ["ml_home_open", "ml_away_open", "ml_home_close", "ml_away_close"]
    has_all = (
        # pyrefly: ignore [bad-argument-type]
        out[required].notna().all(axis=1)
        if all(c in out.columns for c in required)
        else pd.Series(False, index=out.index)
    )

    if not has_all.any():
        out["opening_value"] = nan
        out["closing_value"] = nan
        out["clv"] = nan
        return out

    valid = out.loc[has_all, :]

    # The no_vig() helper is scalar-pair-in, pair-out. Apply per row
    # within this subset; outside callers see only the aggregated result.
    open_pairs = valid.apply(
        lambda r: no_vig(int(r["ml_home_open"]), int(r["ml_away_open"])),
        axis=1,
    )
    close_pairs = valid.apply(
        lambda r: no_vig(int(r["ml_home_close"]), int(r["ml_away_close"])),
        axis=1,
    )

    open_home = open_pairs.map(lambda p: p[0])
    open_away = open_pairs.map(lambda p: p[1])
    close_home = close_pairs.map(lambda p: p[0])
    close_away = close_pairs.map(lambda p: p[1])

    is_home = valid["side"].astype(str) == "home"
    open_prob = open_home.where(is_home, open_away)
    close_prob = close_home.where(is_home, close_away)

    # closing_line_value scalar formula, vectorized; guard against zero
    # denominators.
    clv_vals = (close_prob - open_prob) / open_prob.replace(0.0, float("nan"))

    out["opening_value"] = nan
    out["closing_value"] = nan
    out["clv"] = nan
    out.loc[has_all, "opening_value"] = open_prob.values
    out.loc[has_all, "closing_value"] = close_prob.values
    out.loc[has_all, "clv"] = clv_vals.values

    return out


def _vectorized_spread_clv(group: pd.DataFrame) -> pd.DataFrame:
    """Compute spread CLV for a group of edge-report rows.

    Vectorized counterpart to ``_spread_row_clv``. NaN inputs propagate.
    """
    out = group.copy()
    nan = float("nan")

    open_col = "spread_line_home_open"
    close_col = "spread_line_home_close"
    if open_col not in out.columns or close_col not in out.columns:
        out["opening_value"] = nan
        out["closing_value"] = nan
        out["clv"] = nan
        return out

    # pyrefly: ignore [bad-assignment]
    spread_open: Series = pd.to_numeric(out[open_col], errors="coerce")
    # pyrefly: ignore [bad-assignment]
    spread_close: Series = pd.to_numeric(out[close_col], errors="coerce")
    # pyrefly: ignore [bad-assignment]
    side: Series[str] = out["side"].astype(str)

    # Vectorized form of:
    #   side == "home" -> bet_spread - close_spread
    #   else            -> close_spread - bet_spread
    home_clv: Series = spread_open - spread_close
    away_clv: Series = spread_close - spread_open
    clv_vals: Series = home_clv.where(side == "home", away_clv)

    has_both: Series[bool] = spread_open.notna() & spread_close.notna()
    out["opening_value"] = spread_open.where(has_both, nan)
    out["closing_value"] = spread_close.where(has_both, nan)
    out["clv"] = clv_vals.where(has_both, nan)

    return out


def _vectorized_total_clv(group: pd.DataFrame) -> pd.DataFrame:
    """Compute total CLV for a group of edge-report rows.

    Vectorized counterpart to ``_total_row_clv``. NaN inputs propagate.
    """
    out = group.copy()
    nan = float("nan")

    open_col = "total_line_open"
    close_col = "total_line_close"
    if open_col not in out.columns or close_col not in out.columns:
        out["opening_value"] = nan
        out["closing_value"] = nan
        out["clv"] = nan
        return out

    # pyrefly: ignore [bad-assignment]
    total_open: Series = pd.to_numeric(out[open_col], errors="coerce")
    # pyrefly: ignore [bad-assignment]
    total_close: Series = pd.to_numeric(out[close_col], errors="coerce")
    side: Series = out["side"].astype(str)

    # Vectorized form of:
    #   side == "over"  -> close - bet
    #   side == "under" -> bet - close
    over_clv: Series = total_close - total_open
    under_clv: Series = total_open - total_close
    clv_vals: Series = over_clv.where(side == "over", under_clv)

    has_both: Series[bool] = total_open.notna() & total_close.notna()
    out["opening_value"] = total_open.where(has_both, nan)
    out["closing_value"] = total_close.where(has_both, nan)
    out["clv"] = clv_vals.where(has_both, nan)

    return out
