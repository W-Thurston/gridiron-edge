# src/gridiron_edge/market/recommendations.py
"""Edge report builder - join model predictions with market odds.

Takes enriched prediction DataFrames and long-format odds DataFrames,
pivots odds to a per-game wide format, computes edges across moneyline,
spread, and total markets, and produces a ranked edge report.

Unlike ``odds_math.py``, ``kelly.py``, and ``edge.py``, this module
**does** use pandas - it orchestrates the data joins that connect model
outputs to market prices.  Callers are responsible for loading the data
(via ``load_prediction_log`` or the model enrichment pipeline); this
module operates on DataFrames passed as arguments.

Public API:
    pivot_odds_to_wide          Long odds → one row per game
    join_predictions_to_odds    Inner-join predictions to wide odds
    compute_game_edges          Per-game edge list (ML, spread, total)
    build_edge_report           Full edge report DataFrame
    rank_edges                  Filter + sort by EV
"""

from __future__ import annotations

import logging
from logging import Logger

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.market.edge import (
    MoneylineEdge,
    SpreadEdge,
    TotalEdge,
    classify_edge_strength,
    moneyline_edge,
    spread_edge,
    total_edge,
)

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report column schema
# ---------------------------------------------------------------------------

_REPORT_COLUMNS: list[str] = [
    "game_id",
    "game_date",
    "season",
    "week",
    "away_team",
    "home_team",
    "model_key",
    "confidence_tier",
    "market_type",
    "side",
    "model_value",
    "market_value",
    "american_odds",
    "point_edge",
    "cover_prob",
    "ev",
    "edge_strength",
    "kelly_frac",
    "kelly_stake",
]


# ---------------------------------------------------------------------------
# Odds pivot
# ---------------------------------------------------------------------------


def pivot_odds_to_wide(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format odds into one row per game.

    The odds ledger stores one row per market/side/game pull.  This
    function pivots to a wide format suitable for edge calculations,
    with one row per ``game_id``.

    Parameters
    ----------
    odds_df : pd.DataFrame
        Long-format odds with columns ``game_id``, ``market``, ``side``,
        ``odds``, ``line``.

    Returns:
    -------
    pd.DataFrame
        Wide DataFrame with columns: ``game_id``, ``ml_home``,
        ``ml_away``, ``spread_line_home``, ``spread_odds_home``,
        ``spread_odds_away``, ``total_line``, ``over_odds``,
        ``under_odds``.
    """
    if odds_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "ml_home",
                "ml_away",
                "spread_line_home",
                "spread_odds_home",
                "spread_odds_away",
                "total_line",
                "over_odds",
                "under_odds",
            ]
        )

    # Build a mapping of (game_id, market, side) -> last odds/line values.
    # Using groupby().last() to handle duplicate fetches (most recent wins).
    keyed: DataFrame = (
        odds_df.sort_values("fetched_at", na_position="first")
        .groupby(["game_id", "market", "side"], sort=False)
        .last()
        .reset_index()
    )

    rows: list[dict] = []
    for gid, grp in keyed.groupby("game_id", sort=False):
        row: dict = {
            "game_id": gid,
            "ml_home": float("nan"),
            "ml_away": float("nan"),
            "spread_line_home": float("nan"),
            "spread_odds_home": float("nan"),
            "spread_odds_away": float("nan"),
            "total_line": float("nan"),
            "over_odds": float("nan"),
            "under_odds": float("nan"),
        }
        for _, r in grp.iterrows():
            mkt: str = r["market"]
            side: str = r["side"]
            if mkt == "moneyline" and side == "home":
                row["ml_home"] = r["odds"]
            elif mkt == "moneyline" and side == "away":
                row["ml_away"] = r["odds"]
            elif mkt == "spread" and side == "home":
                row["spread_line_home"] = r["line"]
                row["spread_odds_home"] = r["odds"]
            elif mkt == "spread" and side == "away":
                row["spread_odds_away"] = r["odds"]
            elif mkt == "total" and side == "over":
                row["total_line"] = r["line"]
                row["over_odds"] = r["odds"]
            elif mkt == "total" and side == "under":
                row["under_odds"] = r["odds"]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction ↔ odds join
# ---------------------------------------------------------------------------


def join_predictions_to_odds(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join enriched predictions with wide-format odds on ``game_id``.

    If *odds_df* is still in long format (no ``ml_home`` column), it is
    pivoted automatically via :func:`pivot_odds_to_wide`.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Enriched predictions (must include ``game_id``).
    odds_df : pd.DataFrame
        Odds DataFrame - either long-format or already wide.

    Returns:
    -------
    pd.DataFrame
        Merged DataFrame with both prediction and odds columns.
    """
    odds_wide: DataFrame = (
        pivot_odds_to_wide(odds_df) if "ml_home" not in odds_df.columns else odds_df
    )

    merged: DataFrame = predictions_df.merge(odds_wide, on="game_id", how="inner")

    n_preds: int = len(predictions_df)
    n_matched: int = len(merged)
    logger.info(
        "join_predictions_to_odds: %d/%d predictions matched to odds",
        n_matched,
        n_preds,
    )
    return merged


# ---------------------------------------------------------------------------
# Per-game edge computation
# ---------------------------------------------------------------------------


def compute_game_edges(
    row: pd.Series,
    *,
    margin_std: float,
    total_std: float,
) -> list[MoneylineEdge | SpreadEdge | TotalEdge]:
    """Compute edges for all available markets in a single game row.

    Parameters
    ----------
    row : pd.Series
        A single row from the joined predictions + odds DataFrame.
        Expected fields: ``home_win_prob``, ``model_spread``,
        ``model_total``, ``ml_home``, ``ml_away``, ``spread_line_home``,
        ``spread_odds_home``, ``spread_odds_away``, ``total_line``,
        ``over_odds``, ``under_odds``.
    margin_std : float
        Standard deviation of margin-of-victory residuals (from
        model calibration).
    total_std : float
        Standard deviation of total-score residuals.

    Returns:
    -------
    list[MoneylineEdge | SpreadEdge | TotalEdge]
        Non-None edge results across all markets.
    """
    edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = []

    # Moneyline. Prefer the explicit ``home_win_prob`` column when
    # present; otherwise derive it from ``away_win_prob`` so the
    # function is robust against archives that only carry one side.
    if _has(row, "ml_home") and _has(row, "ml_away"):
        home_prob: float | None = None
        if _has(row, "home_win_prob"):
            home_prob = float(row["home_win_prob"])
        elif _has(row, "away_win_prob"):
            home_prob = 1.0 - float(row["away_win_prob"])

        if home_prob is not None:
            ml: MoneylineEdge | None = moneyline_edge(
                home_prob,
                int(row["ml_home"]),
                int(row["ml_away"]),
            )
            if ml is not None:
                edges.append(ml)

    # Spread
    if (
        _has(row, "spread_line_home")
        and _has(row, "spread_odds_home")
        and _has(row, "spread_odds_away")
        and _has(row, "model_spread")
    ):
        sp: SpreadEdge | None = spread_edge(
            row["model_spread"],
            row["spread_line_home"],
            int(row["spread_odds_home"]),
            int(row["spread_odds_away"]),
            margin_std,
        )
        if sp is not None:
            edges.append(sp)

    # Total
    if (
        _has(row, "total_line")
        and _has(row, "over_odds")
        and _has(row, "under_odds")
        and _has(row, "model_total")
    ):
        tot: TotalEdge | None = total_edge(
            row["model_total"],
            row["total_line"],
            int(row["over_odds"]),
            int(row["under_odds"]),
            total_std,
        )
        if tot is not None:
            edges.append(tot)

    return edges


# ---------------------------------------------------------------------------
# Edge report builder
# ---------------------------------------------------------------------------


def build_edge_report(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    margin_std: float,
    total_std: float,
    bankroll: float | None = None,
    kelly_multiplier: float = 0.25,
) -> pd.DataFrame:
    """Build a ranked edge report across all games and markets.

    Joins predictions to odds, computes per-game edges for moneyline,
    spread, and total markets, and returns one row per game x market
    type.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Enriched predictions (from ``load_prediction_log`` or the model
        enrichment pipeline). Must contain ``model_name`` and
        ``model_type`` columns; ``model_key`` is derived in the output.
    odds_df : pd.DataFrame
        Odds data - long or wide format.
    margin_std : float
        Spread residual std for spread cover probability.
    total_std : float
        Total residual std for total cover probability.
    bankroll : float | None, default None
        Current bankroll in dollars. When None, edge calculations retain
        full-Kelly fractions but kelly_stake remains unavailable.
    kelly_multiplier : float, default 0.25
        Fraction of full Kelly to apply (e.g. 0.25 for quarter-Kelly).

    Returns:
    -------
    pd.DataFrame
        Edge report with columns matching :data:`_REPORT_COLUMNS`.
        Contains all edges including ``"no_edge"`` rows. The
        ``model_key`` column is the composite ``f"{model_name}_{model_type}"``
        derived from the prediction row.
    """
    if bankroll is not None and bankroll < 0:
        raise ValueError(f"bankroll must be >= 0, got {bankroll}")

    if not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError(f"kelly_multiplier must be in [0, 1], got {kelly_multiplier}")

    joined: DataFrame = join_predictions_to_odds(predictions_df, odds_df)

    if joined.empty:
        return pd.DataFrame(columns=_REPORT_COLUMNS)

    report_rows: list[dict] = []

    for _, row in joined.iterrows():
        model_name: str = str(row.get("model_name", ""))
        model_type: str = str(row.get("model_type", ""))
        model_key: str = f"{model_name}_{model_type}"

        game_base: dict = {
            "game_id": row.get("game_id", ""),
            "game_date": row.get("game_date", ""),
            "season": row.get("season", ""),
            "week": row.get("week", ""),
            "away_team": row.get("away_team", ""),
            "home_team": row.get("home_team", ""),
            "model_key": model_key,
            "confidence_tier": row.get("confidence_tier", ""),
        }

        edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = compute_game_edges(
            row,
            margin_std=margin_std,
            total_std=total_std,
        )

        for edge in edges:
            edge_row: dict = {**game_base}

            if isinstance(edge, MoneylineEdge):
                edge_row["market_type"] = "moneyline"
                edge_row["side"] = edge.side
                edge_row["model_value"] = edge.model_prob
                edge_row["market_value"] = edge.market_prob
                edge_row["point_edge"] = float("nan")
                edge_row["cover_prob"] = float("nan")
            elif isinstance(edge, SpreadEdge):
                edge_row["market_type"] = "spread"
                edge_row["side"] = edge.side
                edge_row["model_value"] = edge.model_spread
                edge_row["market_value"] = edge.market_spread
                edge_row["point_edge"] = edge.point_edge
                edge_row["cover_prob"] = edge.cover_prob
            elif isinstance(edge, TotalEdge):
                edge_row["market_type"] = "total"
                edge_row["side"] = edge.side
                edge_row["model_value"] = edge.model_total
                edge_row["market_value"] = edge.market_total
                edge_row["point_edge"] = edge.point_edge
                edge_row["cover_prob"] = edge.cover_prob

            edge_row["american_odds"] = edge.odds
            edge_row["ev"] = edge.ev
            edge_row["edge_strength"] = classify_edge_strength(edge.ev)
            edge_row["kelly_frac"] = edge.kelly_frac

            if bankroll is None:
                edge_row["kelly_stake"] = None
            else:
                max_stake: float = bankroll * kelly_multiplier
                raw_stake: float = bankroll * kelly_multiplier * edge.kelly_frac
                edge_row["kelly_stake"] = min(raw_stake, max_stake)

            report_rows.append(edge_row)

    if not report_rows:
        return pd.DataFrame(columns=_REPORT_COLUMNS)

    return pd.DataFrame(report_rows, columns=_REPORT_COLUMNS)


# ---------------------------------------------------------------------------
# Edge ranking / filtering
# ---------------------------------------------------------------------------


def rank_edges(
    report_df: pd.DataFrame,
    *,
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """Filter and rank edges by expected value.

    Parameters
    ----------
    report_df : pd.DataFrame
        Output of :func:`build_edge_report`.
    min_ev : float, default 0.0
        Minimum EV threshold.  Rows with ``ev <= min_ev`` are excluded.

    Returns:
    -------
    pd.DataFrame
        Filtered and sorted copy, highest EV first.
    """
    filtered: DataFrame = report_df.loc[report_df["ev"] > min_ev, :].copy()
    # pyrefly: ignore [no-matching-overload]
    return filtered.sort_values("ev", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has(row: pd.Series, col: str) -> bool:
    """Return True if *col* exists in *row* and is not NaN."""
    if col not in row.index:
        return False
    val = row[col]
    try:
        return not np.isnan(val)
    except (TypeError, ValueError):
        # Non-numeric values (strings, etc.) are "present"
        return val is not None
