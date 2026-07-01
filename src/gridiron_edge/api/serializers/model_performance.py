# src/gridiron_edge/api/serializers/model_performance.py

"""Serializer for /model/performance.

Per D17, hand-written. Per D18, owns _meta.field_status construction
for fields that are null due to data limits.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.model_performance import (
    BettingPerformanceBlock,
    GroupedMetricRow,
    ModelPerformance,
    ModelPerformanceFilters,
    ModelQualityBlock,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def serialize_model_performance(
    df_eval: DataFrame,
    summary_df: DataFrame,
    model_bet_summary: dict,
    filters: dict,
) -> ModelPerformance:
    """Build the /model/performance response.

    Args:
        df_eval: Output of build_evaluation_df, filtered to the requested
            scope. Empty if no evaluated games.
        summary_df: Output of summarise(df_eval, group_by=filters['group_by']).
            One row per group value.
        model_bet_summary: The subset of performance.summary() fields
            scoped to model bets (n_model_bets, mean_ev_at_bet, etc.).
            Empty dict if no model bets exist.
        filters: The query parameters that were applied. Keys: season,
            model_name, model_type, group_by.

    Returns:
        Fully assembled ModelPerformance response.
    """
    from gridiron_edge.evaluation.metrics import (
        accuracy,
        brier_decomposition,
        brier_score,
        expected_calibration_error,
        log_loss,
        roc_auc,
    )

    filters_block = ModelPerformanceFilters(
        season=filters.get("season"),
        model_name=filters.get("model_name"),
        model_type=filters.get("model_type"),
        group_by=filters["group_by"],
    )

    meta = ResponseMeta()

    # ------------------------------------------------------------------
    # model_quality: compute against df_eval as a whole
    # ------------------------------------------------------------------
    if df_eval.empty:
        model_quality = ModelQualityBlock(n_games=0)
        # No data at all — mark every metric field as unavailable.
        for field in (
            "brier",
            "log_loss",
            "accuracy",
            "ece",
            "roc_auc",
            "brier_reliability",
            "brier_resolution",
            "brier_uncertainty",
        ):
            meta = meta.with_blocked(
                f"model_quality.{field}",
                *Unavailable.NO_EVALUATION_DATA,
            )
    else:
        p = df_eval["away_win_prob"]
        y = df_eval["away_team_won"]

        # Scalar metrics
        brier = brier_score(p, y)
        ll = log_loss(p, y)
        acc = accuracy(p, y)
        ece = expected_calibration_error(p, y)

        # roc_auc requires both classes present; guard against
        # single-class filters.
        try:
            auc = roc_auc(p, y) if y.nunique() > 1 else None
        except Exception:
            auc = None
        if auc is None:
            meta = meta.with_blocked(
                "model_quality.roc_auc",
                *Unavailable.SINGLE_CLASS_OUTCOME,
            )

        # Brier decomposition
        decomp = brier_decomposition(p, y)

        model_quality = ModelQualityBlock(
            n_games=len(df_eval),
            brier=_none_if_nan(brier),
            log_loss=_none_if_nan(ll),
            accuracy=_none_if_nan(acc),
            ece=_none_if_nan(ece),
            roc_auc=_none_if_nan(auc),
            brier_reliability=_none_if_nan(decomp.get("reliability")),
            brier_resolution=_none_if_nan(decomp.get("resolution")),
            brier_uncertainty=_none_if_nan(decomp.get("uncertainty")),
        )

    # ------------------------------------------------------------------
    # betting_performance: pull from the pre-computed model-bet summary
    # ------------------------------------------------------------------
    if not model_bet_summary or model_bet_summary.get("n_model_bets", 0) == 0:
        betting_perf = BettingPerformanceBlock(n_model_bets=0)
        for field in (
            "mean_ev_at_bet",
            "ev_vs_actual_gap",
            "mean_clv",
            "pct_positive_clv",
            "roi_pct",
            "calibration_health",
        ):
            meta = meta.with_blocked(
                f"betting_performance.{field}",
                *Unavailable.NO_MODEL_CONTEXT,
            )
    else:
        betting_perf = BettingPerformanceBlock(
            n_model_bets=model_bet_summary.get("n_model_bets"),
            mean_ev_at_bet=_none_if_nan(model_bet_summary.get("mean_ev_at_bet")),
            ev_vs_actual_gap=_none_if_nan(
                model_bet_summary.get("ev_vs_actual_gap"),
            ),
            mean_clv=_none_if_nan(model_bet_summary.get("mean_clv")),
            pct_positive_clv=_none_if_nan(
                model_bet_summary.get("pct_positive_clv"),
            ),
            roi_pct=_none_if_nan(model_bet_summary.get("roi_pct")),
            calibration_health=model_bet_summary.get("calibration_health"),
        )

    # ------------------------------------------------------------------
    # by_group: one row per group from summarise()
    # ------------------------------------------------------------------
    group_key = filters["group_by"]
    if summary_df.empty:
        by_group: list[GroupedMetricRow] = []
    else:
        by_group = [
            GroupedMetricRow(
                group_key=str(row[group_key]),
                n_games=int(row["n_games"]) if pd.notna(row.get("n_games")) else None,
                brier=_none_if_nan(row.get("brier")),
                accuracy=_none_if_nan(row.get("accuracy")),
            )
            for _, row in summary_df.iterrows()
        ]

    return ModelPerformance(
        filters=filters_block,
        model_quality=model_quality,
        betting_performance=betting_perf,
        by_group=by_group,
        response_meta=meta if meta.field_status else None,  # pyrefly: ignore[unexpected-keyword]
    )
