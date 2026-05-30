# src/gridiron_edge/evaluation/select.py

"""Model selection and ranking utilities.

Provides the domain logic for comparing registered prediction models
by evaluation metrics and producing ranked results. These functions are
intentionally CLI-agnostic so they can be called from tests or notebooks
without importing the CLI layer.

Public API
----------
collect_model_metrics   Compute evaluation metrics for all models with archived data.
rank_models             Rank a list of metric dicts by composite criteria.
compute_report_data     Load predictions and compute all four report DataFrames.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame, Series


def collect_model_metrics(
    model_names: list[str],
    *,
    repo: Path,
) -> list[dict]:
    """Compute evaluation metrics for all models with archived predictions.

    Args:
        model_names: List of registered model version strings.
        repo: Repository root.

    Returns:
        List of metric dicts, one per model that has archived predictions.
        Models with no archived data are silently skipped.
    """
    from gridiron_edge.evaluation.metrics import (
        accuracy,
        brier_score,
        build_evaluation_df,
        expected_calibration_error,
        log_loss,
        roc_auc,
    )

    rows: list[dict[str, float | int | str]] = []
    for mv in model_names:
        df_eval: DataFrame = build_evaluation_df(model_version=mv, repo=repo)
        if df_eval.empty:
            continue
        p: Series = df_eval["away_win_prob"]
        y: Series = df_eval["away_team_won"]
        rows.append(
            {
                "model_version": mv,
                "n_games": len(df_eval),
                "brier": round(brier_score(p, y), 5),
                "ece": round(expected_calibration_error(p, y), 5),
                "auc": round(roc_auc(p, y), 5),
                "accuracy": round(accuracy(p, y), 5),
                "log_loss": round(log_loss(p, y), 5),
            }
        )
    return rows


def rank_models(
    rows: list[dict],
    *,
    criteria_list: list[str],
    lower_is_better: set[str],
) -> DataFrame:
    """Rank a list of model metric dicts and return a sorted DataFrame.

    Args:
        rows: List of dicts from ``collect_model_metrics``.
        criteria_list: Ordered list of metric names to rank on.
        lower_is_better: Set of criteria where lower values are better.

    Returns:
        Ranked DataFrame sorted by composite_rank ascending, then primary
        criterion. Includes a ``composite_rank`` column and one
        ``rank_{criterion}`` column per criterion.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    for criterion in criteria_list:
        rank_col: str = f"rank_{criterion}"
        ascending: bool = criterion in lower_is_better
        df[rank_col] = df[criterion].rank(ascending=ascending, method="min").astype(int)

    # pyrefly: ignore [bad-argument-type]
    rank_cols: list[str] = [f"rank_{c}" for c in criteria_list]
    # pyrefly: ignore [bad-argument-type]
    df["composite_rank"] = df[rank_cols].sum(axis=1)
    df = df.sort_values(
        ["composite_rank", criteria_list[0]],
        ascending=[True, criteria_list[0] in lower_is_better],
    ).reset_index(drop=True)
    return df


def compute_report_data(
    *,
    target_mv: str,
    season: str | None,
    top_misses: int,
    repo: Path,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load predictions and compute all four report DataFrames.

    Args:
        target_mv: Model version to analyse.
        season: Optional season filter.
        top_misses: Number of worst predictions to surface.
        repo: Repository root.

    Returns:
        Tuple of (df_eval, df_tiers, df_seasons, df_misses).

    Raises:
        ValueError: If no completed games are found for the model.
    """
    from gridiron_edge.evaluation.metrics import (
        biggest_misses,
        brier_by_confidence_tier,
        brier_by_season,
        build_evaluation_df,
    )

    df_eval: DataFrame = build_evaluation_df(model_version=target_mv, season=season, repo=repo)
    if df_eval.empty:
        raise ValueError(f"No completed games found for '{target_mv}'.")

    df_tiers: DataFrame = brier_by_confidence_tier(df_eval)
    df_seasons: DataFrame = brier_by_season(df_eval)
    df_misses: DataFrame = biggest_misses(df_eval, n=top_misses)

    return df_eval, df_tiers, df_seasons, df_misses
