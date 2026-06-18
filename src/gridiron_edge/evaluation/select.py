# src/gridiron_edge/evaluation/select.py

"""Model selection and ranking utilities.

Provides the domain logic for comparing registered prediction models
by evaluation metrics and producing ranked results. These functions are
intentionally CLI-agnostic so they can be called from tests or notebooks
without importing the CLI layer.

Workstream 2 convention:
    All registered ``PredictorRegistry`` keys are composite strings
    of the form ``f"{model_name}_{model_type}"`` (e.g.
    ``"win_prob_random_forest"``, ``"total_xgboost"``, ``"win_prob_elo"``).
    Functions in this module split each key on the first underscore
    into ``(model_name, model_type)`` before querying the archive. The
    output ``model_key`` column carries the registry key as a display
    label.

Public API
----------
collect_model_metrics   Compute evaluation metrics for all models with archived data.
rank_models             Rank a list of metric dicts by composite criteria.
compute_report_data     Load predictions and compute all four report DataFrames.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame, Series


def _parse_composite_key(key: str) -> tuple[str, str]:
    """Split a PredictorRegistry composite key into (model_name, model_type).

    All registered keys follow ``f"{model_name}_{model_type}"`` where
    ``model_name`` is a single token without underscores. Splits on the
    first underscore.

    Args:
        key: Composite registry key, e.g. ``"win_prob_random_forest"``.

    Returns:
        Tuple of ``(model_name, model_type)``.

    Raises:
        ValueError: If the key does not contain an underscore.
            All registry entries must be composite keys; a non-composite
            key indicates a registration bug.
    """
    if "_" not in key:
        msg: str = (
            f"Composite key {key!r} does not contain an underscore. "
            f"Expected format: '{{model_name}}_{{model_type}}'."
        )
        raise ValueError(msg)
    name, _, mtype = key.partition("_")
    return name, mtype


def collect_model_metrics(
    model_keys: list[str],
    *,
    repo: Path,
) -> list[dict]:
    """Compute evaluation metrics for all models with archived predictions.

    Args:
        model_keys: List of PredictorRegistry composite keys
            (e.g. ``"win_prob_random_forest"``).
        repo: Repository root.

    Returns:
        List of metric dicts, one per model that has archived predictions.
        Models with no archived data are silently skipped. Each dict
        carries a ``"model_key"`` field holding the registry key as a
        display label.
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
    for key in model_keys:
        model_name, model_type = _parse_composite_key(key)

        df_eval: DataFrame = build_evaluation_df(
            model_name=model_name,
            model_type=model_type,
            repo=repo,
        )
        if df_eval.empty:
            continue

        p: Series = df_eval["away_win_prob"]
        y: Series = df_eval["away_team_won"]
        rows.append(
            {
                "model_key": key,
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
        ``rank_{criterion}`` column per criterion. The model identity
        column is ``"model_key"``.
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
    target_key: str,
    season: str | None,
    top_misses: int,
    repo: Path,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load predictions and compute all four report DataFrames.

    Args:
        target_key: Composite PredictorRegistry key of the model to analyse
            (e.g. ``"win_prob_random_forest"``).
        season: Optional season filter.
        top_misses: Number of worst predictions to surface.
        repo: Repository root.

    Returns:
        Tuple of (df_eval, df_tiers, df_seasons, df_misses).

    Raises:
        ValueError: If ``target_key`` is not a valid composite key, or if
            no completed games are found for the model.
    """
    from gridiron_edge.evaluation.metrics import (
        biggest_misses,
        brier_by_confidence_tier,
        brier_by_season,
        build_evaluation_df,
    )

    model_name, model_type = _parse_composite_key(target_key)

    df_eval: DataFrame = build_evaluation_df(
        model_name=model_name,
        model_type=model_type,
        season=season,
        repo=repo,
    )
    if df_eval.empty:
        raise ValueError(f"No completed games found for {target_key!r}.")

    df_tiers: DataFrame = brier_by_confidence_tier(df_eval)
    df_seasons: DataFrame = brier_by_season(df_eval)
    df_misses: DataFrame = biggest_misses(df_eval, n=top_misses)

    return df_eval, df_tiers, df_seasons, df_misses
