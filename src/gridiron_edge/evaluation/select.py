# src/gridiron_edge/evaluation/select.py

"""Model selection and ranking utilities.

Provides the domain logic for comparing registered prediction models
by evaluation metrics and producing ranked results. These functions are
intentionally CLI-agnostic so they can be called from tests or notebooks
without importing the CLI layer.

Composite model identity:
    All registered ``ModelRegistry`` keys are composite strings
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
    """Split a ModelRegistry composite key into (model_name, model_type).

    All registered keys follow ``f"{model_name}_{model_type}"``. Because both
    ``model_name`` (e.g. ``"win_prob"``) and ``model_type``
    (e.g. ``"random_forest"``) can contain underscores, this function matches
    the key against the known model_name prefixes returned by
    :func:`gridiron_edge.models.game_prediction.model.get_known_model_names`
    rather than splitting on a single underscore.
    """
    from gridiron_edge.models.game_prediction.model import get_known_model_names

    known_names: tuple[str, ...] = get_known_model_names()
    for model_name in known_names:
        prefix: str = f"{model_name}_"
        if key.startswith(prefix):
            model_type: str = key[len(prefix) :]
            return model_name, model_type
    msg: str = (
        f"Composite key {key!r} does not match any known model_name prefix. "
        f"Known prefixes: {sorted(known_names)}."
    )
    raise ValueError(msg)


def _classification_metric_row(
    evaluation: DataFrame,
    *,
    model_key: str,
) -> dict[str, float | int | str] | None:
    """Compute one classification metric row from evaluation data."""
    from gridiron_edge.evaluation.metrics import (
        accuracy,
        brier_score,
        expected_calibration_error,
        log_loss,
        roc_auc,
    )

    if evaluation.empty or "away_win_prob" not in evaluation.columns:
        return None
    if evaluation["away_win_prob"].isna().all():
        return None
    binary = evaluation.loc[evaluation["away_team_won"].isin([0, 1]), :].copy()
    if binary.empty:
        return None
    p: Series = binary["away_win_prob"]
    y: Series = binary["away_team_won"]
    return {
        "model_key": model_key,
        "n_games": len(binary),
        "brier": round(brier_score(p, y), 5),
        "ece": round(expected_calibration_error(p, y), 5),
        "auc": round(roc_auc(p, y), 5),
        "accuracy": round(accuracy(p, y), 5),
        "log_loss": round(log_loss(p, y), 5),
    }


def collect_model_metrics(
    model_keys: list[str],
    *,
    repo: Path,
) -> list[dict[str, float | int | str]]:
    """Compute metrics from the legacy overwriteable prediction archive."""
    from gridiron_edge.evaluation.metrics import build_evaluation_df

    rows: list[dict[str, float | int | str]] = []
    for key in model_keys:
        model_name, model_type = _parse_composite_key(key)
        row = _classification_metric_row(
            build_evaluation_df(
                model_name=model_name,
                model_type=model_type,
                repo=repo,
            ),
            model_key=key,
        )
        if row is not None:
            rows.append(row)
    return rows


def collect_forecast_run_metrics(
    model_runs: dict[tuple[str, str], str],
    *,
    repo: Path,
) -> list[dict[str, float | int | str]]:
    """Compute metrics from explicitly selected immutable backfill runs."""
    from gridiron_edge.evaluation.metrics import build_forecast_run_evaluation_df

    rows: list[dict[str, float | int | str]] = []
    for (model_name, model_type), run_id in model_runs.items():
        key = f"{model_name}_{model_type}"
        row = _classification_metric_row(
            build_forecast_run_evaluation_df(
                run_id=run_id,
                model_name=model_name,
                model_type=model_type,
                repo=repo,
            ),
            model_key=key,
        )
        if row is not None:
            rows.append(row)
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
        target_key: Composite ModelRegistry key of the model to analyse
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
