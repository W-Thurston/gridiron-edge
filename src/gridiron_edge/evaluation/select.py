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
registered_game_model_pairs    Every (model_name, model_type) pair game-model evaluation owns.
latest_backfilled_run_id       Most recent immutable backfill run_id for one family.
collect_forecast_run_metrics   Compute metrics from explicitly selected immutable runs.
collect_latest_forecast_run_metrics
                                Convenience: resolve each family's latest run, then evaluate.
build_latest_run_evaluation_df Evaluate every family matching an optional
                                model_name/model_type filter at its own latest run.
rank_models                    Rank a list of metric dicts by composite criteria.
compute_report_data            Load one immutable run's predictions and compute all
                                four report DataFrames.
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


def registered_game_model_pairs() -> list[tuple[str, str]]:
    """Return every (model_name, model_type) pair game-model evaluation owns."""
    from gridiron_edge.models.game_prediction.model import get_known_model_names
    from gridiron_edge.models.registry import ModelRegistry

    prefixes = tuple(f"{model_name}_" for model_name in get_known_model_names())
    return [_parse_composite_key(key) for key in ModelRegistry.names() if key.startswith(prefixes)]


def build_latest_run_evaluation_df(
    *,
    model_name: str | None,
    model_type: str | None,
    repo: Path,
) -> DataFrame:
    """Evaluate every family matching an optional filter at its own latest run.

    A family with no backfill run yet is silently excluded. Replaces the
    legacy archive-backed ``build_evaluation_df`` for callers that filter by
    ``model_name``/``model_type`` rather than supplying an exact composite
    key list.
    """
    import pandas as pd

    from gridiron_edge.evaluation.metrics import build_forecast_run_evaluation_df

    pairs = [
        (name, type_)
        for name, type_ in registered_game_model_pairs()
        if (model_name is None or name == model_name)
        and (model_type is None or type_ == model_type)
    ]
    frames: list[DataFrame] = []
    for name, type_ in pairs:
        run_id = latest_backfilled_run_id(name, type_, repo=repo)
        if run_id is None:
            continue
        frame = build_forecast_run_evaluation_df(
            run_id=run_id, model_name=name, model_type=type_, repo=repo
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


def latest_backfilled_run_id(
    model_name: str,
    model_type: str,
    *,
    repo: Path,
) -> str | None:
    """Return the most recently generated immutable backfill run_id, if any.

    "Most recent" is by the run's own ``generated_at`` timestamp, never by
    load-time recency of anything request-scoped - this only selects among
    already-persisted immutable backfill runs for one family. Ties (e.g. a
    an identical timestamp) are broken by ``run_id`` for determinism.
    """
    from gridiron_edge.evaluation.forecast_contracts import ForecastRole
    from gridiron_edge.evaluation.forecast_store import load_forecast_events

    events = load_forecast_events(
        model_name=model_name,
        model_type=model_type,
        role=ForecastRole.BACKFILLED,
        repo=repo,
    )
    if events.empty:
        return None
    by_run = events.groupby("run_id")["generated_at"].max().sort_values()
    latest_timestamp = by_run.iloc[-1]
    candidates = sorted(by_run.index[by_run == latest_timestamp])
    return str(candidates[-1])


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


def collect_latest_forecast_run_metrics(
    model_keys: list[str],
    *,
    repo: Path,
) -> list[dict[str, float | int | str]]:
    """Resolve each family's latest immutable backfill run, then evaluate it.

    Convenience wrapper over ``collect_forecast_run_metrics`` for CLI
    commands that want "all registered models, whatever's freshest" without
    the caller having to look up every run_id by hand. Families with no
    backfill run yet are silently skipped, matching the legacy archive
    path's prior behavior of skipping models with no evaluable data.
    """
    model_runs: dict[tuple[str, str], str] = {}
    for key in model_keys:
        model_name, model_type = _parse_composite_key(key)
        run_id = latest_backfilled_run_id(model_name, model_type, repo=repo)
        if run_id is not None:
            model_runs[(model_name, model_type)] = run_id
    return collect_forecast_run_metrics(model_runs, repo=repo)


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
    run_id: str,
    season: str | None,
    top_misses: int,
    repo: Path,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load one immutable backfill run and compute all four report DataFrames.

    Args:
        target_key: Composite ModelRegistry key of the model to analyse
            (e.g. ``"win_prob_random_forest"``).
        run_id: Exact immutable backfill run to evaluate.
        season: Optional season post-filter applied to the run's own games.
        top_misses: Number of worst predictions to surface.
        repo: Repository root.

    Returns:
        Tuple of (df_eval, df_tiers, df_seasons, df_misses).

    Raises:
        ValueError: If ``target_key`` is not a valid composite key, or if
            no completed games are found for the model (or the season filter
            excludes every game in the run).
    """
    from gridiron_edge.evaluation.metrics import (
        biggest_misses,
        brier_by_confidence_tier,
        brier_by_season,
        build_forecast_run_evaluation_df,
    )

    model_name, model_type = _parse_composite_key(target_key)

    df_eval: DataFrame = build_forecast_run_evaluation_df(
        run_id=run_id,
        model_name=model_name,
        model_type=model_type,
        repo=repo,
    )
    if season is not None:
        df_eval = df_eval.loc[df_eval["season"] == season].reset_index(drop=True)
    if df_eval.empty:
        raise ValueError(f"No completed games found for {target_key!r} in run {run_id!r}.")

    df_tiers: DataFrame = brier_by_confidence_tier(df_eval)
    df_seasons: DataFrame = brier_by_season(df_eval)
    df_misses: DataFrame = biggest_misses(df_eval, n=top_misses)

    return df_eval, df_tiers, df_seasons, df_misses
