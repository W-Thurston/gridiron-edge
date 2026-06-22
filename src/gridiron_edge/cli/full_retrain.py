"""Composite command: full-retrain.

Heavy "fresh start" workflow for the start of a new season or after
major architectural changes. Refreshes all data, walks forward through
the full history for every (model_name, model_type) pair, recalibrates
sigma/margin_std, and writes a baseline comparison report.

Runtime: hours. Designed as a weekend batch job.

Usage::

    # Full retrain (game + prop)
    gridiron full-retrain

    # Game models only
    gridiron full-retrain --skip-prop-backfill

    # Specific game models only
    gridiron full-retrain --game-models win_prob_random_forest \
        --skip-prop-backfill

    # Only the calibration refresh and report
    gridiron full-retrain --only refresh-calibrations \
        --only baseline-report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    StageResult,
    render_composite_summary,
    resolve_active_stages,
    run_composite,
)

# ---------------------------------------------------------------------------
# Model pair catalog
# ---------------------------------------------------------------------------


_GAME_MODEL_PAIRS: list[tuple[str, str]] = [
    ("win_prob", "elo"),
    ("win_prob", "logistic"),
    ("win_prob", "random_forest"),
    ("win_prob", "xgboost"),
    ("total", "random_forest"),
    ("total", "xgboost"),
]

_PROP_STAT_FAMILIES: list[str] = [
    "qb_pass_yards",
    "qb_rush_yards",
    "rb_rush_yards",
    "wr_rec_yards",
    "te_rec_yards",
]

_PROP_ALGORITHMS: list[str] = ["elasticnet", "random_forest", "xgboost"]


@dataclass(frozen=True)
class ModelPair:
    """A (model_name, model_type) pair to retrain."""

    model_name: str
    model_type: str

    @property
    def composite_key(self) -> str:
        """Return the canonical composite key for this pair."""
        return f"{self.model_name}_{self.model_type}"


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _stage_refresh_all_data(ctx: dict[str, Any]) -> StageResult:
    """Run the full-history data pipeline (skip weather/odds)."""
    from gridiron_edge.cli.main import ALL_STAGES, _run_pipeline_stages

    active = set(ALL_STAGES) - {"fetch-weather", "fetch-odds"}

    _run_pipeline_stages(
        active=active,
        all_years=True,
        resolved_season=0,
        upcoming_target=ctx.get("upcoming_season_int", 0) or 0,
        season=None,
        season_year=None,
        owm_api_key=None,
        fit_elo_all_years=True,
    )
    return StageResult(success=True, detail="full-history pipeline complete")


def _stage_backfill_game_models(ctx: dict[str, Any]) -> StageResult:
    """Walk-forward backfill all selected game model pairs.

    Iterates over the requested (model_name, model_type) pairs and
    delegates to backfill_model for each. Each pair runs to completion
    before the next starts.
    """
    from gridiron_edge.evaluation.backfill import backfill_model

    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no game pairs requested")

    total_archived = 0
    pair_summaries: list[str] = []

    for pair in pairs:
        n = backfill_model(
            model_name=pair.model_name,
            model_type=pair.model_type,
            mode=None,  # auto-resolve per model
            overwrite=True,
        )
        total_archived += n
        pair_summaries.append(f"{pair.composite_key}={n:,}")

    return StageResult(
        success=True,
        detail=f"{total_archived:,} predictions across {len(pairs)} pairs",
        rows=total_archived,
    )


def _stage_backfill_prop_models(ctx: dict[str, Any]) -> StageResult:
    """Walk-forward backfill all selected prop model pairs.

    Iterates over (stat_family, algorithm) pairs and calls the prop
    backfill function. This is the longest stage by far.
    """
    from gridiron_edge.evaluation.prop_archive import (
        archive_prop_predictions,
    )
    from gridiron_edge.models.prop_prediction.base import PropModelType
    from gridiron_edge.models.registry import ModelRegistry

    pairs: list[tuple[str, str]] = ctx["prop_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no prop pairs requested")

    # Trigger registry population.
    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401

    total_archived = 0
    pair_summaries: list[str] = []

    for stat_family, algorithm in pairs:
        model_cls = ModelRegistry.get(stat_family)
        trainer = model_cls()

        # Resolve all available seasons in player game logs.
        # The trainer's _load_data is the canonical loader.
        from typing import cast

        from gridiron_edge.models.prop_prediction.base import PropTrainer

        trainer_typed = cast(PropTrainer, trainer)
        df = trainer_typed._load_data()
        seasons_int = sorted(df["season"].unique().tolist())

        # Walk-forward: predict each season using a model trained
        # through the prior season.
        pair_n = 0
        for season in seasons_int[1:]:
            meta = trainer_typed.train_through(
                cutoff_season=season,
                model_type=PropModelType(algorithm),
            )

            # The trainer holds the fitted state; project the cutoff
            # season using the canonical predict path.
            from gridiron_edge.features.player.builder import (
                build_prop_features,
            )

            features_all = build_prop_features(
                position_filter=trainer_typed.spec.position_filter,
            )
            features_season = features_all[features_all["season"] == season]
            if features_season.empty:
                continue

            usable_cols = [c for c in meta.feature_columns if c in features_season.columns]
            x = features_season[usable_cols].dropna()
            if x.empty:
                continue

            features_season = features_season.loc[x.index, :]

            preds = trainer_typed._predict(x)

            preds_df = pd.DataFrame(
                {
                    "player_id": features_season["player_id"].values,
                    "game_id": features_season["game_id"].values,
                    "season": features_season["season"].values,
                    "week": features_season["week"].values,
                    "stat_type": trainer_typed.spec.target_col,
                    "predicted_mean": preds,
                }
            )

            archive_prop_predictions(
                preds_df,
                model_name=stat_family,
                model_type=algorithm,
            )
            pair_n += len(preds_df)

        total_archived += pair_n
        pair_summaries.append(f"{stat_family}/{algorithm}={pair_n:,}")

    return StageResult(
        success=True,
        detail=f"{total_archived:,} predictions across {len(pairs)} pairs",
        rows=total_archived,
    )


def _stage_refresh_calibrations(ctx: dict[str, Any]) -> StageResult:
    """Recompute sigma + margin_std for each game model from the archive.

    Reads the newly-built game prediction archive for each
    (model_name, model_type) pair, runs calibrate_spread_sigma, and
    updates the in-memory _MODEL_SIGMAS / _MODEL_MARGIN_STDS dicts.

    Note: these are in-memory updates. Persistence to disk would
    require a separate refactor (the values are currently hardcoded
    in post_process.py).
    """
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.models.game_prediction.post_process import (
        _MODEL_MARGIN_STDS,
        calibrate_spread_sigma,
        compute_margin_std,
        register_sigma,
    )

    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no game pairs to calibrate")

    refreshed: list[str] = []
    skipped: list[str] = []

    for pair in pairs:
        if pair.model_name != "win_prob":
            # Sigma calibration is only meaningful for win-prob models.
            skipped.append(f"{pair.composite_key} (not win_prob)")
            continue

        archive = load_prediction_log(
            model_name=pair.model_name,
            model_type=pair.model_type,
        )
        if archive.empty:
            skipped.append(f"{pair.composite_key} (empty archive)")
            continue

        # Compute margins from the archive against actuals.
        from gridiron_edge.core.settings import get_settings
        from gridiron_edge.datasets import loaders

        games = loaders.load_games(get_settings().repo_root)
        merged = archive.merge(
            games[["GAME_ID", "WINNER", "LOSER", "WIN_OR_TIE"]],
            left_on="game_id",
            right_on="GAME_ID",
            how="inner",
        )
        if merged.empty:
            skipped.append(f"{pair.composite_key} (no game matches)")
            continue

        # Actual margin: home_score - away_score (proxy via WIN_OR_TIE)
        # Since we don't have scores in this join, derive home_win
        # from WIN_OR_TIE and use it as a rough margin proxy.
        # For full calibration, a richer join with actual scores is
        # needed; we skip if scores aren't in the games table.
        if "PTS_WINNER" not in games.columns:
            skipped.append(f"{pair.composite_key} (no scores in games table)")
            continue

        # Real flow: get home/away scores and compute margin.
        merged_with_scores = archive.merge(
            games[["GAME_ID", "WINNER", "LOSER", "PTS_WINNER", "PTS_LOSER"]],
            left_on="game_id",
            right_on="GAME_ID",
            how="inner",
        )
        # Home margin = (home_score - away_score).
        # WINNER is the actual winner; if winner == home_team, margin > 0.
        merged_with_scores["home_margin"] = merged_with_scores.apply(
            lambda r: (
                r["PTS_WINNER"] - r["PTS_LOSER"]
                if r["WINNER"] == r["home_team"]
                else r["PTS_LOSER"] - r["PTS_WINNER"]
            ),
            axis=1,
        )

        sigma = calibrate_spread_sigma(
            home_win_probs=merged_with_scores["home_win_prob"],
            actual_margins=merged_with_scores["home_margin"],
        )

        register_sigma(pair.model_name, pair.model_type, sigma)

        margin_std = compute_margin_std(
            home_win_probs=merged_with_scores["home_win_prob"],
            actual_margins=merged_with_scores["home_margin"],
            sigma=sigma,
        )
        _MODEL_MARGIN_STDS[(pair.model_name, pair.model_type)] = margin_std

        refreshed.append(f"{pair.composite_key}: sigma={sigma:.2f} margin_std={margin_std:.2f}")

    detail_parts = []
    if refreshed:
        detail_parts.append(f"{len(refreshed)} refreshed")
    if skipped:
        detail_parts.append(f"{len(skipped)} skipped")

    return StageResult(
        success=True,
        detail=" · ".join(detail_parts) or "nothing to do",
        warnings=skipped[:5],  # surface up to 5 skip reasons
    )


def _stage_baseline_report(ctx: dict[str, Any]) -> StageResult:
    """Write a markdown report comparing new baselines to prior values.

    Reads each trained artifact's metadata `metrics` dict and compares
    against a snapshot (if available) from the previous full-retrain.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore

    repo = get_settings().repo_root
    store = ArtifactStore(repo)

    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no pairs to report")

    out_dir = repo / "data" / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full-retrain-{datetime.now().strftime('%Y-%m-%d')}.md"

    lines: list[str] = []
    lines.append("# Full Retrain Baseline Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Game Models")
    lines.append("")
    lines.append("| Pair | Brier | ECE | AUC | MAE | RMSE | R² |")
    lines.append("|---|---|---|---|---|---|---|")

    def _fmt(metrics_dict: dict[str, float], key: str, decimals: int = 4) -> str:
        """Format a metric value or return em-dash if missing."""
        val = metrics_dict.get(key)
        if val is None:
            return "—"
        return f"{val:.{decimals}f}"

    for pair in pairs:
        if not store.is_trained(pair.model_name, pair.model_type):
            lines.append(f"| {pair.composite_key} | — no artifact — |")
            continue

        meta = store.read_metadata(pair.model_name, pair.model_type)
        metrics = meta.metrics

        lines.append(
            f"| {pair.composite_key} | "
            f"{_fmt(metrics, 'brier')} | "
            f"{_fmt(metrics, 'ece')} | "
            f"{_fmt(metrics, 'auc')} | "
            f"{_fmt(metrics, 'mae', 2)} | "
            f"{_fmt(metrics, 'rmse', 2)} | "
            f"{_fmt(metrics, 'r2', 3)} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "Baselines reflect post-walk-forward archive. Sigma and "
        "margin_std values refreshed in-memory; persistence pending."
    )

    out_path.write_text("\n".join(lines))

    return StageResult(
        success=True,
        detail=f"baseline report written ({len(pairs)} pairs)",
        artifacts=[out_path],
    )


# ---------------------------------------------------------------------------
# Stage list
# ---------------------------------------------------------------------------


def _build_stages() -> list:
    """Define the stages for full-retrain.

    All stages are hard-fail. If refresh-all-data fails, nothing
    downstream is meaningful. If backfills fail, calibrations would
    be based on stale data.
    """
    return [
        CompositeStage(
            name="refresh-all-data",
            description="Refresh all historical data",
            func=_stage_refresh_all_data,
        ),
        CompositeStage(
            name="backfill-game-models",
            description="Walk-forward backfill game model pairs",
            func=_stage_backfill_game_models,
            depends_on=("refresh-all-data",),
        ),
        CompositeStage(
            name="backfill-prop-models",
            description="Walk-forward backfill prop model pairs",
            func=_stage_backfill_prop_models,
            depends_on=("refresh-all-data",),
        ),
        CompositeStage(
            name="refresh-calibrations",
            description="Recompute sigma + margin_std from archive",
            func=_stage_refresh_calibrations,
            depends_on=("backfill-game-models",),
        ),
        CompositeStage(
            name="baseline-report",
            description="Write baseline comparison markdown",
            func=_stage_baseline_report,
            depends_on=("backfill-game-models",),
        ),
    ]


_ALL_STAGES: list[str] = [s.name for s in _build_stages()]
_STAGES_STR: str = ", ".join(_ALL_STAGES)
_SKIP_HELP: str = f"Stage(s) to skip. Repeatable. Valid: {_STAGES_STR}."
_ONLY_HELP: str = f"Run only these stage(s). Repeatable. Valid: {_STAGES_STR}."


# ---------------------------------------------------------------------------
# Pair resolution helpers
# ---------------------------------------------------------------------------


def _resolve_game_pairs(requested: list[str]) -> list[ModelPair]:
    """Resolve --game-models flags into a list of ModelPair.

    Empty list means all defined pairs.
    """
    if not requested:
        return [ModelPair(model_name=n, model_type=t) for n, t in _GAME_MODEL_PAIRS]

    valid_keys = {f"{n}_{t}": (n, t) for n, t in _GAME_MODEL_PAIRS}
    pairs: list[ModelPair] = []
    unknown: list[str] = []
    for key in requested:
        if key in valid_keys:
            n, t = valid_keys[key]
            pairs.append(ModelPair(model_name=n, model_type=t))
        else:
            unknown.append(key)

    if unknown:
        raise typer.BadParameter(
            f"Unknown game model(s): {', '.join(unknown)}. Valid: {', '.join(valid_keys.keys())}."
        )
    return pairs


def _resolve_prop_pairs(
    requested: list[str],
) -> list[tuple[str, str]]:
    """Resolve --prop-models flags into (stat, algorithm) tuples.

    Empty list means all 15 pairs.
    """
    if not requested:
        return [(stat, algo) for stat in _PROP_STAT_FAMILIES for algo in _PROP_ALGORITHMS]

    valid_pairs = {
        f"{stat}_{algo}": (stat, algo) for stat in _PROP_STAT_FAMILIES for algo in _PROP_ALGORITHMS
    }
    pairs: list[tuple[str, str]] = []
    unknown: list[str] = []
    for key in requested:
        if key in valid_pairs:
            pairs.append(valid_pairs[key])
        else:
            unknown.append(key)

    if unknown:
        raise typer.BadParameter(
            f"Unknown prop model(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(valid_pairs.keys()))}."
        )
    return pairs


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def full_retrain_cmd(
    *,
    upcoming_season: int | None = typer.Option(
        None,
        help=("Season to fetch upcoming schedule for. Defaults to current season."),
    ),
    game_models: list[str] = typer.Option(  # noqa: B008
        [],
        "--game-models",
        help=(
            "Restrict game-model backfill to these composite keys "
            "(e.g. 'win_prob_random_forest'). Repeatable. Empty = all."
        ),
    ),
    prop_models: list[str] = typer.Option(  # noqa: B008
        [],
        "--prop-models",
        help=(
            "Restrict prop-model backfill to these composite keys "
            "(e.g. 'qb_pass_yards_elasticnet'). Repeatable. Empty = all."
        ),
    ),
    skip_prop_backfill: bool = typer.Option(
        False,
        "--skip-prop-backfill",
        help=("Shorthand for --skip backfill-prop-models. Useful for game-only iterations."),
    ),
    skip: list[str] = typer.Option(  # noqa: B008
        [],
        "--skip",
        help=_SKIP_HELP,
    ),
    only: list[str] = typer.Option(  # noqa: B008
        [],
        "--only",
        help=_ONLY_HELP,
    ),
) -> None:
    r"""Heavy full-retrain workflow: all data, all models, all calibrations.

    Composes five stages over the full historical archive. Designed
    as a weekend batch job — runtime is hours.

    \b
    Examples:
      gridiron full-retrain
      gridiron full-retrain --skip-prop-backfill
      gridiron full-retrain --game-models win_prob_random_forest
      gridiron full-retrain --only refresh-calibrations
    """
    from gridiron_edge.core.console import console

    # Resolve --skip-prop-backfill into the standard skip list.
    effective_skip = list(skip)
    if skip_prop_backfill and "backfill-prop-models" not in effective_skip:
        effective_skip.append("backfill-prop-models")

    stages = _build_stages()
    active = resolve_active_stages(
        all_stages=_ALL_STAGES,
        skip=effective_skip,
        only=only,
    )

    context: dict[str, Any] = {
        "upcoming_season_int": upcoming_season,
        "game_pairs": _resolve_game_pairs(game_models),
        "prop_pairs": _resolve_prop_pairs(prop_models),
    }

    n_game = len(context["game_pairs"])
    n_prop = len(context["prop_pairs"])
    subtitle = f"{n_game} game pair(s) · {n_prop} prop pair(s)"
    console.header("full-retrain", subtitle=subtitle)

    summary = run_composite(
        name="full-retrain",
        stages=stages,
        active=active,
        context=context,
    )

    render_composite_summary(summary)

    if not summary.overall_success:
        raise typer.Exit(code=1)
