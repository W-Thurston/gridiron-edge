# src/gridiron_edge/cli/full_retrain.py

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
        --only promote-champions \
        --only baseline-report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    StageResult,
    render_composite_summary,
    resolve_active_stages,
    run_composite,
)
from gridiron_edge.core.console import console
from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.archive import load_prediction_log
from gridiron_edge.evaluation.backfill import backfill_model
from gridiron_edge.evaluation.prop_archive import archive_prop_predictions
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.catalog import (
    GAME_MODEL_PAIRS as _GAME_MODEL_PAIRS,
)
from gridiron_edge.models.catalog import (
    PROP_ALGORITHMS as _PROP_ALGORITHMS,
)
from gridiron_edge.models.catalog import (
    PROP_STAT_FAMILIES as _PROP_STAT_FAMILIES,
)
from gridiron_edge.models.game_prediction.game_schema import (
    ACTUAL_MARGIN_TARGET,
)

# ---------------------------------------------------------------------------
# Model pair catalog
# ---------------------------------------------------------------------------


__all__: list[str] = ["_GAME_MODEL_PAIRS", "_PROP_ALGORITHMS", "_PROP_STAT_FAMILIES"]


#: Minimum number of prior seasons required before prop walk-forward
#: will attempt a cutoff. Prop features include shift(1) rolling stats,
#: so the earliest available season produces all-NaN training rows and
#: ``train_through`` raises "No training rows precede cutoff_season=N".
#: Mirrors ``_MIN_WALK_FORWARD_TRAIN_SEASONS`` in evaluation/backfill.py.
_MIN_PROP_WALK_FORWARD_TRAIN_SEASONS: int = 3


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

    active: set[str] = set(ALL_STAGES) - {"fetch-weather"}

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


def _stage_promote_champions(ctx: dict[str, Any]) -> StageResult:
    """Rank all model families and persist the champion manifest.

    Thin adapter over ``evaluation.champion.promote_champions``. See that
    function for subset semantics.
    """
    from gridiron_edge.evaluation.champion import promote_champions

    repo = get_settings().repo_root

    game_pairs: list[ModelPair] = ctx["game_pairs"]
    prop_pairs: list[tuple[str, str]] = ctx["prop_pairs"]

    game_pair_tuples: list[tuple[str, str]] = [(p.model_name, p.model_type) for p in game_pairs]
    prop_families: list[str] = sorted({stat for stat, _algorithm in prop_pairs})

    result = promote_champions(
        game_pairs=game_pair_tuples,
        prop_families=prop_families,
        repo=repo,
    )

    detail = (
        f"{len(result.fresh_entries)} fresh champion(s); "
        f"{len(result.preserved_entries)} preserved from prior manifest"
    )

    return StageResult(
        success=True,
        detail=detail,
        rows=result.total_count,
        artifacts=[result.manifest_path],
        warnings=result.warnings,
    )


def _stage_backfill_game_models(ctx: dict[str, Any]) -> StageResult:
    """Walk-forward backfill all selected game model pairs.

    Iterates over the requested (model_name, model_type) pairs and
    delegates to backfill_model for each. Each pair runs to completion
    before the next starts.
    """
    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no game pairs requested")

    total_events = 0
    pair_summaries: list[str] = []

    for pair in pairs:
        n = backfill_model(
            model_name=pair.model_name,
            model_type=pair.model_type,
            mode=None,  # auto-resolve per model
        )
        total_events += n
        pair_summaries.append(f"{pair.composite_key}={n:,}")

    return StageResult(
        success=True,
        detail=(f"{total_events:,} backfilled forecast events across {len(pairs)} pairs"),
        rows=total_events,
    )


def _stage_backfill_prop_models(ctx: dict[str, Any]) -> StageResult:
    """Walk-forward backfill all selected prop model pairs.

    Iterates over (stat_family, algorithm) pairs and delegates to
    the canonical prop walk-forward implementation in
    ``cli/props.py``. Uses the same NaN policy (>50% column threshold)
    as ``gridiron props backfill``.
    """
    from gridiron_edge.cli.props import _walk_forward_predict_for_season
    from gridiron_edge.features.player.builder import build_prop_features
    from gridiron_edge.models.prop_prediction.base import PropModelType, PropTrainer
    from gridiron_edge.models.registry import ModelRegistry

    pairs: list[tuple[str, str]] = ctx["prop_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no prop pairs requested")

    # Trigger registry population.
    from typing import cast

    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401

    total_events = 0
    pair_summaries: list[str] = []

    for stat_family, algorithm in pairs:
        model_cls = ModelRegistry.get(stat_family)
        trainer_typed = cast(PropTrainer, model_cls())

        # Build features once per pair; walk-forward slices by season.
        features_df = build_prop_features(
            position_filter=trainer_typed.spec.position_filter,
        )
        seasons_available: list[int] = sorted(
            int(s) for s in features_df["season"].dropna().unique().tolist()
        )

        # Walk-forward: predict each season using a model trained
        # through the prior season. Skip the earliest season since
        # there's no prior training window.
        pair_n = 0
        for season in seasons_available[1:]:
            enriched, _ = _walk_forward_predict_for_season(
                model_name=stat_family,
                model_type=PropModelType(algorithm),
                season=season,
                features_df=features_df,
            )
            if enriched.empty:
                continue

            archive_prop_predictions(
                enriched,
                is_backfilled=True,
                model_name=stat_family,
                model_type=algorithm,
            )
            pair_n += len(enriched)

        total_events += pair_n
        pair_summaries.append(f"{stat_family}/{algorithm}={pair_n:,}")

    return StageResult(
        success=True,
        detail=f"{total_events:,} predictions across {len(pairs)} pairs",
        rows=total_events,
    )


def _stage_refresh_calibrations(ctx: dict[str, Any]) -> StageResult:
    """Recompute and persist sigma + margin_std for each game model.

    Reads the newly-built game prediction archive for each
    ``(model_name, model_type)`` pair, runs ``calibrate_spread_sigma``,
    computes residual ``margin_std``, updates the in-memory maps for the
    current process, and persists values to the disk-backed calibration
    registry consumed by post-processing.
    """
    from gridiron_edge.models.game_prediction.post_process import (
        _MODEL_MARGIN_STDS,
        calibrate_spread_sigma,
        compute_margin_std,
        register_sigma,
        save_model_calibration,
    )

    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no game pairs to calibrate")

    refreshed: list[str] = []
    skipped: list[str] = []

    repo: Path = get_settings().repo_root

    modeling: DataFrame = loaders.load_modeling_file(repo)

    if ACTUAL_MARGIN_TARGET not in modeling.columns:
        return StageResult(
            success=False,
            detail=(f"canonical modeling artifact is missing {ACTUAL_MARGIN_TARGET}"),
        )

    actuals = (
        modeling.loc[
            :,
            [
                "GAME_ID",
                ACTUAL_MARGIN_TARGET,
            ],
        ]
        .dropna(
            subset=[
                "GAME_ID",
                ACTUAL_MARGIN_TARGET,
            ]
        )
        .copy()
    )

    if actuals["GAME_ID"].duplicated().any():
        duplicate_ids: list = sorted(
            actuals.loc[
                actuals["GAME_ID"].duplicated(keep=False),
                "GAME_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        return StageResult(
            success=False,
            detail=(
                "canonical modeling artifact contains "
                "duplicate game IDs: " + ", ".join(duplicate_ids)
            ),
        )

    for pair in pairs:
        if pair.model_name != "win_prob":
            skipped.append(f"{pair.composite_key} (not win_prob)")
            continue

        archive: DataFrame = load_prediction_log(
            model_name=pair.model_name,
            model_type=pair.model_type,
        )
        if archive.empty:
            skipped.append(f"{pair.composite_key} (empty archive)")
            continue

        merged: DataFrame = archive.merge(
            actuals,
            left_on="game_id",
            right_on="GAME_ID",
            how="inner",
            validate="many_to_one",
        )

        if merged.empty:
            skipped.append(f"{pair.composite_key} (no game matches)")
            continue

        sigma: float = calibrate_spread_sigma(
            home_win_probs=merged["home_win_prob"],
            actual_margins=merged[ACTUAL_MARGIN_TARGET],
        )

        register_sigma(
            pair.model_name,
            pair.model_type,
            sigma,
        )

        margin_std: float = compute_margin_std(
            home_win_probs=merged["home_win_prob"],
            actual_margins=merged[ACTUAL_MARGIN_TARGET],
            sigma=sigma,
        )

        _MODEL_MARGIN_STDS[(pair.model_name, pair.model_type)] = margin_std

        save_model_calibration(
            model_name=pair.model_name,
            model_type=pair.model_type,
            sigma=sigma,
            margin_std=margin_std,
            repo=repo,
        )

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


def _stage_promote_champions(ctx: dict[str, Any]) -> StageResult:
    """Rank all model families and persist the champion manifest.

    Runs three selectors (game classification, game regression, prop),
    merges their results with any existing manifest entries for families
    outside the current subset, and writes the combined manifest via
    ``write_manifest``.

    Subset semantics:
        - Ranking respects ``ctx["game_pairs"]`` and ``ctx["prop_pairs"]``.
          If the CLI passed ``--game-models win_prob_random_forest``,
          only that pair participates in win_prob's ranking.
        - Existing manifest entries for families NOT touched by this run
          are preserved verbatim. A partial retrain never shrinks the
          manifest.
        - Cold-start (no prior manifest): only families in the current
          subset are written.
    """
    from gridiron_edge.evaluation.champion import (
        select_game_classification_champions,
        select_game_regression_champions,
        select_prop_champions_all_families,
    )
    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        read_manifest,
        write_manifest,
    )

    repo = get_settings().repo_root

    game_pairs: list[ModelPair] = ctx["game_pairs"]
    prop_pairs: list[tuple[str, str]] = ctx["prop_pairs"]

    game_pair_tuples: list[tuple[str, str]] = [(p.model_name, p.model_type) for p in game_pairs]
    prop_families: list[str] = sorted({stat for stat, _algorithm in prop_pairs})

    classification_entries = select_game_classification_champions(
        game_pair_tuples,
        repo=repo,
    )
    regression_entries = select_game_regression_champions(
        game_pair_tuples,
        repo=repo,
    )
    prop_entries = select_prop_champions_all_families(
        prop_families,
        repo=repo,
    )

    fresh_entries: dict[str, dict[str, Any]] = {
        **classification_entries,
        **regression_entries,
        **prop_entries,
    }

    warnings: list[str] = []
    if game_pair_tuples and not classification_entries and not regression_entries:
        warnings.append(
            "no game champions selected — check that game backfill "
            "produced archive rows and artifacts"
        )
    if prop_families and not prop_entries:
        warnings.append(
            "no prop champions selected — check that prop backfill produced archive rows"
        )

    try:
        existing = read_manifest(repo=repo)
        existing_models: dict[str, dict[str, Any]] = existing.get("models", {})
    except ChampionNotFoundError:
        existing_models = {}

    merged_entries: dict[str, dict[str, Any]] = {**existing_models, **fresh_entries}

    source_run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_path = write_manifest(
        merged_entries,
        source_run_id=source_run_id,
        repo=repo,
    )

    fresh_count = len(fresh_entries)
    preserved_count = len(set(existing_models) - set(fresh_entries))
    detail = f"{fresh_count} fresh champion(s); {preserved_count} preserved from prior manifest"

    return StageResult(
        success=True,
        detail=detail,
        rows=len(merged_entries),
        artifacts=[manifest_path],
        warnings=warnings,
    )


_METRIC_SPECS: list[tuple[str, str, int]] = [
    ("brier", "Brier", 4),
    ("ece", "ECE", 4),
    ("auc", "AUC", 4),
    ("mae", "MAE", 2),
    ("rmse", "RMSE", 2),
    ("r2", "R²", 3),
]


def _find_previous_baseline_report(out_dir: Path) -> Path | None:
    """Return the most recent full-retrain report, if one exists."""
    reports = sorted(out_dir.glob("full-retrain-*.md"))
    return reports[-1] if reports else None


def _parse_metric_cell(value: str) -> float | None:
    """Parse a metric table cell from a baseline report.

    Returns None for em dash, missing values, no-artifact markers, or
    otherwise non-numeric cells.
    """
    cleaned = value.strip()
    if cleaned in {"", "-", "- no artifact -"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_baseline_report(path: Path) -> dict[str, dict[str, float | None]]:
    """Parse game-model metric rows from a full-retrain markdown report.

    Expected table shape:

        | Pair | Brier | ECE | AUC | MAE | RMSE | R² |
        |---|---|---|---|---|---|---|
        | win_prob_logistic | 0.2215 | 0.0153 | ... |

    Rows with missing artifacts are kept with all metric values as None.
    """
    rows: dict[str, dict[str, float | None]] = {}

    for line in path.read_text().splitlines():
        if not line.startswith("| "):
            continue
        if line.startswith("| Pair ") or line.startswith("|---"):
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue

        pair = cells[0]
        if pair in {"Pair", ""}:
            continue

        metrics: dict[str, float | None] = {key: None for key, _, _ in _METRIC_SPECS}

        # Normal metric row has 7 cells. A no-artifact row may have
        # fewer cells; keep it with None metrics.
        metric_cells = cells[1:]
        for (key, _, _), cell in zip(_METRIC_SPECS, metric_cells, strict=False):
            metrics[key] = _parse_metric_cell(cell)

        rows[pair] = metrics

    return rows


def _format_metric_value(value: float | None, decimals: int) -> str:
    """Format a metric value or em dash when missing."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _build_current_metrics(
    *,
    pairs: list[ModelPair],
    store: ArtifactStore,
) -> dict[str, dict[str, float | None]]:
    """Build current metric snapshot keyed by composite model key."""
    current_metrics: dict[str, dict[str, float | None]] = {}

    for pair in pairs:
        metrics_for_pair: dict[str, float | None] = {key: None for key, _, _ in _METRIC_SPECS}

        if store.is_trained(pair.model_name, pair.model_type):
            meta = store.read_metadata(pair.model_name, pair.model_type)
            for key, _, _ in _METRIC_SPECS:
                metrics_for_pair[key] = meta.metrics.get(key)

        current_metrics[pair.composite_key] = metrics_for_pair

    return current_metrics


def _build_delta_row(
    *,
    pair_key: str,
    current: dict[str, float | None],
    previous: dict[str, float | None],
) -> str:
    """Build one markdown delta row."""
    values: list[str] = [pair_key]

    for key, _, decimals in _METRIC_SPECS:
        values.append(
            _format_metric_delta(
                current=current.get(key),
                previous=previous.get(key),
                decimals=decimals,
            )
        )

    return "| " + " | ".join(values) + " |"


def _append_current_metrics_table(
    *,
    lines: list[str],
    pairs: list[ModelPair],
    current_metrics: dict[str, dict[str, float | None]],
) -> None:
    """Append current metrics table to report lines."""
    lines.append("## Game Models")
    lines.append("")
    lines.append("| Pair | Brier | ECE | AUC | MAE | RMSE | R² |")
    lines.append("|---|---|---|---|---|---|---|")

    for pair in pairs:
        metrics = current_metrics[pair.composite_key]

        if all(value is None for value in metrics.values()):
            lines.append(f"| {pair.composite_key} | - no artifact - |")
            continue

        lines.append(
            f"| {pair.composite_key} | "
            f"{_format_metric_value(metrics['brier'], 4)} | "
            f"{_format_metric_value(metrics['ece'], 4)} | "
            f"{_format_metric_value(metrics['auc'], 4)} | "
            f"{_format_metric_value(metrics['mae'], 2)} | "
            f"{_format_metric_value(metrics['rmse'], 2)} | "
            f"{_format_metric_value(metrics['r2'], 3)} |"
        )


def _append_delta_table(
    *,
    lines: list[str],
    pairs: list[ModelPair],
    current_metrics: dict[str, dict[str, float | None]],
    previous_metrics: dict[str, dict[str, float | None]],
    previous_report: Path | None,
) -> None:
    """Append delta-vs-previous table to report lines."""
    lines.append("")
    lines.append("## Delta vs Previous Report")
    lines.append("")

    if previous_report is None:
        lines.append("No previous report found; delta table omitted.")
        return

    lines.append(
        "Deltas are current minus previous. Negative is better for "
        "Brier, ECE, MAE, and RMSE. Positive is better for AUC and R²."
    )
    lines.append("")
    lines.append("| Pair | Δ Brier | Δ ECE | Δ AUC | Δ MAE | Δ RMSE | Δ R² |")
    lines.append("|---|---|---|---|---|---|---|")

    for pair in pairs:
        key = pair.composite_key

        lines.append(
            _build_delta_row(
                pair_key=key,
                current=current_metrics.get(key, {}),
                previous=previous_metrics.get(key, {}),
            )
        )


def _format_metric_delta(
    *,
    current: float | None,
    previous: float | None,
    decimals: int,
) -> str:
    """Format signed metric delta as current - previous."""
    if current is None or previous is None:
        return "-"
    delta = current - previous
    return f"{delta:+.{decimals}f}"


def _append_champions_block(
    *,
    lines: list[str],
    repo: Path,
) -> None:
    """Append the current champions summary block to the report lines.

    Reads the manifest via ``list_current_champions``. Silent no-op when
    no manifest exists (cold-start report generation before manifest
    was ever written). Uses a bullet list rather than a markdown table
    so ``_parse_baseline_report`` (which looks for pipe-delimited rows)
    ignores this block entirely.
    """
    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        read_manifest,
    )

    try:
        manifest = read_manifest(repo=repo)
    except ChampionNotFoundError:
        return

    models = manifest.get("models", {})
    if not models:
        return

    lines.append("## Current Champions")
    lines.append("")
    lines.append(f"Manifest updated: {manifest.get('updated_at', 'unknown')}")
    lines.append("")

    for model_name in sorted(models.keys()):
        entry = models[model_name]
        model_type = entry.get("model_type", "?")
        promoted_at = entry.get("promoted_at", "?")
        composite_key = f"{model_name}_{model_type}"
        lines.append(f"- **{model_name}** → 🏆 `{composite_key}` (promoted {promoted_at})")

    lines.append("")


def _stage_baseline_report(ctx: dict[str, Any]) -> StageResult:
    """Write a markdown report comparing new baselines to prior values.

    Reads each trained artifact's metadata ``metrics`` dict, writes the
    current metric table, and - when a previous full-retrain report
    exists - writes a delta table comparing current metrics against the
    previous report.
    """
    repo = get_settings().repo_root
    store = ArtifactStore(repo)

    pairs: list[ModelPair] = ctx["game_pairs"]
    if not pairs:
        return StageResult(success=True, detail="no pairs to report")

    out_dir = repo / "data" / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_report = _find_previous_baseline_report(out_dir)
    previous_metrics: dict[str, dict[str, float | None]] = {}
    if previous_report is not None:
        previous_metrics = _parse_baseline_report(previous_report)

    timestamp = datetime.now()
    out_path = out_dir / f"full-retrain-{timestamp.strftime('%Y-%m-%d-%H%M%S')}.md"

    current_metrics: dict[str, dict[str, float | None]] = _build_current_metrics(
        pairs=pairs,
        store=store,
    )

    lines: list[str] = []
    lines.append("# Full Retrain Baseline Report")
    lines.append("")
    lines.append(f"Generated: {timestamp.isoformat()}")
    lines.append("")
    if previous_report is None:
        lines.append("Previous report: none found")
    else:
        lines.append(f"Previous report: `{previous_report.name}`")
    lines.append("")

    _append_champions_block(lines=lines, repo=repo)

    _append_current_metrics_table(
        lines=lines,
        pairs=pairs,
        current_metrics=current_metrics,
    )

    _append_delta_table(
        lines=lines,
        pairs=pairs,
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        previous_report=previous_report,
    )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "Baselines reflect the current trained artifact metadata. "
        "Sigma and margin_std values are refreshed by the "
        "refresh-calibrations stage and persisted to "
        "`data/output/calibration/game_model_calibration.json`."
    )

    out_path.write_text("\n".join(lines))

    detail = f"baseline report written ({len(pairs)} pairs)"
    if previous_report is not None:
        detail += f" vs {previous_report.name}"

    return StageResult(
        success=True,
        detail=detail,
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
            name="promote-champions",
            description="Rank families and persist champion manifest",
            func=_stage_promote_champions,
            depends_on=("refresh-calibrations",),
        ),
        CompositeStage(
            name="baseline-report",
            description="Write baseline comparison markdown",
            func=_stage_baseline_report,
            depends_on=("promote-champions",),
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
    assume_done: list[str] = typer.Option(  # noqa: B008
        [],
        "--assume-done",
        help=(
            "Stage(s) that completed in a prior run and whose artifacts "
            "are on disk. Their dependencies are considered satisfied "
            "without re-running them. Useful for resuming after a failure."
        ),
    ),
) -> None:
    r"""Heavy full-retrain workflow: all data, all models, all calibrations.

    Composes five stages over the full historical archive. Designed
    as a weekend batch job - runtime is hours.

    \b
    Examples:
      gridiron full-retrain
      gridiron full-retrain --skip-prop-backfill
      gridiron full-retrain --game-models win_prob_random_forest
      gridiron full-retrain --only refresh-calibrations
    """
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
        assume_satisfied=set(assume_done),
    )

    render_composite_summary(summary)

    if not summary.overall_success:
        raise typer.Exit(code=1)
