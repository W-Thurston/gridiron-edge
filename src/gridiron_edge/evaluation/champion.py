# src/gridiron_edge/evaluation/champion.py

"""Champion/challenger model comparison and promotion logic.

Classification:
    ClassificationPromotionGates           Gate thresholds (Brier/ECE/AUC)
    ClassificationComparisonResult         Comparison outcome
    extract_classification_metrics         Pull metrics from GameModelMetadata
    compare_classification_models          Run classification gates
    format_classification_comparison       Human-readable classification report

Regression:
    RegressionPromotionGates               Gate thresholds (R²/Coverage)
    RegressionModelResult                  Standardised regression metrics
    RegressionComparisonResult             Comparison outcome
    compare_regression_models              Run regression gates
    select_prop_champion                   Pick best model from multiple results
    format_regression_comparison           Human-readable regression report

Metric storage:
    Classification metrics (ECE, AUC, log_loss, accuracy) live on the
    ``metrics`` dict on :class:`BaseModelMetadata`, populated by
    ``GamesTrainer._build_classification_metadata``.
    ``extract_classification_metrics`` reads from that dict.

Usage::

    from gridiron_edge.evaluation.champion import (
        compare_classification_models,
        format_classification_comparison,
    )

    result = compare_classification_models(champion_meta, challenger_meta)
    print(format_classification_comparison(result))
    if result.should_promote:
        store.save(metadata=challenger_meta, model_obj=challenger_obj)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

if TYPE_CHECKING:
    from gridiron_edge.models.game_prediction.base import GameModelMetadata


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationPromotionGates:
    """Gate thresholds for champion/challenger promotion decisions.

    Attributes:
        min_brier_improvement: Challenger Brier must be at least this much
            lower than champion Brier to pass the primary gate.
        max_ece_degradation: Challenger ECE may be at most this much higher
            than champion ECE without failing the calibration gate.
        max_auc_degradation: Challenger AUC may be at most this much lower
            than champion AUC without failing the discrimination gate.
    """

    min_brier_improvement: float = 0.002
    max_ece_degradation: float = 0.01
    max_auc_degradation: float = 0.01


@dataclass(frozen=True)
class RegressionPromotionGates:
    """Gate thresholds for regression model promotion decisions.

    Attributes:
        min_r2: Minimum holdout R² to pass. Must beat mean baseline.
        min_coverage: Lower bound for 90% prediction interval coverage.
        max_coverage: Upper bound for 90% prediction interval coverage.
    """

    min_r2: float = 0.0
    min_coverage: float = 0.85
    max_coverage: float = 0.97


@dataclass(frozen=True)
class ClassificationComparisonResult:
    """Full outcome of a champion vs challenger comparison.

    Attributes:
        champion_metrics: Standardised metric dict for the champion.
        challenger_metrics: Standardised metric dict for the challenger.
        gates: Per-gate pass/fail results.
        should_promote: True if all gates passed.
        reason: Human-readable explanation of the verdict.
    """

    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    gates: dict[str, bool]
    should_promote: bool
    reason: str


@dataclass(frozen=True)
class PromoteChampionsResult:
    """Outcome of a manifest promotion run.

    Attributes:
        manifest_path: Path to the written manifest file.
        fresh_entries: Newly-selected champion entries from this run.
            Keyed by ``model_name``.
        preserved_entries: Entries carried forward verbatim from a
            prior manifest because their families were outside the
            current subset. Keyed by ``model_name``.
        warnings: Non-fatal warnings surfaced by the selectors.
    """

    manifest_path: Path
    fresh_entries: dict[str, dict[str, Any]]
    preserved_entries: dict[str, dict[str, Any]]
    warnings: list[str]

    @property
    def total_count(self) -> int:
        """Total number of entries in the merged manifest."""
        return len(self.fresh_entries) + len(self.preserved_entries)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_classification_metrics(meta: GameModelMetadata) -> dict[str, float]:
    """Pull a standardised metric dict from GameModelMetadata.

    Reads the classification metrics from the ``metrics`` dict on
    :class:`BaseModelMetadata` (Unit 9). Any metric not recorded at
    training time surfaces as NaN so the comparator gates can treat
    them uniformly.

    Args:
        meta: Trained model metadata.

    Returns:
        Dict with keys ``brier``, ``ece``, ``auc``, ``log_loss``,
        ``accuracy``.
    """
    return {
        "brier": meta.metrics.get("brier", float("nan")),
        "ece": meta.metrics.get("ece", float("nan")),
        "auc": meta.metrics.get("auc", float("nan")),
        "log_loss": meta.metrics.get("log_loss", float("nan")),
        "accuracy": meta.metrics.get("accuracy", float("nan")),
    }


@dataclass(frozen=True)
class RegressionModelResult:
    """Standardised regression metrics for champion comparison.

    Attributes:
        model_type: Algorithm identifier (e.g. "elasticnet", "random_forest").
        mae: Mean absolute error on holdout.
        rmse: Root mean squared error on holdout.
        r2: R-squared on holdout.
        coverage: Actual coverage of 90% prediction interval, or NaN if unavailable.
    """

    model_type: str
    mae: float
    rmse: float
    r2: float
    coverage: float = float("nan")


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_classification_models(
    champion: GameModelMetadata,
    challenger: GameModelMetadata,
    criteria: ClassificationPromotionGates | None = None,
) -> ClassificationComparisonResult:
    """Compare champion and challenger on all gates.

    Gate 1 (Brier): challenger must improve by at least
    ``criteria.min_brier_improvement``.

    Gate 2 (ECE): challenger ECE must not degrade by more than
    ``criteria.max_ece_degradation``.  Skipped (passes) if either
    model lacks ECE data.

    Gate 3 (AUC): challenger AUC must not degrade by more than
    ``criteria.max_auc_degradation``.  Skipped (passes) if either
    model lacks AUC data.

    Args:
        champion: Metadata for the current champion model.
        challenger: Metadata for the candidate challenger model.
        criteria: Gate thresholds.  Defaults to ``ClassificationPromotionGates()``.

    Returns:
        ``ClassificationComparisonResult`` with per-gate outcomes and overall verdict.
    """
    if criteria is None:
        criteria = ClassificationPromotionGates()

    champ: dict[str, float] = extract_classification_metrics(champion)
    chall: dict[str, float] = extract_classification_metrics(challenger)

    # Gate 1: Brier must meaningfully improve (lower is better)
    brier_delta: float = champ["brier"] - chall["brier"]
    brier_improved: bool = brier_delta >= criteria.min_brier_improvement

    # Gate 2: ECE must not degrade significantly (lower is better)
    if isnan(champ["ece"]) or isnan(chall["ece"]):
        ece_acceptable = True
    else:
        ece_acceptable: bool = chall["ece"] < champ["ece"] + criteria.max_ece_degradation

    # Gate 3: AUC must not degrade significantly (higher is better)
    if isnan(champ["auc"]) or isnan(chall["auc"]):
        auc_acceptable = True
    else:
        auc_acceptable: bool = chall["auc"] > champ["auc"] - criteria.max_auc_degradation

    gates: dict[str, bool] = {
        "brier_improved": brier_improved,
        "ece_acceptable": ece_acceptable,
        "auc_acceptable": auc_acceptable,
    }
    should_promote: bool = all(gates.values())

    # Build reason string
    if should_promote:
        reason: str = f"Challenger passes all gates (Brier improved by {brier_delta:.5f})"
    else:
        failed: list[str] = [name for name, passed in gates.items() if not passed]
        reason = f"Challenger failed {len(failed)} gate(s): {', '.join(failed)}"

    return ClassificationComparisonResult(
        champion_metrics=champ,
        challenger_metrics=chall,
        gates=gates,
        should_promote=should_promote,
        reason=reason,
    )


@dataclass(frozen=True)
class RegressionComparisonResult:
    """Outcome of a regression champion vs challenger comparison.

    Attributes:
        champion: Metrics for the current champion.
        challenger: Metrics for the candidate challenger.
        gates: Per-gate pass/fail for the challenger.
        should_promote: True if challenger passes all gates AND beats champion MAE.
        reason: Human-readable explanation.
    """

    champion: RegressionModelResult
    challenger: RegressionModelResult
    gates: dict[str, bool]
    should_promote: bool
    reason: str


def compare_regression_models(
    champion: RegressionModelResult,
    challenger: RegressionModelResult,
    gates: RegressionPromotionGates | None = None,
) -> RegressionComparisonResult:
    """Compare champion and challenger regression models on all gates.

    Gate 1 (R²): challenger R² must exceed ``gates.min_r2``.
    Gate 2 (Coverage low): challenger coverage must be ≥ ``gates.min_coverage``.
        Skipped if coverage is NaN.
    Gate 3 (Coverage high): challenger coverage must be ≤ ``gates.max_coverage``.
        Skipped if coverage is NaN.
    Gate 4 (MAE): challenger MAE must be lower than champion MAE.

    Args:
        champion: Metrics for the current champion.
        challenger: Metrics for the candidate.
        gates: Gate thresholds. Defaults to ``RegressionPromotionGates()``.

    Returns:
        ``RegressionComparisonResult`` with per-gate outcomes and verdict.
    """
    if gates is None:
        gates = RegressionPromotionGates()

    # Gate 1: R² > min_r2
    r2_pass: bool = challenger.r2 > gates.min_r2

    # Gate 2 & 3: Coverage within [min, max] - skip if NaN
    if isnan(challenger.coverage):
        coverage_low_pass: bool = True
        coverage_high_pass: bool = True
    else:
        coverage_low_pass = challenger.coverage >= gates.min_coverage
        coverage_high_pass = challenger.coverage <= gates.max_coverage

    # Gate 4: MAE must beat champion (lower is better)
    mae_pass: bool = challenger.mae < champion.mae

    gate_results: dict[str, bool] = {
        "r2_above_zero": r2_pass,
        "coverage_in_range_low": coverage_low_pass,
        "coverage_in_range_high": coverage_high_pass,
        "mae_improved": mae_pass,
    }

    should_promote: bool = all(gate_results.values())

    if should_promote:
        delta: float = champion.mae - challenger.mae
        reason: str = (
            f"Challenger ({challenger.model_type}) passes all gates (MAE improved by {delta:.2f})"
        )
    else:
        failed: list[str] = [k for k, v in gate_results.items() if not v]
        reason = (
            f"Challenger ({challenger.model_type}) failed "
            f"{len(failed)} gate(s): {', '.join(failed)}"
        )

    return RegressionComparisonResult(
        champion=champion,
        challenger=challenger,
        gates=gate_results,
        should_promote=should_promote,
        reason=reason,
    )


def select_prop_champion(
    results: list[RegressionModelResult],
    gates: RegressionPromotionGates | None = None,
) -> tuple[RegressionModelResult, str]:
    """Select the best prop model from a list of trained results.

    Logic:
    1. Filter to models passing all guardrails (R² > 0, coverage in range).
    2. Among eligible models, select the one with lowest MAE.
    3. If no model passes guardrails, return the ElasticNet result as
       fallback (Decision #11: known stable baseline).

    Args:
        results: List of regression model results to compare.
        gates: Gate thresholds. Defaults to ``RegressionPromotionGates()``.

    Returns:
        Tuple of (champion result, summary string).

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        msg = "Cannot select champion from empty results list."
        raise ValueError(msg)

    if gates is None:
        gates = RegressionPromotionGates()

    # Evaluate guardrails for each model
    eligible: list[RegressionModelResult] = []
    for r in results:
        r2_ok: bool = r.r2 > gates.min_r2

        if isnan(r.coverage):
            coverage_ok: bool = True
        else:
            coverage_ok = gates.min_coverage <= r.coverage <= gates.max_coverage

        if r2_ok and coverage_ok:
            eligible.append(r)

    # Build summary table
    lines: list[str] = []
    lines.append("")
    lines.append("  === Prop Champion Selection ===")
    lines.append("")
    lines.append(
        f"  {'Model Type':<16s}  {'MAE':>8s}  {'RMSE':>8s}"
        f"  {'R²':>8s}  {'Coverage':>10s}  {'Eligible'}"
    )
    lines.append(f"  {'─' * 16}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 10}")

    for r in results:
        is_eligible: bool = r in eligible
        cov_str: str = f"{r.coverage:.1%}" if not isnan(r.coverage) else "n/a"
        lines.append(
            f"  {r.model_type:<16s}  {r.mae:>8.2f}  {r.rmse:>8.2f}"
            f"  {r.r2:>8.3f}  {cov_str:>10s}  {'✅' if is_eligible else '❌'}"
        )

    if eligible:
        champion: RegressionModelResult = min(eligible, key=lambda r: r.mae)
        lines.append("")
        lines.append(f"  🏆 Champion: {champion.model_type} (MAE={champion.mae:.2f})")
    else:
        # Fallback to ElasticNet
        elasticnet_results: list[RegressionModelResult] = [
            r for r in results if r.model_type == "elasticnet"
        ]
        if elasticnet_results:
            champion = elasticnet_results[0]
        else:
            # If no ElasticNet either, just pick lowest MAE as last resort
            champion = min(results, key=lambda r: r.mae)
        lines.append("")
        lines.append(
            f"  ⚠️  No model passed all guardrails. "
            f"Fallback: {champion.model_type} (MAE={champion.mae:.2f})"
        )

    lines.append("")
    summary: str = "\n".join(lines)

    return champion, summary


def select_game_regression_champions(
    pairs: list[tuple[str, str]],
    *,
    repo: Path,
) -> dict[str, dict[str, Any]]:
    """Select the champion model_type for each regression game model_name.

    For each unique ``model_name`` in ``pairs`` whose trained artifacts
    have ``task == "regression"``, reads ``GameModelMetadata`` from
    :class:`ArtifactStore` for every trained ``model_type`` in the pair
    list, picks the one with the lowest holdout MAE, and returns a
    manifest entry.

    Ties on MAE are broken by preferring ``random_forest`` over
    ``xgboost`` (matches classification convention; see W13 Tier 2
    design decisions). Ties are not expected in practice.

    Args:
        pairs: List of ``(model_name, model_type)`` pairs to consider.
            Typically ``ctx["game_pairs"]`` from ``full-retrain``. Pairs
            whose artifacts are missing or whose task is not regression
            are silently skipped.
        repo: Repository root.

    Returns:
        Mapping of ``model_name`` → manifest entry. Each entry has
        ``model_type``, ``promoted_at`` (from ``trained_at``), and
        ``metrics`` (``mae``, ``rmse``, ``r2``). Empty dict if no
        regression pairs have trained artifacts.
    """
    from gridiron_edge.models.artifact import ArtifactStore

    store = ArtifactStore(repo)

    # Group by model_name. Preserve pair-list order for deterministic
    # tie-breaking when MAEs coincide.
    by_model_name: dict[str, list[str]] = {}
    for model_name, model_type in pairs:
        by_model_name.setdefault(model_name, []).append(model_type)

    entries: dict[str, dict[str, Any]] = {}

    for model_name, model_types in by_model_name.items():
        candidates: list[tuple[str, float, dict[str, float], str]] = []
        for model_type in model_types:
            if not store.is_trained(model_name, model_type):
                continue
            meta = store.read_metadata(model_name, model_type)
            if meta.task != "regression":
                continue
            mae = meta.metrics.get("mae")
            if mae is None:
                continue
            candidates.append((model_type, mae, dict(meta.metrics), meta.trained_at))

        if not candidates:
            continue

        winner = _pick_regression_winner(candidates)
        model_type, _mae, metrics, trained_at = winner
        entries[model_name] = {
            "model_type": model_type,
            "promoted_at": trained_at,
            "metrics": {key: metrics[key] for key in ("mae", "rmse", "r2") if key in metrics},
        }

    return entries


def select_game_classification_champions(
    pairs: list[tuple[str, str]],
    *,
    repo: Path,
) -> dict[str, dict[str, Any]]:
    """Select the champion model_type for each classification game model_name.

    For each unique ``model_name`` in ``pairs`` whose trained artifacts
    have ``task == "classification"``, ranks the trained ``model_types``
    via :func:`evaluation.select.collect_model_metrics` +
    :func:`evaluation.select.rank_models` on Brier / ECE / AUC, and
    returns the top-ranked entry as a manifest entry.

    Reuses the composite-key ranking already used by
    ``gridiron evaluate select-model`` so the two surfaces agree
    exactly on which model wins.

    Args:
        pairs: List of ``(model_name, model_type)`` pairs to consider.
            Typically ``ctx["game_pairs"]`` from ``full-retrain``. Pairs
            whose artifacts are missing, whose task is not classification,
            or whose model_name produces no archive rows are silently
            skipped.
        repo: Repository root.

    Returns:
        Mapping of ``model_name`` → manifest entry. Each entry has
        ``model_type``, ``promoted_at`` (from artifact metadata's
        ``trained_at``), and ``metrics`` (``brier``, ``ece``, ``auc``).
        Empty dict if no classification pairs have rankable archive rows.
    """
    from gridiron_edge.evaluation.select import (
        collect_model_metrics,
        rank_models,
    )
    from gridiron_edge.models.artifact import ArtifactStore

    store = ArtifactStore(repo)

    # Group by model_name, preserving pair order for downstream determinism.
    by_model_name: dict[str, list[str]] = {}
    for model_name, model_type in pairs:
        by_model_name.setdefault(model_name, []).append(model_type)

    entries: dict[str, dict[str, Any]] = {}

    for model_name, model_types in by_model_name.items():
        # Filter to trained classification artifacts only.
        eligible_types: list[str] = []
        for model_type in model_types:
            if not store.is_trained(model_name, model_type):
                continue
            meta = store.read_metadata(model_name, model_type)
            if meta.task != "classification":
                continue
            eligible_types.append(model_type)

        if not eligible_types:
            continue

        # Build composite keys for select.py's API.
        composite_keys: list[str] = [f"{model_name}_{model_type}" for model_type in eligible_types]

        rows = collect_model_metrics(composite_keys, repo=repo)
        if not rows:
            # No archive rows for any of these pairs — nothing to rank.
            continue

        ranked = rank_models(
            rows,
            criteria_list=["brier", "ece", "auc"],
            lower_is_better={"brier", "ece"},
        )
        if ranked.empty:
            continue

        winner_key: str = ranked.iloc[0]["model_key"]
        winner_type: str = _model_type_from_composite_key(winner_key, model_name)

        # Read metadata again for trained_at and defensive metrics narrowing.
        winner_meta = store.read_metadata(model_name, winner_type)
        entries[model_name] = {
            "model_type": winner_type,
            "promoted_at": winner_meta.trained_at,
            "metrics": {
                key: winner_meta.metrics[key]
                for key in ("brier", "ece", "auc")
                if key in winner_meta.metrics
            },
        }

    return entries


def select_prop_champion_for_family(
    family: str,
    *,
    repo: Path,
    season: int | None = None,
) -> dict[str, Any] | None:
    """Select the champion algorithm for a single prop stat family.

    Iterates the three prop algorithms (elasticnet, random_forest, xgboost),
    builds :class:`RegressionModelResult` for each from the prop archive,
    and calls :func:`select_prop_champion` to pick the winner using the
    R²/coverage/MAE gates.

    Args:
        family: Prop stat family name (e.g. ``"qb_pass_yards"``).
        repo: Repository root.
        season: Optional season filter passed to
            :func:`build_prop_evaluation_df`. ``None`` = all seasons.

    Returns:
        Manifest entry with ``model_type``, ``promoted_at``, and
        ``metrics`` (``mae``, ``rmse``, ``r2``, ``coverage``), or
        ``None`` if no algorithm has archive rows for this family
        (cold-start case; caller decides what to do with it).

    Notes:
        ``promoted_at`` is sourced from ``ArtifactStore.read_metadata`` for
        the winning algorithm. If the winning algorithm has no trained
        artifact (archive rows exist from a prior training run whose
        artifact was later discarded), ``promoted_at`` falls back to the
        current UTC timestamp — the champion decision itself is what's
        being persisted, and staleness of the underlying artifact is a
        separate concern tracked via ``source_run_id``.
    """
    from datetime import UTC, datetime

    from gridiron_edge.evaluation.prop_archive import build_prop_evaluation_df
    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.prop_prediction.base import PropModelType

    results: list[RegressionModelResult] = []
    for model_type in PropModelType:
        try:
            eval_df = build_prop_evaluation_df(
                model_name=family,
                model_type=model_type.value,
                season=season,
            )
        except KeyError:
            # Unregistered family — should not happen in normal use
            # since callers pass registered families, but silently skip
            # if it does.
            continue

        if eval_df.empty:
            continue

        report = evaluate_prop_model(
            model_name=family,
            actual=eval_df["actual"],
            predicted_mean=eval_df["predicted_mean"],
            predicted_std=eval_df.get("predicted_std"),
            lo_90=eval_df.get("lo_90"),
            hi_90=eval_df.get("hi_90"),
        )
        coverage: float = (
            report.coverage.actual_coverage if report.coverage is not None else float("nan")
        )
        results.append(
            RegressionModelResult(
                model_type=str(model_type),
                mae=report.accuracy.mae,
                rmse=report.accuracy.rmse,
                r2=report.accuracy.r2,
                coverage=coverage,
            )
        )

    if not results:
        return None

    champion, _summary = select_prop_champion(results)

    # Source promoted_at from the winning artifact's trained_at when available.
    store = ArtifactStore(repo)
    if store.is_trained(family, champion.model_type):
        promoted_at = store.read_metadata(family, champion.model_type).trained_at
    else:
        promoted_at = datetime.now(UTC).isoformat()

    return {
        "model_type": champion.model_type,
        "promoted_at": promoted_at,
        "metrics": {
            "mae": champion.mae,
            "rmse": champion.rmse,
            "r2": champion.r2,
            "coverage": champion.coverage,
        },
    }


def select_prop_champions_all_families(
    families: list[str],
    *,
    repo: Path,
) -> dict[str, dict[str, Any]]:
    """Select champions for every listed prop stat family.

    Thin iterator over :func:`select_prop_champion_for_family`. Families
    with no archive rows for any algorithm are silently skipped — the
    resulting mapping only contains entries for families where at least
    one algorithm produced predictions.

    Args:
        families: List of prop stat family names.
        repo: Repository root.

    Returns:
        Mapping of ``family`` → manifest entry.
    """
    entries: dict[str, dict[str, Any]] = {}
    for family in families:
        entry = select_prop_champion_for_family(family, repo=repo)
        if entry is not None:
            entries[family] = entry
    return entries


def promote_champions(
    *,
    game_pairs: list[tuple[str, str]],
    prop_families: list[str],
    repo: Path,
    source_run_id: str | None = None,
) -> PromoteChampionsResult:
    """Run all three selectors, merge with existing manifest, and persist.

    Reusable core logic for W13 Tier 2. Called by
    ``cli/full_retrain.py::_stage_promote_champions`` (from
    ``gridiron full-retrain``) and by CLI flags on ``evaluate select-model``
    and ``props champion`` (manual overrides).

    Subset semantics:
        - Fresh selector output for the given ``game_pairs`` and
          ``prop_families``.
        - Existing manifest entries for families outside the current
          subset are preserved verbatim (with their original
          ``source_run_id`` — see ``champion_resolver.write_manifest``).
        - Cold-start (no prior manifest): only fresh entries are written.

    Args:
        game_pairs: List of ``(model_name, model_type)`` pairs to rank
            on the game side.
        prop_families: List of prop stat family names to rank.
        repo: Repository root.
        source_run_id: Optional identifier for this run's writes.
            Defaults to a wall-clock timestamp.

    Returns:
        :class:`PromoteChampionsResult` describing the manifest write.
    """
    from datetime import datetime

    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        read_manifest,
        write_manifest,
    )

    classification_entries = select_game_classification_champions(
        game_pairs,
        repo=repo,
    )
    regression_entries = select_game_regression_champions(
        game_pairs,
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
    if game_pairs and not classification_entries and not regression_entries:
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

    preserved_entries: dict[str, dict[str, Any]] = {
        model_name: entry
        for model_name, entry in existing_models.items()
        if model_name not in fresh_entries
    }

    merged_entries: dict[str, dict[str, Any]] = {
        **preserved_entries,
        **fresh_entries,
    }

    resolved_run_id: str = source_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_path = write_manifest(
        merged_entries,
        source_run_id=resolved_run_id,
        repo=repo,
    )

    return PromoteChampionsResult(
        manifest_path=manifest_path,
        fresh_entries=fresh_entries,
        preserved_entries=preserved_entries,
        warnings=warnings,
    )


def _model_type_from_composite_key(composite_key: str, model_name: str) -> str:
    """Strip the ``{model_name}_`` prefix from a composite key.

    Mirrors :func:`evaluation.select._parse_composite_key` but is
    specialized for the case where we already know the model_name — the
    caller iterated over grouped pairs, so we don't need the
    known-model-name lookup.
    """
    prefix = f"{model_name}_"
    if not composite_key.startswith(prefix):
        msg = (
            f"Composite key {composite_key!r} does not start with expected "
            f"prefix {prefix!r}. This suggests rank_models returned a key "
            f"that was not passed to collect_model_metrics."
        )
        raise ValueError(msg)
    return composite_key[len(prefix) :]


def _pick_regression_winner(
    candidates: list[tuple[str, float, dict[str, float], str]],
) -> tuple[str, float, dict[str, float], str]:
    """Pick lowest-MAE candidate; tie-break by preferring random_forest.

    Candidates are (model_type, mae, metrics, trained_at) tuples.
    """
    min_mae = min(c[1] for c in candidates)
    tied = [c for c in candidates if c[1] == min_mae]
    if len(tied) == 1:
        return tied[0]
    # Tie-breaker: prefer random_forest, else preserve input order (already
    # deterministic because the caller passed pairs in a defined order).
    for c in tied:
        if c[0] == "random_forest":
            return c
    return tied[0]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_METRIC_LABELS: list[tuple[str, str, bool]] = [
    ("brier", "Brier", True),
    ("ece", "ECE", True),
    ("auc", "AUC", True),
    ("log_loss", "Log Loss", False),
    ("accuracy", "Accuracy", False),
]

_GATE_NAMES: dict[str, str] = {
    "brier": "brier_improved",
    "ece": "ece_acceptable",
    "auc": "auc_acceptable",
}


def format_classification_comparison(result: ClassificationComparisonResult) -> str:
    """Build a human-readable comparison report.

    Args:
        result: Output of ``compare_classification_models()``.

    Returns:
        Multi-line formatted string showing metric table and verdict.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("  === Champion vs Challenger ===")
    lines.append("")
    lines.append(
        f"  {'Metric':<14s}  {'Champion':>10s}  {'Challenger':>10s}  {'Delta':>10s}  {'Gate'}"
    )
    lines.append(f"  {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    for key, label, is_gated in _METRIC_LABELS:
        c_val: float = result.champion_metrics.get(key, float("nan"))
        r_val: float = result.challenger_metrics.get(key, float("nan"))

        if isnan(c_val) or isnan(r_val):
            c_str: str = "n/a" if isnan(c_val) else f"{c_val:.5f}"
            r_str: str = "n/a" if isnan(r_val) else f"{r_val:.5f}"
            d_str = "n/a"
            gate_str = "(skipped)" if is_gated else "(info only)"
        else:
            c_str = f"{c_val:.5f}"
            r_str = f"{r_val:.5f}"
            delta: float = r_val - c_val
            sign: Literal["", "+"] = "+" if delta >= 0 else ""
            d_str: str = f"{sign}{delta:.5f}"

            if is_gated:
                gate_key: str = _GATE_NAMES[key]
                passed: bool = result.gates[gate_key]
                gate_str = "✅" if passed else "❌"
            else:
                gate_str = "(info only)"

        lines.append(f"  {label:<14s}  {c_str:>10s}  {r_str:>10s}  {d_str:>10s}  {gate_str}")

    lines.append("")
    verdict: Literal["✅ PROMOTE", "❌ REJECT"] = (
        "✅ PROMOTE" if result.should_promote else "❌ REJECT"
    )
    lines.append(f"  VERDICT: {verdict} - {result.reason}")
    lines.append("")
    return "\n".join(lines)


def format_regression_comparison(result: RegressionComparisonResult) -> str:
    """Build a human-readable regression comparison report.

    Args:
        result: Output of ``compare_regression_models()``.

    Returns:
        Multi-line formatted string showing metric table and verdict.
    """
    c: RegressionModelResult = result.champion
    r: RegressionModelResult = result.challenger

    lines: list[str] = []
    lines.append("")
    lines.append(f"  === {c.model_type} (champion) vs {r.model_type} (challenger) ===")
    lines.append("")
    lines.append(
        f"  {'Metric':<14s}  {'Champion':>10s}  {'Challenger':>10s}  {'Delta':>10s}  {'Gate'}"
    )
    lines.append(f"  {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    # MAE (lower is better)
    mae_delta: float = r.mae - c.mae
    mae_gate: str = "✅" if result.gates["mae_improved"] else "❌"
    lines.append(f"  {'MAE':<14s}  {c.mae:>10.2f}  {r.mae:>10.2f}  {mae_delta:>+10.2f}  {mae_gate}")

    # RMSE (info only)
    rmse_delta: float = r.rmse - c.rmse
    lines.append(
        f"  {'RMSE':<14s}  {c.rmse:>10.2f}  {r.rmse:>10.2f}  {rmse_delta:>+10.2f}  (info only)"
    )

    # R² (higher is better)
    r2_delta: float = r.r2 - c.r2
    r2_gate: str = "✅" if result.gates["r2_above_zero"] else "❌"
    lines.append(f"  {'R²':<14s}  {c.r2:>10.3f}  {r.r2:>10.3f}  {r2_delta:>+10.3f}  {r2_gate}")

    # Coverage
    c_cov: str = f"{c.coverage:.1%}" if not isnan(c.coverage) else "n/a"
    r_cov: str = f"{r.coverage:.1%}" if not isnan(r.coverage) else "n/a"
    if isnan(c.coverage) or isnan(r.coverage):
        cov_delta: str = "n/a"
        cov_gate = "(skipped)"
    else:
        cov_d: float = r.coverage - c.coverage
        cov_delta = f"{cov_d:>+.1%}"
        low_ok: bool = result.gates["coverage_in_range_low"]
        high_ok: bool = result.gates["coverage_in_range_high"]
        cov_gate = "✅" if (low_ok and high_ok) else "❌"
    lines.append(f"  {'Coverage':<14s}  {c_cov:>10s}  {r_cov:>10s}  {cov_delta:>10s}  {cov_gate}")

    lines.append("")
    verdict: str = "✅ PROMOTE" if result.should_promote else "❌ REJECT"
    lines.append(f"  VERDICT: {verdict} - {result.reason}")
    lines.append("")

    return "\n".join(lines)
