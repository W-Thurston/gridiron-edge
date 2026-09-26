# src/gridiron_edge/evaluation/game_model_evaluation_report.py

"""Immutable cross-family game-model evaluation report contracts.

Sibling to ``historical_backtest_report.py``, not an extension of it: that
report measures betting/ROI performance for the current champion Win/Total
pair; this report measures model-quality metrics (calibration, sharpness,
season stability, error, interval coverage, environment slices) for all six
game-model families on one verified-common game set. The two schemas do not
overlap and are persisted independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from pandas import DataFrame

GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION: Final[int] = 1

# The six game-model families this report always covers.
WIN_MODEL_TYPES: Final[tuple[str, ...]] = ("elo", "logistic", "random_forest", "xgboost")
TOTAL_MODEL_TYPES: Final[tuple[str, ...]] = ("random_forest", "xgboost")


@dataclass(frozen=True, slots=True)
class EvaluationFrameReference:
    """One content-addressed Parquet frame owned by a report."""

    artifact: str
    row_count: int
    columns: tuple[str, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class WinFamilyMetrics:
    """Model-quality metrics for one Win (classification) family."""

    model_type: str
    run_id: str
    n_games: int
    brier: float
    log_loss: float
    accuracy: float
    auc: float | None
    ece: float
    calibration_slope: float | None
    calibration_intercept: float | None
    sharpness: float
    season_stability: float | None


@dataclass(frozen=True, slots=True)
class TotalFamilyMetrics:
    """Model-quality metrics for one Total (regression) family."""

    model_type: str
    run_id: str
    n_games: int
    mae: float
    median_absolute_error: float
    rmse: float
    nominal_coverage: float
    actual_coverage: float | None


@dataclass(frozen=True, slots=True)
class GameModelEvaluationReport:
    """Frozen cross-family evaluation summary and exact row-level references."""

    schema_version: int
    report_id: str
    generated_at: datetime
    game_count: int
    game_set_digest: str
    win_metrics: tuple[WinFamilyMetrics, ...]
    total_metrics: tuple[TotalFamilyMetrics, ...]
    win_evidence: EvaluationFrameReference
    total_evidence: EvaluationFrameReference
    environment_slices: EvaluationFrameReference


def game_set_digest(game_ids: frozenset[str]) -> str:
    """Return a stable digest of a common game-ID set.

    Used to prove, by content, that every family in the report was evaluated
    against the exact same set of games.
    """
    canonical = json.dumps(sorted(game_ids), sort_keys=True, separators=(",", ":")).encode()
    return sha256(canonical).hexdigest()


def create_game_model_evaluation_report(
    *,
    generated_at: datetime,
    game_ids: frozenset[str],
    win_metrics: tuple[WinFamilyMetrics, ...],
    total_metrics: tuple[TotalFamilyMetrics, ...],
    win_evidence: DataFrame,
    total_evidence: DataFrame,
    environment_slices: DataFrame,
    win_evidence_artifact: str,
    total_evidence_artifact: str,
    environment_slices_artifact: str,
) -> GameModelEvaluationReport:
    """Create a deterministic report contract from validated report inputs."""
    _utc(generated_at)
    _require_families(win_metrics, expected=WIN_MODEL_TYPES, label="win_metrics")
    _require_families(total_metrics, expected=TOTAL_MODEL_TYPES, label="total_metrics")
    if not game_ids:
        raise ValueError("game_ids must not be empty.")

    win_ref = _frame_reference(win_evidence, win_evidence_artifact)
    total_ref = _frame_reference(total_evidence, total_evidence_artifact)
    slices_ref = _frame_reference(environment_slices, environment_slices_artifact)
    digest = game_set_digest(game_ids)

    core = _identity_payload(
        schema_version=GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION,
        generated_at=generated_at,
        game_count=len(game_ids),
        game_set_digest=digest,
        win_metrics=win_metrics,
        total_metrics=total_metrics,
        win_evidence=win_ref,
        total_evidence=total_ref,
        environment_slices=slices_ref,
    )
    report = GameModelEvaluationReport(
        schema_version=GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION,
        report_id=sha256(_canonical(core)).hexdigest(),
        generated_at=generated_at,
        game_count=len(game_ids),
        game_set_digest=digest,
        win_metrics=win_metrics,
        total_metrics=total_metrics,
        win_evidence=win_ref,
        total_evidence=total_ref,
        environment_slices=slices_ref,
    )
    validate_game_model_evaluation_report(report)
    return report


def validate_game_model_evaluation_report(report: GameModelEvaluationReport) -> None:
    """Validate report identity and internal cross-field invariants."""
    if report.schema_version != GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported game-model evaluation report schema version.")
    _digest(report.report_id, "report_id")
    _utc(report.generated_at)
    _digest(report.game_set_digest, "game_set_digest")
    if report.game_count <= 0:
        raise ValueError("game_count must be positive.")
    _require_families(report.win_metrics, expected=WIN_MODEL_TYPES, label="win_metrics")
    _require_families(report.total_metrics, expected=TOTAL_MODEL_TYPES, label="total_metrics")
    for reference in (report.win_evidence, report.total_evidence, report.environment_slices):
        _validate_frame_reference(reference)
    expected_id = sha256(
        _canonical(
            _identity_payload(
                schema_version=report.schema_version,
                generated_at=report.generated_at,
                game_count=report.game_count,
                game_set_digest=report.game_set_digest,
                win_metrics=report.win_metrics,
                total_metrics=report.total_metrics,
                win_evidence=report.win_evidence,
                total_evidence=report.total_evidence,
                environment_slices=report.environment_slices,
            )
        )
    ).hexdigest()
    if report.report_id != expected_id:
        raise ValueError("report_id does not match canonical report content.")


def frame_content_digest(frame: DataFrame) -> str:
    """Return a stable digest of canonical frame values and schema."""
    payload = frame.to_json(
        orient="table",
        date_format="iso",
        index=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _require_families(
    metrics: tuple[object, ...],
    *,
    expected: tuple[str, ...],
    label: str,
) -> None:
    actual = tuple(metric.model_type for metric in metrics)  # type: ignore[attr-defined]
    if actual != expected:
        raise ValueError(f"{label} must cover exactly {expected!r} in that order, got {actual!r}.")


def _frame_reference(
    frame: DataFrame,
    artifact: str,
) -> EvaluationFrameReference:
    reference = EvaluationFrameReference(
        artifact=artifact,
        row_count=len(frame),
        columns=tuple(frame.columns),
        content_digest=frame_content_digest(frame),
    )
    _validate_frame_reference(reference)
    return reference


def _validate_frame_reference(reference: EvaluationFrameReference) -> None:
    path = Path(reference.artifact)
    if not reference.artifact.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError("Report artifact must be a safe relative path.")
    if reference.row_count < 0:
        raise ValueError("Report frame row_count must be nonnegative.")
    if not reference.columns:
        raise ValueError("Report frame columns must not be empty.")
    _digest(reference.content_digest, "content_digest")


def _identity_payload(
    *,
    schema_version: int,
    generated_at: datetime,
    game_count: int,
    game_set_digest: str,
    win_metrics: tuple[WinFamilyMetrics, ...],
    total_metrics: tuple[TotalFamilyMetrics, ...],
    win_evidence: EvaluationFrameReference,
    total_evidence: EvaluationFrameReference,
    environment_slices: EvaluationFrameReference,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "generated_at": generated_at.isoformat(),
        "game_count": game_count,
        "game_set_digest": game_set_digest,
        "win_metrics": [asdict(metric) for metric in win_metrics],
        "total_metrics": [asdict(metric) for metric in total_metrics],
        "win_evidence": asdict(win_evidence),
        "total_evidence": asdict(total_evidence),
        "environment_slices": asdict(environment_slices),
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode()


def _utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC.")


def _digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value
