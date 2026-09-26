# src/gridiron_edge/evaluation/game_model_evaluation_report_loader.py

"""Strict loading for stored and explicitly selected game-model evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import cast

from pandas import DataFrame

from gridiron_edge.evaluation.game_model_evaluation_report import (
    GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION,
    TOTAL_MODEL_TYPES,
    WIN_MODEL_TYPES,
    EvaluationFrameReference,
    GameModelEvaluationReport,
    TotalFamilyMetrics,
    WinFamilyMetrics,
    validate_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_selection import (
    game_model_evaluation_report_path,
    get_current_game_model_evaluation_report_selection,
)
from gridiron_edge.evaluation.game_model_evaluation_report_store import (
    GAME_MODEL_EVALUATION_REPORT_STORE_SCHEMA_VERSION,
    verify_game_model_evaluation_report,
)


@dataclass(frozen=True, slots=True)
class CurrentGameModelEvaluationReport:
    """Explicitly selected report and its verified persisted frames."""

    report: GameModelEvaluationReport
    win_evidence: DataFrame
    total_evidence: DataFrame
    environment_slices: DataFrame
    selected_at: datetime


def read_game_model_evaluation_report(path: Path) -> GameModelEvaluationReport:
    """Strictly deserialize and validate one identity-addressed report manifest."""
    raw = _object(json.loads(path.read_text(encoding="utf-8")), "report artifact")
    _exact_keys(raw, {"store_schema_version", "report_id", "report"}, "Report artifact")
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != GAME_MODEL_EVALUATION_REPORT_STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported game-model evaluation report store schema version.")
    embedded_id = _digest(_text(raw["report_id"], "report_id"), "report_id")
    report = _report(raw["report"])
    if embedded_id != report.report_id:
        raise ValueError("Stored report identity does not match report content.")
    expected_path = game_model_evaluation_report_path(
        report.report_id,
        repo=_artifact_repo(path),
    )
    if path.resolve() != expected_path.resolve():
        raise ValueError("Game-model evaluation report path and embedded identity disagree.")
    return report


def load_current_game_model_evaluation_report(
    *,
    repo: Path | None = None,
) -> CurrentGameModelEvaluationReport:
    """Load the explicitly selected report and verify all three persisted frames."""
    selection = get_current_game_model_evaluation_report_selection(repo=repo)
    path = game_model_evaluation_report_path(selection.report_id, repo=repo)
    report = read_game_model_evaluation_report(path)
    win_evidence, total_evidence, environment_slices = verify_game_model_evaluation_report(
        report,
        repo=repo,
    )
    return CurrentGameModelEvaluationReport(
        report=report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        selected_at=selection.selected_at,
    )


def _report(value: object) -> GameModelEvaluationReport:
    data = _object(value, "report")
    _exact_keys(
        data,
        {
            "schema_version",
            "report_id",
            "generated_at",
            "game_count",
            "game_set_digest",
            "win_metrics",
            "total_metrics",
            "win_evidence",
            "total_evidence",
            "environment_slices",
        },
        "Report",
    )
    schema_version = _integer(data["schema_version"], "schema_version")
    if schema_version != GAME_MODEL_EVALUATION_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported game-model evaluation report schema version.")
    win_metrics = tuple(_win_metrics(item) for item in _list(data["win_metrics"], "win_metrics"))
    total_metrics = tuple(
        _total_metrics(item) for item in _list(data["total_metrics"], "total_metrics")
    )
    report = GameModelEvaluationReport(
        schema_version=schema_version,
        report_id=_digest(_text(data["report_id"], "report_id"), "report_id"),
        generated_at=_datetime(data["generated_at"], "generated_at"),
        game_count=_integer(data["game_count"], "game_count"),
        game_set_digest=_digest(
            _text(data["game_set_digest"], "game_set_digest"), "game_set_digest"
        ),
        win_metrics=win_metrics,
        total_metrics=total_metrics,
        win_evidence=_frame_reference(data["win_evidence"], "win_evidence"),
        total_evidence=_frame_reference(data["total_evidence"], "total_evidence"),
        environment_slices=_frame_reference(data["environment_slices"], "environment_slices"),
    )
    validate_game_model_evaluation_report(report)
    return report


def _win_metrics(value: object) -> WinFamilyMetrics:
    data = _object(value, "win_metrics entry")
    _exact_keys(
        data,
        {
            "model_type",
            "run_id",
            "n_games",
            "brier",
            "log_loss",
            "accuracy",
            "auc",
            "ece",
            "calibration_slope",
            "calibration_intercept",
            "sharpness",
            "season_stability",
        },
        "Win family metrics",
    )
    model_type = _text(data["model_type"], "win_metrics.model_type")
    if model_type not in WIN_MODEL_TYPES:
        raise ValueError(f"Unknown Win family model_type: {model_type!r}")
    return WinFamilyMetrics(
        model_type=model_type,
        run_id=_text(data["run_id"], "win_metrics.run_id"),
        n_games=_integer(data["n_games"], "win_metrics.n_games"),
        brier=_number(data["brier"], "brier"),
        log_loss=_number(data["log_loss"], "log_loss"),
        accuracy=_number(data["accuracy"], "accuracy"),
        auc=_optional_number(data["auc"], "auc"),
        ece=_number(data["ece"], "ece"),
        calibration_slope=_optional_number(data["calibration_slope"], "calibration_slope"),
        calibration_intercept=_optional_number(
            data["calibration_intercept"], "calibration_intercept"
        ),
        sharpness=_number(data["sharpness"], "sharpness"),
        season_stability=_optional_number(data["season_stability"], "season_stability"),
    )


def _total_metrics(value: object) -> TotalFamilyMetrics:
    data = _object(value, "total_metrics entry")
    _exact_keys(
        data,
        {
            "model_type",
            "run_id",
            "n_games",
            "mae",
            "median_absolute_error",
            "rmse",
            "nominal_coverage",
            "actual_coverage",
        },
        "Total family metrics",
    )
    model_type = _text(data["model_type"], "total_metrics.model_type")
    if model_type not in TOTAL_MODEL_TYPES:
        raise ValueError(f"Unknown Total family model_type: {model_type!r}")
    return TotalFamilyMetrics(
        model_type=model_type,
        run_id=_text(data["run_id"], "total_metrics.run_id"),
        n_games=_integer(data["n_games"], "total_metrics.n_games"),
        mae=_number(data["mae"], "mae"),
        median_absolute_error=_number(data["median_absolute_error"], "median_absolute_error"),
        rmse=_number(data["rmse"], "rmse"),
        nominal_coverage=_number(data["nominal_coverage"], "nominal_coverage"),
        actual_coverage=_optional_number(data["actual_coverage"], "actual_coverage"),
    )


def _frame_reference(value: object, label: str) -> EvaluationFrameReference:
    data = _object(value, f"{label} reference")
    _exact_keys(
        data,
        {"artifact", "row_count", "columns", "content_digest"},
        f"{label.title()} reference",
    )
    columns = _list(data["columns"], f"{label}.columns")
    return EvaluationFrameReference(
        artifact=_text(data["artifact"], f"{label}.artifact"),
        row_count=_integer(data["row_count"], f"{label}.row_count"),
        columns=tuple(_text(column, f"{label}.column") for column in columns),
        content_digest=_digest(
            _text(data["content_digest"], f"{label}.content_digest"),
            f"{label}.content_digest",
        ),
    )


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = ("data", "output", "game_model_evaluation")
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Game-model evaluation report path is outside the canonical store.")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _exact_keys(data: dict[str, object], expected: set[str], label: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{label} keys do not match the current schema.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label)


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp string.")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return result


def _digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value
