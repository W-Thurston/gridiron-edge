# src/gridiron_edge/evaluation/game_model_evaluation_report_store.py

"""Immutable JSON plus Parquet storage for game-model evaluation reports."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.game_model_evaluation_report import (
    EvaluationFrameReference,
    GameModelEvaluationReport,
    frame_content_digest,
    validate_game_model_evaluation_report,
)

GAME_MODEL_EVALUATION_REPORT_STORE_SCHEMA_VERSION = 1


def game_model_evaluation_report_root(repo: Path | None = None) -> Path:
    """Return the canonical report store root."""
    return (repo or get_settings().repo_root) / "data/output/game_model_evaluation"


def write_game_model_evaluation_report(
    report: GameModelEvaluationReport,
    *,
    win_evidence: DataFrame,
    total_evidence: DataFrame,
    environment_slices: DataFrame,
    repo: Path | None = None,
) -> Path:
    """Persist exact report frames and manifest or accept an exact replay."""
    validate_game_model_evaluation_report(report)
    root = game_model_evaluation_report_root(repo)
    _validate_frame(report.win_evidence, win_evidence, label="win_evidence")
    _validate_frame(report.total_evidence, total_evidence, label="total_evidence")
    _validate_frame(report.environment_slices, environment_slices, label="environment_slices")
    win_evidence_path = _resolved(root, report.win_evidence.artifact)
    total_evidence_path = _resolved(root, report.total_evidence.artifact)
    slices_path = _resolved(root, report.environment_slices.artifact)
    manifest = root / f"schema={report.schema_version}" / "reports" / f"{report.report_id}.json"
    _write_parquet(win_evidence_path, win_evidence)
    _write_parquet(total_evidence_path, total_evidence)
    _write_parquet(slices_path, environment_slices)
    encoded = (
        json.dumps(
            {
                "store_schema_version": (GAME_MODEL_EVALUATION_REPORT_STORE_SCHEMA_VERSION),
                "report_id": report.report_id,
                "report": asdict(report),
            },
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if manifest.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                "Game-model evaluation report identity cannot be reused with different content."
            )
        return manifest
    temporary = manifest.with_name(f".{manifest.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        if manifest.exists():
            if manifest.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    "Game-model evaluation report identity cannot be reused with different content."
                )
        else:
            temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def verify_game_model_evaluation_report(
    report: GameModelEvaluationReport,
    *,
    repo: Path | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Load and verify all three exact frames without recomputing analytics."""
    validate_game_model_evaluation_report(report)
    root = game_model_evaluation_report_root(repo)
    win_evidence_path = _resolved(root, report.win_evidence.artifact)
    total_evidence_path = _resolved(root, report.total_evidence.artifact)
    slices_path = _resolved(root, report.environment_slices.artifact)
    for label, path in (
        ("win evidence", win_evidence_path),
        ("total evidence", total_evidence_path),
        ("environment slices", slices_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Game-model evaluation {label} artifact is missing: {path}")
    win_evidence = pd.read_parquet(win_evidence_path)
    total_evidence = pd.read_parquet(total_evidence_path)
    environment_slices = pd.read_parquet(slices_path)
    _validate_frame(report.win_evidence, win_evidence, label="win_evidence")
    _validate_frame(report.total_evidence, total_evidence, label="total_evidence")
    _validate_frame(report.environment_slices, environment_slices, label="environment_slices")
    return win_evidence, total_evidence, environment_slices


def _validate_frame(
    reference: EvaluationFrameReference,
    frame: DataFrame,
    *,
    label: str,
) -> None:
    if len(frame) != reference.row_count:
        raise ValueError(f"{label} row count does not match report.")
    if tuple(frame.columns) != reference.columns:
        raise ValueError(f"{label} columns do not match report.")
    if frame_content_digest(frame) != reference.content_digest:
        raise ValueError(f"{label} content digest does not match report.")


def _resolved(root: Path, artifact: str) -> Path:
    resolved_root = root.resolve()
    path = (root / artifact).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Report artifact escapes storage root.") from exc
    return path


def _write_parquet(path: Path, frame: DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(frame):
            raise ValueError("Report frame identity cannot be reused with different content.")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        normalized = pd.read_parquet(temporary)
        if not normalized.equals(frame):
            raise ValueError("Serialized report frame does not replay exactly.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
