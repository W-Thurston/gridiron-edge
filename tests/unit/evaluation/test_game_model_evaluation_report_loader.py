"""Tests for strict loading of stored game-model evaluation reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.evaluation.game_model_evaluation_report import (
    TOTAL_MODEL_TYPES,
    WIN_MODEL_TYPES,
    TotalFamilyMetrics,
    WinFamilyMetrics,
    create_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_loader import (
    load_current_game_model_evaluation_report,
    read_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_selection import (
    game_model_evaluation_report_path,
    select_current_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_store import (
    write_game_model_evaluation_report,
)


def _build_and_store(tmp_path: Path):
    win_evidence = pd.DataFrame({"game_id": ["g1"], "away_win_prob": [0.6]})
    total_evidence = pd.DataFrame({"game_id": ["g1"], "model_total": [44.0]})
    environment_slices = pd.DataFrame({"model_type": ["random_forest"], "slice_label": ["True"]})
    report = create_game_model_evaluation_report(
        generated_at=datetime(2026, 9, 25, tzinfo=UTC),
        game_ids=frozenset({"g1"}),
        win_metrics=tuple(
            WinFamilyMetrics(
                model_type=model_type,
                run_id=f"run-{model_type}",
                n_games=1,
                brier=0.2,
                log_loss=0.6,
                accuracy=0.6,
                auc=None,
                ece=0.02,
                calibration_slope=None,
                calibration_intercept=None,
                sharpness=0.03,
                season_stability=None,
            )
            for model_type in WIN_MODEL_TYPES
        ),
        total_metrics=tuple(
            TotalFamilyMetrics(
                model_type=model_type,
                run_id=f"run-{model_type}",
                n_games=1,
                mae=10.0,
                median_absolute_error=9.0,
                rmse=13.0,
                nominal_coverage=0.9,
                actual_coverage=None,
            )
            for model_type in TOTAL_MODEL_TYPES
        ),
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        win_evidence_artifact="schema=1/win_evidence/token.parquet",
        total_evidence_artifact="schema=1/total_evidence/token.parquet",
        environment_slices_artifact="schema=1/environment_slices/token.parquet",
    )
    write_game_model_evaluation_report(
        report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        repo=tmp_path,
    )
    return report


def test_read_report_round_trips_including_optional_none_fields(tmp_path: Path) -> None:
    report = _build_and_store(tmp_path)
    path = game_model_evaluation_report_path(report.report_id, repo=tmp_path)

    loaded = read_game_model_evaluation_report(path)

    assert loaded == report
    assert all(metric.auc is None for metric in loaded.win_metrics)
    assert all(metric.actual_coverage is None for metric in loaded.total_metrics)


def test_load_current_report_returns_verified_frames(tmp_path: Path) -> None:
    report = _build_and_store(tmp_path)
    select_current_game_model_evaluation_report(
        report.report_id,
        selected_at=datetime(2026, 9, 25, 1, tzinfo=UTC),
        repo=tmp_path,
    )

    current = load_current_game_model_evaluation_report(repo=tmp_path)

    assert current.report == report
    assert len(current.win_evidence) == 1
    assert len(current.total_evidence) == 1
    assert len(current.environment_slices) == 1


def test_path_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    report = _build_and_store(tmp_path)
    real_path = game_model_evaluation_report_path(report.report_id, repo=tmp_path)
    wrong_path = real_path.with_name(f"{'b' * 64}.json")
    wrong_path.write_text(real_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="path and embedded identity disagree"):
        read_game_model_evaluation_report(wrong_path)
