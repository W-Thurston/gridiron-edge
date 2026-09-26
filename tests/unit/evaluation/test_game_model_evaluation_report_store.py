"""Tests for immutable game-model evaluation report persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.evaluation.game_model_evaluation_report import (
    TOTAL_MODEL_TYPES,
    WIN_MODEL_TYPES,
    GameModelEvaluationReport,
    TotalFamilyMetrics,
    WinFamilyMetrics,
    create_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_store import (
    verify_game_model_evaluation_report,
    write_game_model_evaluation_report,
)


def _win_evidence() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["g1"], "away_win_prob": [0.6]})


def _total_evidence() -> pd.DataFrame:
    return pd.DataFrame({"game_id": ["g1"], "model_total": [44.0]})


def _environment_slices() -> pd.DataFrame:
    return pd.DataFrame({"model_type": ["random_forest"], "slice_label": ["True"]})


def _report() -> tuple[GameModelEvaluationReport, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    win_evidence = _win_evidence()
    total_evidence = _total_evidence()
    environment_slices = _environment_slices()
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
                auc=0.6,
                ece=0.02,
                calibration_slope=1.0,
                calibration_intercept=0.0,
                sharpness=0.03,
                season_stability=0.01,
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
                actual_coverage=0.9,
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
    return report, win_evidence, total_evidence, environment_slices


def test_exact_round_trip_and_replay(tmp_path: Path) -> None:
    report, win_evidence, total_evidence, environment_slices = _report()
    path = write_game_model_evaluation_report(
        report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        repo=tmp_path,
    )
    first = path.read_bytes()
    loaded_win, loaded_total, loaded_slices = verify_game_model_evaluation_report(
        report,
        repo=tmp_path,
    )

    assert (
        write_game_model_evaluation_report(
            report,
            win_evidence=win_evidence,
            total_evidence=total_evidence,
            environment_slices=environment_slices,
            repo=tmp_path,
        )
        == path
    )
    assert path.read_bytes() == first
    pd.testing.assert_frame_equal(loaded_win, win_evidence)
    pd.testing.assert_frame_equal(loaded_total, total_evidence)
    pd.testing.assert_frame_equal(loaded_slices, environment_slices)


def test_tampered_frame_is_rejected(tmp_path: Path) -> None:
    report, win_evidence, total_evidence, environment_slices = _report()
    write_game_model_evaluation_report(
        report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        repo=tmp_path,
    )
    win_evidence_path = (
        tmp_path / "data/output/game_model_evaluation" / report.win_evidence.artifact
    )
    changed = win_evidence.copy()
    changed.loc[0, "away_win_prob"] = 0.99
    changed.to_parquet(win_evidence_path, index=False)

    with pytest.raises(ValueError, match="digest"):
        verify_game_model_evaluation_report(report, repo=tmp_path)


def test_conflicting_frame_replay_is_rejected(tmp_path: Path) -> None:
    report, win_evidence, total_evidence, environment_slices = _report()
    write_game_model_evaluation_report(
        report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        repo=tmp_path,
    )
    changed = win_evidence.copy()
    changed.loc[0, "away_win_prob"] = 0.99

    with pytest.raises(ValueError, match="digest"):
        write_game_model_evaluation_report(
            report,
            win_evidence=changed,
            total_evidence=total_evidence,
            environment_slices=environment_slices,
            repo=tmp_path,
        )


def test_missing_artifact_raises(tmp_path: Path) -> None:
    report, *_ = _report()

    with pytest.raises(FileNotFoundError):
        verify_game_model_evaluation_report(report, repo=tmp_path)
