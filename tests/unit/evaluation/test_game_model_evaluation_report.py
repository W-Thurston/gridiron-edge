"""Tests for immutable game-model evaluation report contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from gridiron_edge.evaluation.game_model_evaluation_report import (
    TOTAL_MODEL_TYPES,
    WIN_MODEL_TYPES,
    TotalFamilyMetrics,
    WinFamilyMetrics,
    create_game_model_evaluation_report,
    game_set_digest,
    validate_game_model_evaluation_report,
)


def _win_metrics() -> tuple[WinFamilyMetrics, ...]:
    return tuple(
        WinFamilyMetrics(
            model_type=model_type,
            run_id=f"run-{model_type}",
            n_games=4,
            brier=0.22,
            log_loss=0.6,
            accuracy=0.6,
            auc=0.65,
            ece=0.02,
            calibration_slope=0.9,
            calibration_intercept=0.01,
            sharpness=0.03,
            season_stability=0.01,
        )
        for model_type in WIN_MODEL_TYPES
    )


def _total_metrics() -> tuple[TotalFamilyMetrics, ...]:
    return tuple(
        TotalFamilyMetrics(
            model_type=model_type,
            run_id=f"run-{model_type}",
            n_games=4,
            mae=10.0,
            median_absolute_error=9.0,
            rmse=13.0,
            nominal_coverage=0.9,
            actual_coverage=0.88,
        )
        for model_type in TOTAL_MODEL_TYPES
    )


def _frame(columns: list[str], rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame({column: list(range(rows)) for column in columns})


def _report_kwargs() -> dict[str, object]:
    return {
        "generated_at": datetime(2026, 9, 25, tzinfo=UTC),
        "game_ids": frozenset({"g1", "g2", "g3", "g4"}),
        "win_metrics": _win_metrics(),
        "total_metrics": _total_metrics(),
        "win_evidence": _frame(["game_id", "away_win_prob"], rows=16),
        "total_evidence": _frame(["game_id", "model_total"], rows=8),
        "environment_slices": _frame(["model_type", "slice_label"], rows=12),
        "win_evidence_artifact": "schema=1/win_evidence/token.parquet",
        "total_evidence_artifact": "schema=1/total_evidence/token.parquet",
        "environment_slices_artifact": "schema=1/environment_slices/token.parquet",
    }


class TestCreateAndValidate:
    def test_create_produces_valid_self_consistent_report(self) -> None:
        report = create_game_model_evaluation_report(**_report_kwargs())

        validate_game_model_evaluation_report(report)
        assert report.game_count == 4
        assert report.game_set_digest == game_set_digest(frozenset({"g1", "g2", "g3", "g4"}))

    def test_tampered_report_fails_validation(self) -> None:
        report = create_game_model_evaluation_report(**_report_kwargs())
        tampered = replace(report, game_count=report.game_count + 1)

        with pytest.raises(ValueError, match="does not match canonical report content"):
            validate_game_model_evaluation_report(tampered)

    def test_naive_generated_at_is_rejected(self) -> None:
        kwargs = _report_kwargs()
        kwargs["generated_at"] = datetime(2026, 9, 25)

        with pytest.raises(ValueError, match="timezone-aware UTC"):
            create_game_model_evaluation_report(**kwargs)

    def test_empty_game_ids_rejected(self) -> None:
        kwargs = _report_kwargs()
        kwargs["game_ids"] = frozenset()

        with pytest.raises(ValueError, match="game_ids must not be empty"):
            create_game_model_evaluation_report(**kwargs)

    def test_incomplete_win_family_coverage_rejected(self) -> None:
        kwargs = _report_kwargs()
        kwargs["win_metrics"] = _win_metrics()[:-1]

        with pytest.raises(ValueError, match="win_metrics must cover exactly"):
            create_game_model_evaluation_report(**kwargs)

    def test_wrong_family_order_rejected(self) -> None:
        kwargs = _report_kwargs()
        kwargs["total_metrics"] = tuple(reversed(_total_metrics()))

        with pytest.raises(ValueError, match="total_metrics must cover exactly"):
            create_game_model_evaluation_report(**kwargs)


class TestGameSetDigest:
    def test_digest_is_order_independent(self) -> None:
        forward = game_set_digest(frozenset({"g1", "g2", "g3"}))
        assert forward == game_set_digest(frozenset({"g3", "g1", "g2"}))

    def test_different_game_sets_have_different_digests(self) -> None:
        assert game_set_digest(frozenset({"g1"})) != game_set_digest(frozenset({"g2"}))
