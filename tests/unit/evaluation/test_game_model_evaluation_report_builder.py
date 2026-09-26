"""Tests for cross-family game-model evaluation report build orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS
from gridiron_edge.evaluation.game_model_evaluation_report_builder import (
    build_and_write_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_store import (
    verify_game_model_evaluation_report,
)

_GAME_IDS: tuple[str, ...] = ("g1", "g2", "g3", "g4")

_WIN_PROBS: dict[str, dict[str, float]] = {
    "elo": {"g1": 0.70, "g2": 0.40, "g3": 0.55, "g4": 0.60},
    "logistic": {"g1": 0.68, "g2": 0.42, "g3": 0.52, "g4": 0.61},
    "random_forest": {"g1": 0.72, "g2": 0.38, "g3": 0.58, "g4": 0.59},
    "xgboost": {"g1": 0.69, "g2": 0.41, "g3": 0.56, "g4": 0.62},
}

_TOTALS: dict[str, dict[str, float]] = {
    "random_forest": {"g1": 44.0, "g2": 41.0, "g3": 47.0, "g4": 39.0},
    "xgboost": {"g1": 45.0, "g2": 40.0, "g3": 46.0, "g4": 38.0},
}

_WIN_RUN_IDS: dict[str, str] = {
    "elo": "run-elo",
    "logistic": "run-logistic",
    "random_forest": "run-rf-win",
    "xgboost": "run-xgb-win",
}

_TOTAL_RUN_IDS: dict[str, str] = {
    "random_forest": "run-rf-total",
    "xgboost": "run-xgb-total",
}


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_id in _GAME_IDS:
        common: dict[str, object] = {
            "role": "backfilled",
            "generated_at": datetime(2026, 8, 18, 20, tzinfo=UTC),
            "season": "2025-2026",
            "week": 1,
            "game_id": game_id,
            "game_date": "2025-09-01",
            "away_team": f"Away {game_id}",
            "home_team": f"Home {game_id}",
        }
        for model_type, probs in _WIN_PROBS.items():
            rows.append(
                common
                | {
                    "event_id": f"win-{model_type}-{game_id}",
                    "run_id": _WIN_RUN_IDS[model_type],
                    "model_name": "win_prob",
                    "model_type": model_type,
                    "away_win_prob": probs[game_id],
                    "model_total": None,
                }
            )
        for model_type, totals in _TOTALS.items():
            rows.append(
                common
                | {
                    "event_id": f"total-{model_type}-{game_id}",
                    "run_id": _TOTAL_RUN_IDS[model_type],
                    "model_name": "total",
                    "model_type": model_type,
                    "away_win_prob": None,
                    "model_total": totals[game_id],
                }
            )
    frame = pd.DataFrame(rows)
    for column in FORECAST_EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(FORECAST_EVENT_COLUMNS)]


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": list(_GAME_IDS),
            "AWAY_SCORE": [20, 17, 24, 14],
            "HOME_SCORE": [27, 21, 20, 24],
        }
    )


def _modeling_file() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": list(_GAME_IDS),
            "IS_DOME": [0, 1, 0, 1],
            "TEMP_F": [72.0, 70.0, 20.0, 70.0],
            "WIND_SPEED_MPH": [5.0, 0.0, 18.0, 0.0],
        }
    )


def _win_run_ids() -> dict[str, str]:
    return dict(_WIN_RUN_IDS)


def _total_run_ids() -> dict[str, str]:
    return dict(_TOTAL_RUN_IDS)


def test_builds_persists_and_verifies_exact_report(tmp_path: Path) -> None:
    events = _events()
    generated_at = datetime(2026, 8, 18, 21, tzinfo=UTC)

    with (
        patch("gridiron_edge.evaluation.metrics.load_forecast_events", return_value=events),
        patch("gridiron_edge.evaluation.metrics.loaders.load_games", return_value=_games()),
        patch(
            "gridiron_edge.evaluation.game_model_evaluation_report_builder.load_modeling_file",
            return_value=_modeling_file(),
        ),
    ):
        result = build_and_write_game_model_evaluation_report(
            win_run_ids=_win_run_ids(),
            total_run_ids=_total_run_ids(),
            generated_at=generated_at,
            repo=tmp_path,
        )

    assert result.manifest_path.is_file()
    assert result.win_evidence_row_count == 4 * 4
    assert result.total_evidence_row_count == 4 * 2
    assert result.environment_slice_row_count == 2 * 3 * 2  # 2 total families x 3 dims x 2 labels
    assert result.report.game_count == 4
    assert len(result.report.win_metrics) == 4
    assert len(result.report.total_metrics) == 2

    stored_win, stored_total, stored_slices = verify_game_model_evaluation_report(
        result.report,
        repo=tmp_path,
    )
    assert len(stored_win) == 16
    assert len(stored_total) == 8
    assert len(stored_slices) == 12


def test_mismatched_game_sets_fail_closed(tmp_path: Path) -> None:
    events = _events()
    # Drop one game from the Elo family only, breaking the common game set.
    mismatched = events.loc[
        ~((events["model_type"] == "elo") & (events["game_id"] == "g4"))
    ].reset_index(drop=True)

    with (
        patch("gridiron_edge.evaluation.metrics.load_forecast_events", return_value=mismatched),
        patch("gridiron_edge.evaluation.metrics.loaders.load_games", return_value=_games()),
        pytest.raises(ValueError, match="do not share a common game set"),
    ):
        build_and_write_game_model_evaluation_report(
            win_run_ids=_win_run_ids(),
            total_run_ids=_total_run_ids(),
            generated_at=datetime(2026, 8, 18, 21, tzinfo=UTC),
            repo=tmp_path,
        )


def test_rejects_non_utc_generation_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        build_and_write_game_model_evaluation_report(
            win_run_ids=_win_run_ids(),
            total_run_ids=_total_run_ids(),
            generated_at=datetime(2026, 8, 18, 21),
            repo=tmp_path,
        )


def test_incomplete_family_coverage_rejected(tmp_path: Path) -> None:
    win_run_ids = _win_run_ids()
    del win_run_ids["elo"]
    with pytest.raises(ValueError, match="win_run_ids must supply exactly"):
        build_and_write_game_model_evaluation_report(
            win_run_ids=win_run_ids,
            total_run_ids=_total_run_ids(),
            generated_at=datetime(2026, 8, 18, 21, tzinfo=UTC),
            repo=tmp_path,
        )
