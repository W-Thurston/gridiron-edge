"""Tests for explicit current selection of game-model evaluation reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gridiron_edge.evaluation.game_model_evaluation_report_selection import (
    game_model_evaluation_report_path,
    get_current_game_model_evaluation_report_selection,
    select_current_game_model_evaluation_report,
)

_REPORT_ID = "a" * 64


def _write_stub_report(tmp_path: Path, report_id: str = _REPORT_ID) -> Path:
    path = game_model_evaluation_report_path(report_id, repo=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def test_select_and_get_current_round_trip(tmp_path: Path) -> None:
    _write_stub_report(tmp_path)
    selected_at = datetime(2026, 9, 25, tzinfo=UTC)

    select_current_game_model_evaluation_report(
        _REPORT_ID,
        selected_at=selected_at,
        repo=tmp_path,
    )
    current = get_current_game_model_evaluation_report_selection(repo=tmp_path)

    assert current.report_id == _REPORT_ID
    assert current.selected_at == selected_at


def test_select_unknown_report_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not stored"):
        select_current_game_model_evaluation_report(
            _REPORT_ID,
            selected_at=datetime(2026, 9, 25, tzinfo=UTC),
            repo=tmp_path,
        )


def test_get_current_without_selection_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No current"):
        get_current_game_model_evaluation_report_selection(repo=tmp_path)


def test_naive_selected_at_is_rejected(tmp_path: Path) -> None:
    _write_stub_report(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        select_current_game_model_evaluation_report(
            _REPORT_ID,
            selected_at=datetime(2026, 9, 25),
            repo=tmp_path,
        )
