# tests/unit/cli/test_weekly_predict_publication.py
"""Failure-order tests for live weekly evidence publication."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from gridiron_edge.cli.weekly_predict import _stage_predict_week
from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
)

SEASON = "2026-2027"
WEEK = 1
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)
REVISION = SourceRevision(commit="a" * 40, tracked_worktree_clean=True)
SOURCES = (
    SourceArtifactReference(
        relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
        state=PredictionSourceState.PRESENT,
        content_digest="b" * 64,
        size_bytes=100,
    ),
)


def _context() -> dict[str, object]:
    return {"season": SEASON, "week": WEEK}


def _schedule() -> pd.DataFrame:
    return pd.DataFrame({"season": [SEASON], "week": [WEEK]})


def _execution() -> SimpleNamespace:
    evidence = SimpleNamespace(
        source_revision=REVISION,
        source_artifacts=SOURCES,
    )
    return SimpleNamespace(
        policy=MagicMock(),
        events=pd.DataFrame({"event_id": ["event-1"]}),
        input_evidence=(evidence,),
        win_display=pd.DataFrame({"GAME_ID": ["game-1"]}),
    )


def _run(
    order: list[str],
    *,
    revision_error: Exception | None = None,
    recapture_error: Exception | None = None,
    snapshot_error: Exception | None = None,
    evidence_error: Exception | None = None,
    event_error: Exception | None = None,
    context: dict[str, object] | None = None,
):
    def effect(name: str, result: object, error: Exception | None = None):
        def call(*_args: object, **_kwargs: object) -> object:
            order.append(name)
            if error is not None:
                raise error
            return result

        return call

    with (
        patch(
            "gridiron_edge.cli.weekly_predict.get_settings",
            return_value=SimpleNamespace(repo_root=Path("/repo")),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict.resolve_clean_source_revision",
            side_effect=effect("revision", REVISION, revision_error),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict.capture_prediction_source_artifacts",
            side_effect=effect("capture", SOURCES),
        ),
        patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming_rich",
            side_effect=effect("schedule", _schedule()),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict.new_forecast_run_id",
            side_effect=effect("run-id", "run-1"),
        ),
        patch("gridiron_edge.cli.weekly_predict.datetime") as clock,
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution."
            "execute_weekly_prediction_policy",
            side_effect=effect("execute", _execution()),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict.recapture_and_require_same_prediction_sources",
            side_effect=effect("recapture", SOURCES, recapture_error),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict._publish_binary_snapshots",
            side_effect=effect("snapshots", (Path("/repo/snapshot.bin"),), snapshot_error),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict._publish_and_reload_evidence",
            side_effect=effect("evidence", (Path("/repo/evidence.json"),), evidence_error),
        ),
        patch(
            "gridiron_edge.cli.weekly_predict.write_forecast_events",
            side_effect=effect(
                "events",
                SimpleNamespace(path=Path("/repo/events.parquet")),
                event_error,
            ),
        ),
    ):
        clock.now.return_value = GENERATED_AT
        return _stage_predict_week(context or _context())


def test_publication_order_is_fail_closed() -> None:
    order: list[str] = []
    result = _run(order)

    assert result.success
    assert order == [
        "revision",
        "capture",
        "schedule",
        "run-id",
        "execute",
        "recapture",
        "snapshots",
        "evidence",
        "events",
    ]
    assert result.artifacts == [
        Path("/repo/snapshot.bin"),
        Path("/repo/evidence.json"),
        Path("/repo/events.parquet"),
    ]


def test_revision_failure_stops_before_schedule_and_execution() -> None:
    order: list[str] = []
    result = _run(order, revision_error=ValueError("dirty tracked worktree"))

    assert not result.success
    assert result.detail == "dirty tracked worktree"
    assert order == ["revision"]


def test_source_drift_writes_nothing() -> None:
    order: list[str] = []
    result = _run(order, recapture_error=ValueError("sources changed"))

    assert not result.success
    assert order[-1] == "recapture"
    assert "snapshots" not in order
    assert "evidence" not in order
    assert "events" not in order


def test_snapshot_failure_writes_no_evidence_or_events() -> None:
    order: list[str] = []
    result = _run(order, snapshot_error=OSError("snapshot failed"))

    assert not result.success
    assert order[-1] == "snapshots"
    assert "evidence" not in order
    assert "events" not in order


def test_evidence_failure_writes_no_events() -> None:
    order: list[str] = []
    result = _run(order, evidence_error=ValueError("strict reload failed"))

    assert not result.success
    assert order[-2:] == ["snapshots", "evidence"]
    assert "events" not in order


def test_event_failure_leaves_only_inert_publications() -> None:
    order: list[str] = []
    context = _context()
    result = _run(
        order,
        event_error=OSError("event persistence failed"),
        context=context,
    )

    assert not result.success
    assert order[-3:] == ["snapshots", "evidence", "events"]
    assert "prediction_policy" not in context
    assert "predictions_df" not in context
    assert "forecast_run_id" not in context
    assert "forecast_generated_at" not in context
    assert "prediction_input_evidence" not in context
