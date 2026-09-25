# tests/unit/cli/test_development_forecast.py

"""Tests for the generate-development-forecast command."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# pyrefly: ignore [missing-import]
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.development_forecast import (
    _generate_development_forecast,
    build_retained_history_schedule,
    generate_development_forecast_cmd,
)
from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
)

SEASON = "2026-2027"
WEEK = 2
GENERATED_AT = datetime(2026, 9, 20, 12, tzinfo=UTC)
REVISION = SourceRevision(commit="a" * 40, tracked_worktree_clean=True)
SOURCES = (
    SourceArtifactReference(
        relative_path="data/cleaned/NFL_wk_by_wk_cleaned.csv",
        state=PredictionSourceState.PRESENT,
        content_digest="b" * 64,
        size_bytes=100,
    ),
)

runner = CliRunner()


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["2026_02_DET_BUF", "2026_02_CAR_ATL"],
            "WEEK_NUM": [2, 2],
            "YEAR": ["2026-2027", "2026-2027"],
            "GAME_DAY_OF_WEEK": ["Sunday", "Sunday"],
            "GAME_DATE": ["2026-09-14", "2026-09-14"],
            "GAMETIME": ["13:00:00", "13:00:00"],
            "AWAY_TEAM": ["Detroit Lions", "Carolina Panthers"],
            "HOME_TEAM": ["Buffalo Bills", "Atlanta Falcons"],
            "AWAY_SCORE": [31, 34],
            "HOME_SCORE": [41, 3],
            "IS_NEUTRAL_SITE": [0, 0],
        }
    )


class TestBuildRetainedHistorySchedule:
    """Column-shape coverage for the retained-history schedule adapter."""

    def test_maps_required_columns(self) -> None:
        schedule = build_retained_history_schedule(_games())

        assert list(schedule.columns) == [
            "season",
            "week",
            "game_id",
            "game_day_of_week",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "neutral_site",
        ]
        assert schedule["season"].tolist() == ["2026-2027", "2026-2027"]
        assert schedule["week"].tolist() == [2, 2]
        assert schedule["game_id"].tolist() == ["2026_02_DET_BUF", "2026_02_CAR_ATL"]
        assert schedule["away_team"].tolist() == ["Detroit Lions", "Carolina Panthers"]
        assert schedule["home_team"].tolist() == ["Buffalo Bills", "Atlanta Falcons"]
        assert schedule["neutral_site"].tolist() == [0, 0]

    def test_does_not_leak_score_columns(self) -> None:
        schedule = build_retained_history_schedule(_games())

        assert "AWAY_SCORE" not in schedule.columns
        assert "HOME_SCORE" not in schedule.columns

    def test_missing_required_column_raises(self) -> None:
        games = _games().drop(columns=["IS_NEUTRAL_SITE"])

        with pytest.raises(ValueError, match="missing required columns"):
            build_retained_history_schedule(games)


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
            "gridiron_edge.cli.development_forecast.resolve_clean_source_revision",
            side_effect=effect("revision", REVISION, revision_error),
        ),
        patch(
            "gridiron_edge.cli.development_forecast.capture_prediction_source_artifacts",
            side_effect=effect("capture", SOURCES),
        ),
        patch(
            "gridiron_edge.cli.development_forecast.load_games",
            side_effect=effect("games", _games()),
        ),
        patch(
            "gridiron_edge.cli.development_forecast.new_forecast_run_id",
            side_effect=effect("run-id", "run-1"),
        ),
        patch("gridiron_edge.cli.development_forecast.datetime") as clock,
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution."
            "execute_development_weekly_prediction_policy",
            side_effect=effect("execute", _execution()),
        ),
        patch(
            "gridiron_edge.cli.development_forecast.recapture_and_require_same_prediction_sources",
            side_effect=effect("recapture", SOURCES, recapture_error),
        ),
        patch(
            "gridiron_edge.cli.development_forecast._publish_binary_snapshots",
            side_effect=effect("snapshots", (Path("/repo/snapshot.bin"),), snapshot_error),
        ),
        patch(
            "gridiron_edge.cli.development_forecast._publish_and_reload_evidence",
            side_effect=effect("evidence", (Path("/repo/evidence.json"),), evidence_error),
        ),
        patch(
            "gridiron_edge.cli.development_forecast.write_forecast_events",
            side_effect=effect(
                "events",
                SimpleNamespace(path=Path("/repo/events.parquet")),
                event_error,
            ),
        ),
    ):
        clock.now.return_value = GENERATED_AT
        return _generate_development_forecast(season=SEASON, week=WEEK, repo=Path("/repo"))


class TestGenerateDevelopmentForecastPublicationOrder:
    """Failure-order coverage mirroring the live weekly-predict publication path."""

    def test_publication_order_is_fail_closed(self) -> None:
        order: list[str] = []
        run_id, event_count, evidence_count, artifacts = _run(order)

        assert run_id == "run-1"
        assert event_count == 1
        assert evidence_count == 1
        assert artifacts == (
            Path("/repo/snapshot.bin"),
            Path("/repo/evidence.json"),
            Path("/repo/events.parquet"),
        )
        assert order == [
            "revision",
            "capture",
            "games",
            "run-id",
            "execute",
            "recapture",
            "snapshots",
            "evidence",
            "events",
        ]

    def test_revision_failure_stops_before_schedule_and_execution(self) -> None:
        order: list[str] = []
        with pytest.raises(ValueError, match="dirty tracked worktree"):
            _run(order, revision_error=ValueError("dirty tracked worktree"))

        assert order == ["revision"]

    def test_source_drift_writes_nothing(self) -> None:
        order: list[str] = []
        with pytest.raises(ValueError, match="sources changed"):
            _run(order, recapture_error=ValueError("sources changed"))

        assert order[-1] == "recapture"
        assert "snapshots" not in order
        assert "evidence" not in order
        assert "events" not in order

    def test_snapshot_failure_writes_no_evidence_or_events(self) -> None:
        order: list[str] = []
        with pytest.raises(OSError, match="snapshot failed"):
            _run(order, snapshot_error=OSError("snapshot failed"))

        assert order[-1] == "snapshots"
        assert "evidence" not in order
        assert "events" not in order

    def test_evidence_failure_writes_no_events(self) -> None:
        order: list[str] = []
        with pytest.raises(ValueError, match="strict reload failed"):
            _run(order, evidence_error=ValueError("strict reload failed"))

        assert order[-2:] == ["snapshots", "evidence"]
        assert "events" not in order


def _command_app() -> typer.Typer:
    """Create an isolated Typer app for command tests."""
    app = typer.Typer()
    app.command()(generate_development_forecast_cmd)
    return app


class TestGenerateDevelopmentForecastCommand:
    """CLI rendering and exit-code behavior."""

    @patch("gridiron_edge.cli.development_forecast._generate_development_forecast")
    def test_success_prints_run_id_and_exits_zero(
        self,
        mock_generate: MagicMock,
    ) -> None:
        mock_generate.return_value = (
            "run-1",
            2,
            2,
            (Path("/repo/evidence.json"),),
        )

        result = runner.invoke(
            _command_app(),
            ["--season", SEASON, "--week", str(WEEK)],
        )

        assert result.exit_code == 0
        assert "Run ID: run-1" in result.output
        mock_generate.assert_called_once()
        _, kwargs = mock_generate.call_args
        assert kwargs["season"] == SEASON
        assert kwargs["week"] == WEEK

    @patch("gridiron_edge.cli.development_forecast._generate_development_forecast")
    def test_failure_exits_nonzero(
        self,
        mock_generate: MagicMock,
    ) -> None:
        mock_generate.side_effect = ValueError("no games for that week")

        result = runner.invoke(
            _command_app(),
            ["--season", SEASON, "--week", str(WEEK)],
        )

        assert result.exit_code != 0
