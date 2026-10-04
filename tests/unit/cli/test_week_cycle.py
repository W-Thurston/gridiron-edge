# tests/unit/cli/test_week_cycle.py

"""Tests for the weekly cycle command (``DECISIONS.md`` D81)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli import week_cycle
from gridiron_edge.cli._composites import CompositeStage, StageResult
from gridiron_edge.cli.main import app
from gridiron_edge.cli.week_cycle import (
    build_close_stages,
    build_forecast_stages,
    collector_lines,
    dirty_tracked_files,
    injury_report_present,
    starter_lines,
    week_completion,
)

pytestmark = pytest.mark.unit

runner = CliRunner()
NOW = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)


def _use_repo(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    settings = SimpleNamespace(repo_root=repo)
    monkeypatch.setattr("gridiron_edge.cli.week_cycle.get_settings", lambda: settings)


def _games(played: int, scheduled: int = 3, *, week: int = 4) -> DataFrame:
    rows = [
        {
            "season": 2026,
            "week": week,
            "game_type": "REG",
            "result": 3.0 if index < played else None,
        }
        for index in range(scheduled)
    ]
    rows.append({"season": 2026, "week": week + 1, "game_type": "REG", "result": None})
    rows.append({"season": 2026, "week": week, "game_type": "POST", "result": None})
    return DataFrame(rows)


def test_week_completion_counts_regular_season_games_of_the_week_only() -> None:
    assert week_completion(_games(3), season_int=2026, week=4) == (3, 3)
    assert week_completion(_games(2), season_int=2026, week=4) == (3, 2)
    assert week_completion(_games(0), season_int=2026, week=9) == (0, 0)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"week": 5, "game_type": "REG"}], True),
        ([{"week": 4, "game_type": "REG"}], False),
        ([{"week": 5, "game_type": "POST"}], False),
        ([], False),
    ],
)
def test_injury_report_is_present_only_for_regular_season_reports_of_the_week(
    rows: list[dict[str, object]], expected: bool
) -> None:
    frame = DataFrame(rows, columns=["week", "game_type"])

    assert injury_report_present(frame, week=5) is expected
    assert injury_report_present(None, week=5) is False


def test_dirty_tracked_files_reads_git_status(tmp_path: Path) -> None:
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(command, cwd=tmp_path, check=True)
    tracked = tmp_path / "a.txt"
    tracked.write_text("one")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    assert dirty_tracked_files(tmp_path) == []
    (tmp_path / "untracked.txt").write_text("x")
    assert dirty_tracked_files(tmp_path) == []
    tracked.write_text("two")
    assert dirty_tracked_files(tmp_path) == ["a.txt"]


def _receipt(repo: Path, name: str, **fields: object) -> None:
    directory = repo / "data" / "odds" / "collection_rollovers"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(fields))


def test_collector_lines_report_a_rollover_and_the_sync_age(tmp_path: Path) -> None:
    _receipt(
        tmp_path,
        "20261006T000000_rolled.json",
        outcome="rolled",
        target="2026-2027 week 5",
        decided_at="2026-10-06T00:20:00+00:00",
    )
    status = tmp_path / "data" / "odds" / "sync_status.json"
    status.write_text(json.dumps({"last_success_at": "2026-10-06T13:48:00+00:00", "error": None}))

    lines = collector_lines(tmp_path, season="2026-2027", next_week=5, now=NOW)

    assert lines[0] == "collector rolled to 2026-2027 week 5 at 2026-10-06T00:20:00+00:00"
    assert lines[1] == "collector sync last succeeded 12 min ago"


def test_collector_lines_say_when_no_rollover_receipt_has_arrived(tmp_path: Path) -> None:
    _receipt(tmp_path, "r.json", outcome="not_due", target="2026-2027 week 5")
    _receipt(tmp_path, "s.json", outcome="rolled", target="2026-2027 week 4")

    lines = collector_lines(tmp_path, season="2026-2027", next_week=5, now=NOW)

    assert lines[0].startswith("no rollover receipt for 2026-2027 week 5")
    assert lines[1] == "no collector sync status recorded"


def test_collector_lines_surface_a_failed_sync(tmp_path: Path) -> None:
    status = tmp_path / "data" / "odds" / "sync_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"last_success_at": "2026-10-05T14:00:00+00:00", "error": "boom"}))

    lines = collector_lines(tmp_path, season="2026-2027", next_week=5, now=NOW)

    assert "collector sync last succeeded 24.0 h ago" in lines
    assert "last sync attempt failed: boom" in lines


def _recorded_product() -> DataFrame:
    return DataFrame(
        {
            "game_id": ["g1", "g2"],
            "starting_qb_status": ["available", "starter_unresolved"],
            "away_qb_source": ["depth_chart_report", "carried_forward"],
            "home_qb_source": ["nflverse_schedule", pd.NA],
            "away_qb_name": ["Ann", pd.NA],
            "home_qb_name": ["Bob", pd.NA],
            "away_qb_id": ["a", "c"],
            "home_qb_id": ["b", pd.NA],
            "away_qb_differs_from_carry_forward": pd.array([True, False], dtype="boolean"),
            "home_qb_differs_from_carry_forward": pd.array([False, pd.NA], dtype="boolean"),
        }
    )


def test_starter_lines_summarize_sources_and_list_differences() -> None:
    lines = starter_lines(_recorded_product())

    assert lines[0] == (
        "starter sources: carried_forward 1, depth_chart_report 1, nflverse_schedule 1; "
        "1 game(s) not fully resolved"
    )
    assert lines[1:] == [
        "  g1 away: Ann (depth_chart_report) differs from the carried-forward starter"
    ]


def test_starter_lines_for_a_product_without_the_record() -> None:
    assert starter_lines(DataFrame({"game_id": ["g1"]})) == [
        "starting QBs were not recorded for this product"
    ]


def test_close_stages_gate_the_closeout_on_a_complete_week() -> None:
    stages = {stage.name: stage for stage in build_close_stages()}

    assert [stage.name for stage in build_close_stages()] == [
        "refresh-results",
        "verify-week-complete",
        "refresh-next-week-state",
        "refresh-season-projections",
        "close-live-forecasts",
        "check-collector",
        "next-step",
    ]
    for name in ("refresh-next-week-state", "refresh-season-projections", "close-live-forecasts"):
        assert stages[name].depends_on == ("verify-week-complete",)
    assert stages["verify-week-complete"].depends_on == ("refresh-results",)
    assert stages["check-collector"].soft_fail is True


def test_forecast_stages_check_first_then_reuse_weekly_predict() -> None:
    names = [stage.name for stage in build_forecast_stages()]

    assert names[:2] == ["check-clean-worktree", "check-injury-report"]
    assert names[2:-1] == [
        "ensure-data-fresh",
        "predict-week",
        "compose-weekly-product",
        "verify-weekly-readiness",
        "render-outputs",
        "generate-edges",
    ]
    assert names[-1] == "report-forecast"
    stages = {stage.name: stage for stage in build_forecast_stages()}
    assert stages["ensure-data-fresh"].depends_on == (
        "check-clean-worktree",
        "check-injury-report",
    )
    assert stages["report-forecast"].soft_fail is True


def test_the_complete_check_refuses_a_partial_week_unless_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    _games(2).to_parquet(raw / "NFL_wk_by_wk_nflverse.parquet")
    _use_repo(monkeypatch, tmp_path)
    context = {"season_int": 2026, "week": 4}

    refused = week_cycle._stage_verify_week_complete(context)
    allowed = week_cycle._stage_verify_week_complete({**context, "allow_incomplete": True})

    assert refused.success is False
    assert "2 of 3 games have results" in refused.detail
    assert "--allow-incomplete" in refused.detail
    assert allowed.success is True
    assert allowed.warnings and "incomplete" in allowed.warnings[0]


def test_the_complete_check_passes_a_finished_week_and_rejects_an_empty_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    _games(3).to_parquet(raw / "NFL_wk_by_wk_nflverse.parquet")
    _use_repo(monkeypatch, tmp_path)

    assert week_cycle._stage_verify_week_complete({"season_int": 2026, "week": 4}).success
    assert not week_cycle._stage_verify_week_complete({"season_int": 2026, "week": 12}).success


def test_the_injury_check_requires_the_report_unless_early_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    present = DataFrame({"week": [5], "game_type": ["REG"]})
    absent = DataFrame({"week": [4], "game_type": ["REG"]})
    module = "gridiron_edge.ingest.nflverse.weekly_availability"
    context = {"season_int": 2026, "week": 5}

    with (
        patch(f"{module}.fetch_injuries") as fetch,
        patch(f"{module}.load_injuries", return_value={2026: present}),
    ):
        ok = week_cycle._stage_check_injury_report(context)
    with (
        patch(f"{module}.fetch_injuries"),
        patch(f"{module}.load_injuries", return_value={2026: absent}),
    ):
        blocked = week_cycle._stage_check_injury_report(context)
        early = week_cycle._stage_check_injury_report({**context, "allow_early": True})

    fetch.assert_called_once()
    assert ok.success
    assert blocked.success is False
    assert "--allow-early" in blocked.detail
    assert early.success is True
    assert early.warnings and "early picture" in early.warnings[0]


def test_the_worktree_check_names_dirty_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    with patch("gridiron_edge.cli.week_cycle.dirty_tracked_files", return_value=["a.py", "b.py"]):
        result = week_cycle._stage_check_clean_worktree({})
    with patch("gridiron_edge.cli.week_cycle.dirty_tracked_files", return_value=[]):
        clean = week_cycle._stage_check_clean_worktree({})

    assert result.success is False
    assert "uncommitted: a.py, b.py" in result.detail
    assert clean.success is True


def _stub(name: str, *, ok: bool = True, report: str | None = None):
    def run(ctx: dict) -> StageResult:
        if report:
            ctx.setdefault("cycle_report", []).append(report)
        return StageResult(success=ok, detail=name)

    return run


def test_close_runs_end_to_end_with_stubbed_stages_and_prints_the_hint() -> None:
    stages = [
        CompositeStage(name="refresh-results", description="r", func=_stub("r")),
        CompositeStage(
            name="next-step",
            description="n",
            func=_stub("n", report="Friday: uv run gridiron week-cycle forecast --week 5"),
        ),
    ]
    with patch("gridiron_edge.cli.week_cycle.build_close_stages", return_value=stages):
        result = runner.invoke(app, ["week-cycle", "close", "--season", "2026-2027", "--week", "4"])

    assert result.exit_code == 0, result.output
    assert "Friday: uv run gridiron week-cycle forecast --week 5" in result.output


def test_close_exits_nonzero_when_a_stage_fails() -> None:
    stages = [CompositeStage(name="refresh-results", description="r", func=_stub("r", ok=False))]
    with patch("gridiron_edge.cli.week_cycle.build_close_stages", return_value=stages):
        result = runner.invoke(app, ["week-cycle", "close", "--season", "2026-2027", "--week", "4"])

    assert result.exit_code == 1


def test_forecast_passes_flags_into_the_context_and_prints_the_summary() -> None:
    seen: dict[str, object] = {}

    def capture(ctx: dict) -> StageResult:
        seen.update(allow_early=ctx["allow_early"], bankroll=ctx["bankroll"], week=ctx["week"])
        ctx.setdefault("cycle_report", []).append("prediction ready: yes")
        return StageResult(success=True, detail="ok")

    stages = [CompositeStage(name="check-injury-report", description="i", func=capture)]
    with patch("gridiron_edge.cli.week_cycle.build_forecast_stages", return_value=stages):
        result = runner.invoke(
            app,
            [
                "week-cycle",
                "forecast",
                "--season",
                "2026-2027",
                "--week",
                "5",
                "--allow-early",
                "--bankroll",
                "1000",
            ],
        )

    assert result.exit_code == 0, result.output
    assert seen == {"allow_early": True, "bankroll": 1000.0, "week": 5}
    assert "prediction ready: yes" in result.output


def test_forecast_supports_only_and_rejects_a_bad_season() -> None:
    stages = [
        CompositeStage(name="check-injury-report", description="i", func=_stub("i")),
        CompositeStage(name="check-clean-worktree", description="w", func=_stub("w", ok=False)),
    ]
    with patch("gridiron_edge.cli.week_cycle.build_forecast_stages", return_value=stages):
        only = runner.invoke(
            app,
            [
                "week-cycle",
                "forecast",
                "--season",
                "2026-2027",
                "--week",
                "5",
                "--only",
                "check-injury-report",
            ],
        )
    bad = runner.invoke(app, ["week-cycle", "close", "--season", "nonsense", "--week", "4"])

    assert only.exit_code == 0, only.output
    assert bad.exit_code == 2
