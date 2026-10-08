# tests/unit/cli/test_week_cycle.py

"""Tests for the weekly cycle command (``DECISIONS.md`` D81, D82)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
from pandas import DataFrame
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli import post_week, week_cycle
from gridiron_edge.cli._composites import CompositeSummary, StageResult
from gridiron_edge.cli.main import app
from gridiron_edge.cli.week_cycle import (
    build_close_stages,
    build_forecast_stages,
    collector_lines,
    dirty_tracked_files,
    injury_report_present,
    starter_lines,
)
from gridiron_edge.weekly_cycle.decision import RecapSelectionState
from gridiron_edge.weekly_cycle.receipts import CycleReceipt, write_receipt
from gridiron_edge.weekly_cycle.runner import CyclePorts, StageRun

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


def test_dirty_tracked_files_reads_git_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A git hook exports GIT_DIR and GIT_INDEX_FILE; inherited, they would point these
    # commands at the enclosing repository instead of the temporary one.
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
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
        "verify-week-data-current",
        "refresh-season-projections",
        "close-live-forecasts",
        "build-weekly-recap",
        "check-collector",
    ]
    assert stages["refresh-next-week-state"].depends_on == ("verify-week-complete",)
    assert stages["verify-week-data-current"].depends_on == ("refresh-next-week-state",)
    for name in ("refresh-season-projections", "close-live-forecasts", "build-weekly-recap"):
        assert stages[name].depends_on == ("verify-week-data-current",)
    assert stages["verify-week-complete"].depends_on == ("refresh-results",)
    assert stages["build-weekly-recap"].soft_fail is True
    assert stages["check-collector"].soft_fail is True


def test_a_receipt_detail_names_the_soft_failed_stages() -> None:
    clean = CompositeSummary(name="close", succeeded=["close-live-forecasts"])
    soft = CompositeSummary(
        name="close",
        succeeded=["close-live-forecasts"],
        soft_failed=["build-weekly-recap", "check-collector"],
        warnings=["build-weekly-recap: recap refused: no weekly product is selected."],
    )
    hard = CompositeSummary(
        name="close",
        soft_failed=["build-weekly-recap"],
        failed=["close-live-forecasts"],
        failure_details={"close-live-forecasts": "incomplete closeout"},
    )

    assert week_cycle._stage_run(clean, {}, done="week 4 closed").detail == "week 4 closed"
    soft_run = week_cycle._stage_run(soft, {"cycle_report": ["x"]}, done="week 4 closed")
    assert soft_run.success is True
    assert soft_run.detail == "week 4 closed; soft-failed: build-weekly-recap, check-collector"
    assert soft_run.stages["build-weekly-recap"] == "soft-failed"
    assert soft_run.warnings == soft.warnings
    assert soft_run.report == ["x"]
    hard_run = week_cycle._stage_run(hard, {}, done="week 4 closed")
    assert hard_run.detail == "close-live-forecasts: incomplete closeout"


def test_forecast_stages_reuse_weekly_predict_then_summarize() -> None:
    names = [stage.name for stage in build_forecast_stages()]

    assert names == [
        "ensure-data-fresh",
        "predict-week",
        "compose-weekly-product",
        "verify-weekly-readiness",
        "render-outputs",
        "generate-edges",
        "report-forecast",
    ]
    stages = {stage.name: stage for stage in build_forecast_stages()}
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


def test_the_timer_runs_the_cycle_hourly_and_catches_up_after_downtime() -> None:
    service = Path("deploy/systemd/gridiron-edge-week-cycle.service").read_text()
    timer = Path("deploy/systemd/gridiron-edge-week-cycle.timer").read_text()

    assert "gridiron week-cycle\n" in service
    assert "Type=oneshot" in service
    assert "OnCalendar=hourly" in timer
    assert "Persistent=true" in timer
    assert "Unit=gridiron-edge-week-cycle.service" in timer


def _ports_with(**overrides: object) -> CyclePorts:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("a dry run must not touch this effect")

    schedule = DataFrame(
        {
            "season": ["2026-2027"],
            "week": [5],
            "game_id": ["g"],
            "game_date": ["2099-10-11"],
            "game_time": ["13:00:00"],
        }
    )
    fields: dict[str, object] = {
        "clock": lambda: NOW,
        "load_schedule": lambda: schedule,
        "product_id": lambda season, week: None,
        "load_product": unavailable,
        "dirty_files": unavailable,
        "run_closeout": unavailable,
        "run_forecast": unavailable,
        "run_edges": unavailable,
        "refresh_availability": unavailable,
        "digests": unavailable,
        "observe": unavailable,
        "edge_inputs": unavailable,
        "recap_state": unavailable,
        "run_recap": unavailable,
    }
    fields.update(overrides)
    return CyclePorts(**fields)  # type: ignore[arg-type]


def test_dry_run_and_status_report_the_due_step_without_acting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    monkeypatch.setattr(week_cycle, "build_ports", _ports_with)

    dry = runner.invoke(app, ["week-cycle", "--dry-run"])
    status = runner.invoke(app, ["week-cycle", "status"])

    assert dry.exit_code == 0, dry.output
    assert "target 2026-2027 week 5: forecast" in dry.output
    assert status.exit_code == 0, status.output
    assert "forecast" in status.output
    assert "No receipts yet." in status.output
    assert not (tmp_path / "data" / "output" / "week_cycle" / "season=2026-2027").exists()


def test_status_lists_the_latest_receipts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_repo(monkeypatch, tmp_path)
    monkeypatch.setattr(week_cycle, "build_ports", _ports_with)
    write_receipt(
        tmp_path,
        CycleReceipt(
            decided_at=NOW.isoformat(),
            season="2026-2027",
            week=5,
            attempted="watch",
            outcome="unchanged",
            detail="30 starters compared, none changed",
        ),
    )

    result = runner.invoke(app, ["week-cycle", "status"])

    assert "week 5  watch: unchanged - 30 starters compared, none changed" in result.output


def test_status_shows_a_pending_recap_rebuild_of_a_closed_week(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    unbuilt = RecapSelectionState(week_selected=True, season_reads_week=False)
    monkeypatch.setattr(
        week_cycle,
        "build_ports",
        lambda: _ports_with(
            product_id=lambda season, week: "p5", recap_state=lambda season, week: unbuilt
        ),
    )
    write_receipt(
        tmp_path,
        CycleReceipt(
            decided_at="2026-10-06T12:00:00+00:00",
            season="2026-2027",
            week=4,
            attempted="closeout",
            outcome="done",
            detail="week 4 closed; soft-failed: build-weekly-recap",
            results={
                "stages": {"build-weekly-recap": "soft-failed"},
                "warnings": ["build-weekly-recap: weekly recap r (new); season recap refused: x"],
            },
        ),
    )

    dry = runner.invoke(app, ["week-cycle", "--dry-run"])
    status = runner.invoke(app, ["week-cycle", "status"])

    line = (
        "week 4 recap rebuild due: the last attempt was an hour or more ago; last: closeout "
        "soft-failed it: weekly recap r (new); season recap refused: x"
    )
    assert line in dry.output
    assert line in status.output


def _raises(ctx: dict[str, object]) -> StageResult:
    raise RuntimeError("boom")


def _recap_stage(monkeypatch: pytest.MonkeyPatch, stage) -> None:
    monkeypatch.setattr(post_week, "_stage_build_weekly_recap", stage)
    monkeypatch.setattr(week_cycle, "_print_report", lambda context: None)


def test_the_recap_rebuild_runs_the_closeout_stage_alone_and_reports_its_recaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []

    def stage(ctx: dict[str, object]) -> StageResult:
        seen.append(dict(ctx))
        ctx.setdefault("cycle_report", []).extend(  # type: ignore[union-attr]
            ["weekly recap r1 (replayed)", "season recap s1 (new)"]
        )
        return StageResult(success=True, detail="weekly recap r1 (replayed); season recap s1 (new)")

    _recap_stage(monkeypatch, stage)

    run = week_cycle._run_recap("2026-2027", 4)

    assert seen == [{"week": 4, "season": "2026-2027", "season_int": 2026}]
    assert run.success is True
    assert run.stages == {"build-weekly-recap": "ok"}
    assert run.detail == "weekly recap r1 (replayed); season recap s1 (new)"
    assert run.report == ["weekly recap r1 (replayed)", "season recap s1 (new)"]


@pytest.mark.parametrize(
    ("stage", "detail"),
    [
        (
            lambda ctx: StageResult(success=False, detail="recap refused: no product."),
            "build-weekly-recap: recap refused: no product.",
        ),
        (_raises, "build-weekly-recap: RuntimeError: boom"),
    ],
)
def test_a_recap_rebuild_that_fails_or_raises_is_a_failure_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, stage, detail: str
) -> None:
    _recap_stage(monkeypatch, stage)

    run = week_cycle._run_recap("2026-2027", 4)

    assert run.success is False
    assert run.failed_stage == "build-weekly-recap"
    assert run.stages == {"build-weekly-recap": "soft-failed"}
    assert run.detail == detail


def _recap_store(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selections: dict[int, str],
    stored: set[str],
    season_reads: tuple[str, ...] | None,
) -> None:
    from gridiron_edge.evaluation import weekly_recap_store as store

    monkeypatch.setattr(
        store,
        "list_current_weekly_recap_selections",
        lambda repo: tuple(
            SimpleNamespace(season=season, week=week, recap_id=recap_id)
            for season in ("2025-2026", "2026-2027")
            for week, recap_id in selections.items()
        ),
    )
    monkeypatch.setattr(
        store,
        "weekly_recap_manifest_path",
        lambda recap_id, repo: SimpleNamespace(is_file=lambda: recap_id in stored),
    )
    monkeypatch.setattr(
        store,
        "get_current_season_recap_selection",
        lambda season, repo: None if season_reads is None else SimpleNamespace(season_recap_id="s"),
    )
    monkeypatch.setattr(
        store,
        "load_season_recap",
        lambda recap_id, repo: SimpleNamespace(weekly_recap_ids=season_reads),
    )


@pytest.mark.parametrize(
    ("selections", "stored", "season_reads", "expected"),
    [
        # The week and a season recap reading it are selected: nothing is pending.
        ({3: "a", 4: "b"}, {"a", "b"}, ("a", "b"), RecapSelectionState(True, True)),
        # The weekly recap was selected but the season roll-up refused.
        ({3: "a", 4: "b"}, {"a", "b"}, ("a",), RecapSelectionState(True, False)),
        # No season recap is selected at all.
        ({4: "b"}, {"b"}, None, RecapSelectionState(True, False)),
        # The week has no selection.
        ({3: "a"}, {"a"}, ("a",), RecapSelectionState(False, False)),
        # Week 2 is selected at an older schema: the season roll-up waits on it.
        ({2: "x", 4: "b"}, {"b"}, ("x",), RecapSelectionState(True, False, (2,))),
        # The week itself is at an older schema: rebuilding it is what the retry does.
        ({4: "old"}, set(), ("old",), RecapSelectionState(False, False)),
    ],
)
def test_the_recap_state_reads_the_selections_of_the_week_and_season(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selections: dict[int, str],
    stored: set[str],
    season_reads: tuple[str, ...] | None,
    expected: RecapSelectionState,
) -> None:
    _use_repo(monkeypatch, tmp_path)
    _recap_store(monkeypatch, selections=selections, stored=stored, season_reads=season_reads)

    assert week_cycle._recap_state("2026-2027", 4) == expected


def test_a_run_does_what_is_due_and_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    failing = StageRun(success=False, detail="predict-week: boom", failed_stage="predict-week")
    monkeypatch.setattr(
        week_cycle,
        "build_ports",
        lambda: _ports_with(dirty_files=lambda: [], run_forecast=lambda season, week: failing),
    )

    result = runner.invoke(app, ["week-cycle"])

    assert result.exit_code == 1
    assert "forecast: failed - predict-week: boom" in result.output


def test_a_second_run_is_refused_while_one_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    monkeypatch.setattr(week_cycle, "build_ports", _ports_with)

    with week_cycle._single_run(tmp_path) as first:
        assert first is True
        result = runner.invoke(app, ["week-cycle"])
        with week_cycle._single_run(tmp_path) as second:
            assert second is False

    assert result.exit_code == 0
    assert "Another week-cycle run is in progress" in result.output


def test_season_and_week_overrides_must_be_given_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, tmp_path)
    monkeypatch.setattr(week_cycle, "build_ports", _ports_with)

    half = runner.invoke(app, ["week-cycle", "--dry-run", "--week", "5"])
    bad = runner.invoke(app, ["week-cycle", "--dry-run", "--season", "nonsense", "--week", "5"])
    both = runner.invoke(app, ["week-cycle", "--dry-run", "--season", "2026-2027", "--week", "5"])

    assert half.exit_code == 2
    assert bad.exit_code == 2
    assert both.exit_code == 0, both.output
    assert "target 2026-2027 week 5" in both.output


def test_coverage_gaps_name_what_each_artifact_lacks() -> None:
    gaps = week_cycle.coverage_gaps(
        {"a", "b", "c"},
        {"complete": {"a", "b", "c", "x"}, "partial": {"a"}, "absent": None},
    )

    assert gaps == {"partial": ["b", "c"], "absent": ["a", "b", "c"]}


def _data_current_repo(tmp_path: Path, *, stats: list[str]) -> Path:
    raw = tmp_path / "data" / "raw"
    (raw / "player_stats").mkdir(parents=True)
    (raw / "pbp").mkdir()
    cleaned = tmp_path / "data" / "cleaned"
    cleaned.mkdir(parents=True)
    modeling = tmp_path / "data" / "modeling"
    modeling.mkdir()
    DataFrame(
        {
            "season": [2026, 2026],
            "week": [4, 4],
            "game_type": ["REG", "REG"],
            "game_id": ["a", "b"],
            "result": [3.0, 7.0],
        }
    ).to_parquet(raw / "NFL_wk_by_wk_nflverse.parquet")
    both = DataFrame({"game_id": ["a", "b"]})
    DataFrame({"game_id": stats}).to_parquet(raw / "player_stats" / "player_stats_2026.parquet")
    both.to_parquet(raw / "pbp" / "play_by_play_2026.parquet")
    both.to_parquet(cleaned / "epa_by_game.parquet")
    both.to_parquet(cleaned / "player_game_logs.parquet")
    DataFrame({"GAME_ID": ["a", "b"]}).to_parquet(modeling / "modeling_file.parquet")
    return tmp_path


def test_data_current_gate_waits_for_a_lagging_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, _data_current_repo(tmp_path, stats=["a"]))
    context: dict = {"season_int": 2026, "week": 4}

    result = week_cycle._stage_verify_week_data_current(context)
    allowed = week_cycle._stage_verify_week_data_current({**context, "allow_incomplete": True})

    assert result.success is False
    assert "player stats missing 1" in result.detail
    assert "retrying hourly" in result.detail
    assert "player stats: 1 of 2 week 4 games" in context["cycle_report"]
    assert "play-by-play: 2 of 2 week 4 games" in context["cycle_report"]
    assert allowed.success is True and allowed.warnings


def test_data_current_gate_passes_when_every_artifact_holds_the_week(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_repo(monkeypatch, _data_current_repo(tmp_path, stats=["a", "b"]))

    result = week_cycle._stage_verify_week_data_current({"season_int": 2026, "week": 4})

    assert result.success is True
