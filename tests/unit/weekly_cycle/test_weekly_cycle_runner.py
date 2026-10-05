# tests/unit/weekly_cycle/test_weekly_cycle_runner.py

"""Tests for running the weekly cycle against stubbed effects (``DECISIONS.md`` D82)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pandas import DataFrame
import pytest

from gridiron_edge.weekly_cycle.decision import StepKind
from gridiron_edge.weekly_cycle.receipts import read_receipts
from gridiron_edge.weekly_cycle.runner import (
    AvailabilityRefresh,
    CyclePorts,
    StageRun,
    run_cycle,
)
from gridiron_edge.weekly_cycle.starter_watch import StarterComparison, StarterObservation

pytestmark = pytest.mark.unit

SEASON = "2026-2027"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _game(week: int, game_id: str, date: str, time: str) -> dict[str, object]:
    return {
        "season": SEASON,
        "week": week,
        "game_id": game_id,
        "game_date": date,
        "game_time": time,
    }


SCHEDULE = DataFrame(
    [
        _game(4, "w4-sun", "2026-10-04", "13:00:00"),
        _game(4, "w4-mon", "2026-10-05", "20:15:00"),  # 2026-10-06T00:15Z
        _game(5, "w5-thu", "2026-10-08", "20:15:00"),  # 2026-10-09T00:15Z
        _game(5, "w5-sun", "2026-10-11", "13:00:00"),
    ]
)


def _comparison(*, changed: bool) -> StarterComparison:
    return StarterComparison(
        game_id="w5-sun",
        side="away",
        recorded_id="a1",
        recorded_source="carried_forward",
        current_id="a2" if changed else "a1",
        current_source="depth_chart_report",
        current_as_of="2026-10-07T12:00:00Z",
        changed=changed,
    )


class Fake:
    """Stub effects that record calls; a product exists for the weeks in ``products``."""

    def __init__(self, now: str, *, products: set[int] | None = None) -> None:
        self.now = _utc(now)
        self.products = set(products or ())
        self.calls: list[str] = []
        self.dirty: list[str] = []
        self.closeout = StageRun(success=True, detail="week closed")
        self.forecast = StageRun(success=True, detail="forecast selected")
        self.edges = StageRun(success=True, detail="3 edges")
        self.refreshed = AvailabilityRefresh(True, True)
        self.changed = False
        self.record_available = True
        self.digest = "d1"
        self.market = "2026-10-06T08:00:00+00:00"
        self.raise_on: str | None = None

    def _tick(self) -> datetime:
        """Return the current time, one second later on every read so receipts order."""
        current = self.now
        self.now = current + timedelta(seconds=1)
        return current

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.raise_on == name:
            raise RuntimeError(f"{name} exploded")

    def ports(self) -> CyclePorts:
        def run_closeout(season: str, week: int, allow_incomplete: bool) -> StageRun:
            self._call("closeout")
            return self.closeout

        def run_forecast(season: str, week: int) -> StageRun:
            self._call("forecast")
            if self.forecast.success:
                self.products.add(week)
            return self.forecast

        def run_edges(season: str, week: int) -> StageRun:
            self._call("edges")
            return self.edges

        def refresh(season_int: int) -> AvailabilityRefresh:
            self._call("refresh_availability")
            return self.refreshed

        def observe(
            season: str, week: int, product: DataFrame, schedule: DataFrame, now: datetime
        ) -> StarterObservation:
            self._call("observe")
            comparisons = (_comparison(changed=self.changed),) if self.record_available else ()
            return StarterObservation(
                record_available=self.record_available,
                comparisons=comparisons,
                started_game_ids=(),
                teams={},
            )

        return CyclePorts(
            clock=self._tick,
            load_schedule=lambda: SCHEDULE,
            product_id=lambda season, week: f"product-{week}" if week in self.products else None,
            load_product=lambda season, week: DataFrame({"game_id": ["w5-sun"]}),
            dirty_files=lambda: self.dirty,
            run_closeout=run_closeout,
            run_forecast=run_forecast,
            run_edges=run_edges,
            refresh_availability=refresh,
            digests=lambda season_int: {"depth_charts": self.digest, "injuries": "i1"},
            observe=observe,
            edge_inputs=lambda season, week: {
                "product_id": f"product-{week}",
                "market_fetched_at": self.market,
            },
        )

    def run(self, repo: Path, **kwargs: Any):
        return run_cycle(repo=repo, ports=self.ports(), tz=UTC, **kwargs)


def _receipts(repo: Path, week: int) -> list[tuple[str, str]]:
    """The week's non-edge receipts, oldest first."""
    return [
        (r.attempted, r.outcome)
        for r in read_receipts(repo, season=SEASON, week=week)
        if r.attempted != "edges"
    ]


def test_a_dry_run_reports_the_decision_and_writes_nothing(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")

    outcome = fake.run(tmp_path, dry_run=True)

    assert outcome.decision is not None and outcome.decision.kind is StepKind.CLOSEOUT
    assert "closeout" in outcome.lines[0]
    assert fake.calls == []
    assert not (tmp_path / "data").exists()


def test_before_the_window_nothing_is_done_and_nothing_recorded(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T02:00:00")

    outcome = fake.run(tmp_path)

    assert fake.calls == []
    assert outcome.failed is False
    assert "next check 2026-10-06 06:00 UTC" in outcome.lines[0]
    assert not (tmp_path / "data").exists()


def test_results_not_in_yet_is_a_quiet_waiting_receipt(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.closeout = StageRun(
        success=False,
        detail="verify-week-complete: 13 of 16 games have results",
        failed_stage="verify-week-complete",
    )

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert _receipts(tmp_path, 4) == [("closeout", "waiting")]
    assert fake.calls == ["closeout"]
    # The hourly retry is held back until an hour has passed.
    fake.now = _utc("2026-10-06T06:50:00")
    fake.run(tmp_path)
    assert fake.calls == ["closeout"]


def test_lagging_downstream_data_is_a_quiet_waiting_receipt(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.closeout = StageRun(
        success=False,
        detail="upstream data is not current for week 4: play-by-play missing 2",
        failed_stage="verify-week-data-current",
        report=["play-by-play: 14 of 16 week 4 games"],
    )

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert _receipts(tmp_path, 4) == [("closeout", "waiting")]
    receipt = read_receipts(tmp_path, season=SEASON, week=4)[0]
    assert receipt.results["report"] == ["play-by-play: 14 of 16 week 4 games"]
    assert "forecast" not in fake.calls


def test_a_closeout_failure_is_a_failed_receipt_and_a_failed_run(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.closeout = StageRun(
        success=False, detail="refresh-results: offline", failed_stage="refresh-results"
    )

    outcome = fake.run(tmp_path)

    assert outcome.failed is True
    assert _receipts(tmp_path, 4) == [("closeout", "failed")]


def test_closeout_then_forecast_run_in_one_tick_and_edges_are_recorded_with_the_forecast(
    tmp_path: Path,
) -> None:
    fake = Fake("2026-10-06T06:30:00")

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert fake.calls == ["closeout", "forecast"]
    assert _receipts(tmp_path, 4) == [("closeout", "done")]
    assert _receipts(tmp_path, 5) == [("forecast", "done")]
    forecast = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert forecast.results["product_before"] is None
    assert forecast.results["product_after"] == "product-5"
    assert forecast.results["edge_inputs"] == {
        "product_id": "product-5",
        "market_fetched_at": fake.market,
    }
    assert forecast.results["digests"] == {"depth_charts": "d1", "injuries": "i1"}


def test_a_soft_failed_recap_still_closes_the_week_and_the_receipt_names_it(
    tmp_path: Path,
) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.closeout = StageRun(
        success=True,
        detail="week 4 closed; soft-failed: build-weekly-recap",
        stages={"close-live-forecasts": "ok", "build-weekly-recap": "soft-failed"},
        warnings=["build-weekly-recap: recap refused: no weekly product is selected."],
        report=["EPA by game: 16 of 16 week 4 games"],
    )

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert fake.calls == ["closeout", "forecast"]
    closeout = read_receipts(tmp_path, season=SEASON, week=4)[0]
    assert (closeout.attempted, closeout.outcome) == ("closeout", "done")
    assert closeout.detail == "week 4 closed; soft-failed: build-weekly-recap"
    assert closeout.results["stages"]["build-weekly-recap"] == "soft-failed"
    assert closeout.results["warnings"] == fake.closeout.warnings


def test_the_closeout_receipt_records_the_selected_recap(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.closeout = StageRun(
        success=True,
        detail="week 4 closed",
        stages={"build-weekly-recap": "ok"},
        report=["weekly recap abc123 (new)"],
    )

    fake.run(tmp_path)

    closeout = read_receipts(tmp_path, season=SEASON, week=4)[0]
    assert closeout.results["stages"] == {"build-weekly-recap": "ok"}
    assert closeout.results["report"] == ["weekly recap abc123 (new)"]


def test_a_dirty_worktree_blocks_the_forecast_without_failing(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.dirty = ["a.py", "b.py"]

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert fake.calls == ["closeout"]
    receipt = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert (receipt.attempted, receipt.outcome) == ("forecast", "blocked")
    assert "a.py, b.py" in receipt.detail
    assert receipt.results["dirty_files"] == ["a.py", "b.py"]


def test_a_forecast_failure_is_recorded_and_not_retried_within_the_hour(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.forecast = StageRun(
        success=False, detail="predict-week: boom", failed_stage="predict-week"
    )

    first = fake.run(tmp_path)
    fake.now = _utc("2026-10-06T06:50:00")
    fake.run(tmp_path)

    assert first.failed is True
    assert fake.calls == ["closeout", "forecast"]
    assert _receipts(tmp_path, 5) == [("forecast", "failed")]


def test_an_unexpected_exception_becomes_a_failed_receipt(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")
    fake.raise_on = "closeout"

    outcome = fake.run(tmp_path)

    assert outcome.failed is True
    receipt = read_receipts(tmp_path, season=SEASON, week=4)[0]
    assert (receipt.attempted, receipt.outcome) == ("closeout", "failed")
    assert "closeout exploded" in receipt.detail


def test_the_first_watch_with_no_change_records_the_evidence(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert fake.calls[:3] == ["refresh_availability", "observe", "edges"]
    receipt = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert (receipt.attempted, receipt.outcome) == ("watch", "unchanged")
    assert receipt.results["disposition"] == "no_change"
    assert receipt.results["digests"] == {"depth_charts": "d1", "injuries": "i1"}
    assert receipt.results["upstream_changed"] == {"depth_charts": None, "injuries": None}
    assert receipt.inputs["interval_hours"] == 12.0
    assert receipt.inputs["game_started"] is False


def test_the_next_watch_says_whether_the_upstream_files_changed(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.run(tmp_path)
    fake.now = _utc("2026-10-06T20:30:00")
    fake.digest = "d2"

    fake.run(tmp_path)

    receipts = read_receipts(tmp_path, season=SEASON, week=5)
    assert [r.attempted for r in receipts if r.attempted == "watch"] == ["watch", "watch"]
    assert receipts[-1].results["upstream_changed"] == {"depth_charts": True, "injuries": False}


def test_a_changed_starter_before_the_first_kickoff_refreshes_the_forecast(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.changed = True

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert "forecast" in fake.calls
    assert _receipts(tmp_path, 5) == [
        ("watch", "changed"),
        ("refresh", "done"),
    ]
    watch = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert watch.results["disposition"] == "refresh"
    assert watch.results["changed"][0]["recorded_id"] == "a1"


def test_a_changed_starter_after_the_first_kickoff_still_refreshes_the_rest_of_the_week(
    tmp_path: Path,
) -> None:
    fake = Fake("2026-10-09T06:00:00", products={5})  # Thursday's game has started
    fake.changed = True

    fake.run(tmp_path)

    assert "forecast" in fake.calls
    assert _receipts(tmp_path, 5) == [("watch", "changed"), ("refresh", "done")]
    receipt = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert receipt.results["disposition"] == "refresh"
    assert receipt.inputs["game_started"] is True


def test_a_changed_starter_with_a_dirty_tree_records_a_blocked_refresh(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.changed = True
    fake.dirty = ["x.py"]

    outcome = fake.run(tmp_path)

    assert outcome.failed is False
    assert "forecast" not in fake.calls
    assert _receipts(tmp_path, 5) == [("watch", "changed"), ("refresh", "blocked")]


def test_a_refresh_is_not_attempted_twice_within_the_hour(tmp_path: Path) -> None:
    fake = Fake("2026-10-09T00:00:00", products={5})  # 15 min to kickoff: hourly watch
    fake.changed = True
    fake.forecast = StageRun(
        success=False, detail="predict-week: boom", failed_stage="predict-week"
    )
    fake.run(tmp_path)
    fake.calls.clear()
    fake.forecast = StageRun(success=True, detail="ok")
    fake.now = _utc("2026-10-09T00:10:00")  # the next tick, still inside the hour

    fake.run(tmp_path)

    assert fake.calls == []  # the failed refresh retries the watch hourly, not sooner


def test_a_product_without_a_starter_record_is_watched_but_never_refreshed(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.record_available = False

    fake.run(tmp_path)

    assert "forecast" not in fake.calls
    receipt = read_receipts(tmp_path, season=SEASON, week=5)[0]
    assert receipt.results["disposition"] == "no_starter_record"


def test_a_failed_upstream_refresh_fails_the_watch_without_comparing(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.refreshed = AvailabilityRefresh(depth_charts_ok=True, injuries_ok=False)

    outcome = fake.run(tmp_path)

    assert outcome.failed is True
    assert "observe" not in fake.calls
    assert _receipts(tmp_path, 5) == [("watch", "failed")]


def test_edges_rebuild_only_when_the_snapshot_or_product_changed(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.run(tmp_path)  # watch, then the first edge report
    assert fake.calls.count("edges") == 1
    fake.now = _utc("2026-10-06T08:30:00")  # nothing due, nothing new
    fake.run(tmp_path)
    assert fake.calls.count("edges") == 1

    fake.market = "2026-10-06T08:15:00+00:00"
    fake.now = _utc("2026-10-06T08:45:00")
    fake.run(tmp_path)

    assert fake.calls.count("edges") == 2
    edges = [r for r in read_receipts(tmp_path, season=SEASON, week=5) if r.attempted == "edges"]
    assert [r.outcome for r in edges] == ["done", "done"]
    assert edges[-1].results["edge_inputs"]["market_fetched_at"] == fake.market


def test_blocked_edges_for_a_stale_market_are_not_retried_on_the_same_snapshot(
    tmp_path: Path,
) -> None:
    fake = Fake("2026-10-06T08:00:00", products={5})
    fake.edges = StageRun(success=False, detail="edge calculation blocked: market_stale")
    fake.run(tmp_path)
    fake.now = _utc("2026-10-06T08:40:00")

    fake.run(tmp_path)

    assert fake.calls.count("edges") == 1
    edges = [r for r in read_receipts(tmp_path, season=SEASON, week=5) if r.attempted == "edges"]
    assert [r.outcome for r in edges] == ["blocked"]


def test_after_the_forecast_the_edge_report_is_not_rebuilt_in_the_same_tick(tmp_path: Path) -> None:
    fake = Fake("2026-10-06T06:30:00")

    fake.run(tmp_path)

    assert "edges" not in fake.calls
