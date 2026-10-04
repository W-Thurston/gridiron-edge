# tests/unit/deployment/test_collector_merge.py

"""Tests for the evidence-preserving merge of a staged worker quote tree."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.deployment.collector_merge import merge_collector_tree
from gridiron_edge.ingest.odds.store import (
    QUOTE_COLUMNS,
    load_current_odds,
    odds_write_lock,
)

pytestmark = pytest.mark.unit

SEASON = "2026-2027"


def _quotes(fetched_at: str, *, odds: float = -110.0, week: int = 4) -> DataFrame:
    rows = []
    for side in ("home", "away"):
        rows.append(
            {
                "fetched_at": pd.Timestamp(fetched_at, tz="UTC"),
                "provider": "the_odds_api",
                "provider_event_id": "evt",
                "sportsbook": "book",
                "sportsbook_updated_at": pd.Timestamp(fetched_at, tz="UTC"),
                "commence_time": pd.Timestamp("2026-10-04T17:00:00Z"),
                "is_live": False,
                "season": SEASON,
                "week": week,
                "game_id": "2026_04_AAA_BBB",
                "game_date": "2026-10-04",
                "away_team": "A",
                "home_team": "B",
                "market": "moneyline",
                "side": side,
                "odds": odds if side == "home" else 100.0,
                "line": float("nan"),
            }
        )
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def _partition(root: Path, week: int = 4) -> Path:
    return root / "history" / f"season={SEASON}" / f"week={week:02d}" / "observations.parquet"


def _write(frame: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    staged = tmp_path / "stage"
    local = tmp_path / "repo" / "data" / "odds"
    staged.mkdir(parents=True)
    local.mkdir(parents=True)
    return staged, local


def test_local_only_observations_survive_and_pulled_ones_are_added(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(_quotes("2026-09-30T12:38:29Z"), _partition(local))
    _write(_quotes("2026-10-03T19:30:38Z"), _partition(staged))

    report = merge_collector_tree(staged, local)

    merged = pd.read_parquet(_partition(local))
    assert sorted(merged["fetched_at"].dt.strftime("%Y-%m-%dT%H:%M:%S").unique()) == [
        "2026-09-30T12:38:29",
        "2026-10-03T19:30:38",
    ]
    assert report.partitions == {"history/season=2026-2027/week=04/observations.parquet": (2, 4)}
    assert report.rows_added == 2


def test_a_partition_never_loses_rows_and_a_repeat_changes_nothing(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(
        pd.concat([_quotes("2026-09-30T12:38:29Z"), _quotes("2026-10-01T12:00:00Z")]),
        _partition(local),
    )
    _write(_quotes("2026-10-01T12:00:00Z"), _partition(staged))
    before = _partition(local).read_bytes()

    report = merge_collector_tree(staged, local)

    assert report.partitions[next(iter(report.partitions))] == (4, 4)
    assert _partition(local).read_bytes() == before
    again = merge_collector_tree(staged, local)
    assert again.rows_added == 0


def test_a_new_partition_is_created(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(_quotes("2026-10-03T19:30:38Z", week=5), _partition(staged, week=5))

    merge_collector_tree(staged, local)

    assert len(pd.read_parquet(_partition(local, week=5))) == 2


def test_conflicting_observations_raise_and_leave_the_local_file_alone(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(_quotes("2026-10-03T19:30:38Z", odds=-110.0), _partition(local))
    _write(_quotes("2026-10-03T19:30:38Z", odds=-120.0), _partition(staged))
    before = _partition(local).read_bytes()

    with pytest.raises(ValueError, match="conflict"):
        merge_collector_tree(staged, local)

    assert _partition(local).read_bytes() == before


def test_the_later_current_snapshot_wins_and_an_older_one_does_not_replace(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    repo = local.parent.parent
    _write(_quotes("2026-09-30T12:38:29Z"), local / "odds_current.parquet")
    _write(_quotes("2026-10-03T19:30:38Z"), staged / "odds_current.parquet")

    report = merge_collector_tree(staged, local)

    assert report.current_snapshot == "replaced by the worker's later snapshot"
    assert load_current_odds(repo=repo)["fetched_at"].max() == pd.Timestamp("2026-10-03T19:30:38Z")

    _write(_quotes("2026-10-01T00:00:00Z"), staged / "odds_current.parquet")
    older = merge_collector_tree(staged, local)

    assert older.current_snapshot == "kept local (not older than the worker's)"
    assert load_current_odds(repo=repo)["fetched_at"].max() == pd.Timestamp("2026-10-03T19:30:38Z")


def test_missing_local_snapshot_is_taken_from_the_worker(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(_quotes("2026-10-03T19:30:38Z"), staged / "odds_current.parquet")

    report = merge_collector_tree(staged, local)

    assert report.current_snapshot == "taken from worker (no local snapshot)"
    assert (local / "odds_current.parquet").exists()


def test_immutable_files_are_copied_only_when_absent_and_conflicts_are_reported(
    tmp_path: Path,
) -> None:
    staged, local = _dirs(tmp_path)
    for name, text in (("new.json", "{}"), ("same.json", "x"), ("diff.json", "worker")):
        (staged / "collection_runs").mkdir(exist_ok=True)
        (staged / "collection_runs" / name).write_text(text)
    (local / "collection_runs").mkdir()
    (local / "collection_runs" / "same.json").write_text("x")
    (local / "collection_runs" / "diff.json").write_text("local")

    report = merge_collector_tree(staged, local)

    assert report.copied == ["collection_runs/new.json"]
    assert report.unchanged == ["collection_runs/same.json"]
    assert report.conflicts == ["collection_runs/diff.json"]
    assert (local / "collection_runs" / "diff.json").read_text() == "local"


def test_the_current_plan_pointer_and_unknown_files_are_never_pulled(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    (staged / "collection_plans").mkdir()
    (staged / "collection_plans" / "current.json").write_text('{"week": 3}')
    (staged / "notes.txt").write_text("ignored")
    (local / "collection_plans").mkdir()
    (local / "collection_plans" / "current.json").write_text('{"week": 4}')

    merge_collector_tree(staged, local)

    assert (local / "collection_plans" / "current.json").read_text() == '{"week": 4}'
    assert not (local / "notes.txt").exists()


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    staged, local = _dirs(tmp_path)
    _write(_quotes("2026-10-03T19:30:38Z"), _partition(staged))
    _write(_quotes("2026-10-03T19:30:38Z"), staged / "odds_current.parquet")
    (staged / "collection_runs").mkdir()
    (staged / "collection_runs" / "r.json").write_text("{}")

    report = merge_collector_tree(staged, local, write=False)

    assert report.rows_added == 2
    assert report.copied == ["collection_runs/r.json"]
    assert not _partition(local).exists()
    assert not (local / "odds_current.parquet").exists()
    assert not (local / "collection_runs").exists()


def test_the_write_lock_is_exclusive(tmp_path: Path) -> None:
    with (
        odds_write_lock(repo=tmp_path),
        pytest.raises(TimeoutError),
        odds_write_lock(repo=tmp_path, timeout_seconds=0.3),
    ):
        pass
    with odds_write_lock(repo=tmp_path):
        pass
