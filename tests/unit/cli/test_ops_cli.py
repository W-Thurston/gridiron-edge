# tests/unit/cli/test_ops_cli.py
"""Tests for the ops CLI: cross-machine collector-evidence pull and rollover."""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame
from typer.testing import CliRunner

from gridiron_edge.cli.main import app
from gridiron_edge.core.settings import Settings

runner = CliRunner()


def _rollover_schedule() -> DataFrame:
    return DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 3,
                "game_id": "g",
                "game_date": "2026-09-29",
                "game_time": "20:20:00",
            }
        ]
    )


def _patch_rollover_dependencies(monkeypatch, tmp_path: Path, *, push=None) -> dict[str, object]:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _settings(tmp_path),
    )

    def fake_fetch_upcoming(**kwargs):
        observed["fetch_upcoming"] = kwargs

    monkeypatch.setattr(
        "gridiron_edge.ingest.nflverse.schedule.fetch_nflverse_upcoming",
        fake_fetch_upcoming,
    )
    monkeypatch.setattr(
        "gridiron_edge.transform.clean.schedule_nflverse.clean_nflverse_upcoming",
        lambda **kwargs: observed.update(clean_upcoming=kwargs) or tmp_path / "schedule.parquet",
    )
    monkeypatch.setattr(
        "gridiron_edge.datasets.loaders.load_schedule_upcoming_rich",
        lambda *_a, **_k: _rollover_schedule(),
    )
    monkeypatch.setattr(
        "gridiron_edge.market.collection_plan_store.write_collection_plan",
        lambda plan, **kwargs: observed.update(written_plan=plan) or tmp_path / "week=03.json",
    )

    def fake_select(**kwargs):
        from types import SimpleNamespace

        observed["select_kwargs"] = kwargs
        return SimpleNamespace(season=kwargs["season"], week=kwargs["week"])

    monkeypatch.setattr(
        "gridiron_edge.market.collection_plan_store.select_current_collection_plan",
        fake_select,
    )

    def fake_push(target, **kwargs):
        observed["push_target"] = target
        observed["push_kwargs"] = kwargs
        if push is not None:
            return push(target, **kwargs)
        return ()

    monkeypatch.setattr(
        "gridiron_edge.deployment.collector_sync.push_collector_plan",
        fake_push,
    )
    return observed


def _settings(repo_root: Path) -> Settings:
    return Settings(
        repo_root=repo_root,
        owm_api_key=None,
        odds_api_key=None,
        data_raw=repo_root / "data" / "raw",
        data_cleaned=repo_root / "data" / "cleaned",
        data_modeling=repo_root / "data" / "modeling",
        data_output=repo_root / "data" / "output",
    )


def test_pull_collector_evidence_requires_host(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GRIDIRON_COLLECTOR_HOST", raising=False)

    result = runner.invoke(app, ["ops", "pull-collector-evidence"])

    assert result.exit_code == 2
    assert "GRIDIRON_COLLECTOR_HOST" in result.stderr


def test_pull_collector_evidence_uses_host_env_var_and_reports_files(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GRIDIRON_COLLECTOR_HOST", "10.0.0.49")
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _settings(tmp_path),
    )
    observed: dict[str, object] = {}

    def fake_pull(target, *, local_repo, dry_run=False):
        observed["target"] = target
        observed["local_repo"] = local_repo
        observed["dry_run"] = dry_run
        import subprocess

        return subprocess.CompletedProcess(
            (),
            0,
            stdout="week=03/observations.parquet\n",
            stderr="",
        )

    monkeypatch.setattr(
        "gridiron_edge.deployment.collector_sync.pull_collector_evidence",
        fake_pull,
    )

    result = runner.invoke(app, ["ops", "pull-collector-evidence"])

    assert result.exit_code == 0
    assert observed["local_repo"] == tmp_path
    assert observed["dry_run"] is False
    target = observed["target"]
    assert target.host == "10.0.0.49"
    assert target.user == "thursty"
    assert target.remote_repository == "/home/thursty/apps/gridiron-edge"
    assert "week=03/observations.parquet" in result.stdout
    assert "pulled 1 file(s)" in result.stdout


def test_pull_collector_evidence_reports_up_to_date_when_nothing_transferred(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GRIDIRON_COLLECTOR_HOST", "10.0.0.49")
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _settings(tmp_path),
    )

    def fake_pull(target, *, local_repo, dry_run=False):
        import subprocess

        return subprocess.CompletedProcess((), 0, stdout="", stderr="")

    monkeypatch.setattr(
        "gridiron_edge.deployment.collector_sync.pull_collector_evidence",
        fake_pull,
    )

    result = runner.invoke(app, ["ops", "pull-collector-evidence"])

    assert result.exit_code == 0
    assert "up to date" in result.stdout


def test_pull_collector_evidence_surfaces_sync_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GRIDIRON_COLLECTOR_HOST", "10.0.0.49")
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _settings(tmp_path),
    )

    def fake_pull(target, *, local_repo, dry_run=False):
        from gridiron_edge.deployment.collector_sync import CollectorSyncError

        raise CollectorSyncError("rsync exited 255: connection timed out")

    monkeypatch.setattr(
        "gridiron_edge.deployment.collector_sync.pull_collector_evidence",
        fake_pull,
    )

    result = runner.invoke(app, ["ops", "pull-collector-evidence"])

    assert result.exit_code == 2
    assert "connection timed out" in result.stderr


def test_pull_collector_evidence_overrides_user_and_repository_and_identity(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_pull(target, *, local_repo, dry_run=False):
        observed["target"] = target
        import subprocess

        return subprocess.CompletedProcess((), 0, stdout="", stderr="")

    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings",
        lambda: _settings(tmp_path),
    )
    monkeypatch.setattr(
        "gridiron_edge.deployment.collector_sync.pull_collector_evidence",
        fake_pull,
    )

    result = runner.invoke(
        app,
        [
            "ops",
            "pull-collector-evidence",
            "--host",
            "10.0.0.49",
            "--user",
            "someone",
            "--remote-repository",
            "/srv/gridiron-edge",
            "--identity",
            "/home/thursty/.ssh/id_ed25519",
        ],
    )

    assert result.exit_code == 0
    target = observed["target"]
    assert target.user == "someone"
    assert target.remote_repository == "/srv/gridiron-edge"
    assert target.identity_file == "/home/thursty/.ssh/id_ed25519"


def _invoke_rollover(*extra_args: str, input_text: str | None = None):
    return runner.invoke(
        app,
        [
            "ops",
            "rollover-collector-week",
            "--season",
            "2026-2027",
            "--week",
            "3",
            "--host",
            "10.0.0.49",
            *extra_args,
        ],
        input=input_text,
    )


def test_rollover_requires_host(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GRIDIRON_COLLECTOR_HOST", raising=False)

    result = runner.invoke(
        app,
        ["ops", "rollover-collector-week", "--season", "2026-2027", "--week", "3"],
    )

    assert result.exit_code == 2
    assert "GRIDIRON_COLLECTOR_HOST" in result.stderr


def test_rollover_declining_confirmation_does_not_select_or_push(
    monkeypatch, tmp_path: Path
) -> None:
    observed = _patch_rollover_dependencies(monkeypatch, tmp_path)

    result = _invoke_rollover(input_text="n\n")

    assert result.exit_code == 1
    assert "select_kwargs" not in observed
    assert "push_kwargs" not in observed
    assert "built plan: season=2026-2027 week=3" in result.stdout


def test_rollover_confirming_selects_and_pushes(monkeypatch, tmp_path: Path) -> None:
    observed = _patch_rollover_dependencies(monkeypatch, tmp_path)

    result = _invoke_rollover(input_text="y\n")

    assert result.exit_code == 0, result.output
    assert observed["select_kwargs"]["season"] == "2026-2027"
    assert observed["select_kwargs"]["week"] == 3
    assert observed["push_kwargs"]["season"] == "2026-2027"
    assert observed["push_kwargs"]["week"] == 3
    assert observed["push_target"].host == "10.0.0.49"
    assert "Verify on the worker:" in result.stdout


def test_rollover_yes_flag_skips_prompt(monkeypatch, tmp_path: Path) -> None:
    observed = _patch_rollover_dependencies(monkeypatch, tmp_path)

    result = _invoke_rollover("--yes")

    assert result.exit_code == 0, result.output
    assert observed["select_kwargs"]["week"] == 3
    assert observed["push_kwargs"]["week"] == 3


def test_rollover_surfaces_push_failure(monkeypatch, tmp_path: Path) -> None:
    def failing_push(target, **kwargs):
        from gridiron_edge.deployment.collector_sync import CollectorSyncError

        raise CollectorSyncError("rsync exited 255: connection timed out")

    observed = _patch_rollover_dependencies(monkeypatch, tmp_path, push=failing_push)

    result = _invoke_rollover("--yes")

    assert result.exit_code == 2
    assert "connection timed out" in result.stderr
    assert observed["select_kwargs"]["week"] == 3


def test_rollover_refreshes_schedule_before_building_plan(monkeypatch, tmp_path: Path) -> None:
    observed = _patch_rollover_dependencies(monkeypatch, tmp_path)

    result = _invoke_rollover("--yes")

    assert result.exit_code == 0, result.output
    assert observed["fetch_upcoming"]["season"] == 2026
    assert "written_plan" in observed
    written_plan = observed["written_plan"]
    assert written_plan.season == "2026-2027"
    assert written_plan.week == 3
