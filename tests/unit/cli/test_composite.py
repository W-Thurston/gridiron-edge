"""Tests for the composite CLI infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    CompositeSummary,
    StageResult,
    _check_dependencies,
    render_composite_summary,
    resolve_active_stages,
    run_composite,
)

# ---------------------------------------------------------------------------
# Stage helpers for tests
# ---------------------------------------------------------------------------


def _always_succeed(ctx: dict[str, Any]) -> StageResult:
    return StageResult(success=True, detail="ok")


def _always_fail(ctx: dict[str, Any]) -> StageResult:
    return StageResult(success=False, detail="failure")


def _always_raise(ctx: dict[str, Any]) -> StageResult:
    raise RuntimeError("boom")


def _write_artifact(ctx: dict[str, Any]) -> StageResult:
    return StageResult(
        success=True,
        detail="wrote 1 artifact",
        artifacts=[Path("/tmp/fake.txt")],
    )


def _write_warning(ctx: dict[str, Any]) -> StageResult:
    return StageResult(
        success=True,
        detail="completed with warning",
        warnings=["deprecation: use --new-flag"],
    )


def _read_ctx_value(ctx: dict[str, Any]) -> StageResult:
    """Stage that reads a context value set by an upstream stage."""
    value = ctx.get("upstream_value")
    if value is None:
        return StageResult(success=False, detail="upstream_value missing")
    return StageResult(success=True, detail=f"got {value}")


def _set_ctx_value(ctx: dict[str, Any]) -> StageResult:
    """Stage that sets a value for downstream stages to consume."""
    ctx["upstream_value"] = 42
    return StageResult(success=True, detail="set upstream value")


# ---------------------------------------------------------------------------
# resolve_active_stages
# ---------------------------------------------------------------------------


class TestResolveActiveStages:
    def test_default_active_is_all(self) -> None:
        active = resolve_active_stages(all_stages=["a", "b", "c"], skip=[], only=[])
        assert active == {"a", "b", "c"}

    def test_skip_removes_stage(self) -> None:
        active = resolve_active_stages(all_stages=["a", "b", "c"], skip=["b"], only=[])
        assert active == {"a", "c"}

    def test_only_restricts_active(self) -> None:
        active = resolve_active_stages(all_stages=["a", "b", "c"], skip=[], only=["b"])
        assert active == {"b"}

    def test_skip_and_only_mutually_exclusive(self) -> None:
        with pytest.raises(typer.BadParameter, match="mutually exclusive"):
            resolve_active_stages(all_stages=["a", "b"], skip=["a"], only=["b"])

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Unknown stage"):
            resolve_active_stages(all_stages=["a", "b"], skip=["nonexistent"], only=[])


# ---------------------------------------------------------------------------
# _check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_passes_when_all_deps_met(self) -> None:
        stages = [
            CompositeStage(name="a", description="a", func=_always_succeed),
            CompositeStage(
                name="b",
                description="b",
                func=_always_succeed,
                depends_on=("a",),
            ),
        ]
        # Should not raise
        _check_dependencies(stages=stages, active={"a", "b"})

    def test_fails_when_dep_inactive(self) -> None:
        stages = [
            CompositeStage(name="a", description="a", func=_always_succeed),
            CompositeStage(
                name="b",
                description="b",
                func=_always_succeed,
                depends_on=("a",),
            ),
        ]
        with pytest.raises(typer.BadParameter, match="b: requires 'a'"):
            _check_dependencies(stages=stages, active={"b"})

    def test_fails_on_unknown_dep(self) -> None:
        stages = [
            CompositeStage(
                name="a",
                description="a",
                func=_always_succeed,
                depends_on=("nonexistent",),
            ),
        ]
        with pytest.raises(typer.BadParameter, match="unknown stage"):
            _check_dependencies(stages=stages, active={"a"})


# ---------------------------------------------------------------------------
# run_composite
# ---------------------------------------------------------------------------


class TestRunComposite:
    def test_runs_all_stages_in_order(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_always_succeed),
            CompositeStage(name="b", description="B", func=_always_succeed),
        ]
        summary = run_composite(name="test", stages=stages, active={"a", "b"})
        assert summary.succeeded == ["a", "b"]
        assert summary.overall_success

    def test_skips_inactive_stages(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_always_succeed),
            CompositeStage(name="b", description="B", func=_always_succeed),
        ]
        summary = run_composite(name="test", stages=stages, active={"a"})
        assert summary.succeeded == ["a"]
        assert summary.skipped == ["b"]

    def test_aborts_on_hard_failure(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_always_fail),
            CompositeStage(name="b", description="B", func=_always_succeed),
        ]
        summary = run_composite(name="test", stages=stages, active={"a", "b"})
        assert summary.failed == ["a"]
        assert "b" not in summary.succeeded
        assert not summary.overall_success

    def test_continues_on_soft_failure(self) -> None:
        stages = [
            CompositeStage(
                name="a",
                description="A",
                func=_always_fail,
                soft_fail=True,
            ),
            CompositeStage(name="b", description="B", func=_always_succeed),
        ]
        summary = run_composite(name="test", stages=stages, active={"a", "b"})
        assert summary.soft_failed == ["a"]
        assert summary.succeeded == ["b"]
        assert summary.overall_success

    def test_propagates_exception_on_hard_failure(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_always_raise),
        ]
        with pytest.raises(RuntimeError, match="boom"):
            run_composite(name="test", stages=stages, active={"a"})

    def test_soft_fails_catch_exceptions(self) -> None:
        stages = [
            CompositeStage(
                name="a",
                description="A",
                func=_always_raise,
                soft_fail=True,
            ),
        ]
        summary = run_composite(name="test", stages=stages, active={"a"})
        assert summary.soft_failed == ["a"]
        assert len(summary.warnings) == 1
        assert "RuntimeError" in summary.warnings[0]

    def test_strict_mode_converts_soft_to_hard(self) -> None:
        stages = [
            CompositeStage(
                name="a",
                description="A",
                func=_always_raise,
                soft_fail=True,
            ),
        ]
        with pytest.raises(RuntimeError, match="boom"):
            run_composite(name="test", stages=stages, active={"a"}, strict=True)

    def test_collects_artifacts(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_write_artifact),
        ]
        summary = run_composite(name="test", stages=stages, active={"a"})
        assert summary.artifacts == [Path("/tmp/fake.txt")]

    def test_collects_warnings_from_successful_stages(self) -> None:
        stages = [
            CompositeStage(name="a", description="A", func=_write_warning),
        ]
        summary = run_composite(name="test", stages=stages, active={"a"})
        assert summary.warnings == ["deprecation: use --new-flag"]
        assert summary.overall_success

    def test_context_shared_across_stages(self) -> None:
        stages = [
            CompositeStage(name="set", description="Set", func=_set_ctx_value),
            CompositeStage(
                name="read",
                description="Read",
                func=_read_ctx_value,
                depends_on=("set",),
            ),
        ]
        ctx: dict[str, Any] = {}
        summary = run_composite(
            name="test",
            stages=stages,
            active={"set", "read"},
            context=ctx,
        )
        assert summary.succeeded == ["set", "read"]
        assert ctx["upstream_value"] == 42


# ---------------------------------------------------------------------------
# render_composite_summary
# ---------------------------------------------------------------------------


class TestRenderCompositeSummary:
    def test_renders_basic_summary(self, capsys: pytest.CaptureFixture) -> None:
        summary = CompositeSummary(
            name="test-composite",
            succeeded=["a", "b"],
        )
        render_composite_summary(summary)
        captured = capsys.readouterr()
        assert "test-composite summary" in captured.out
        assert "2 stages completed" in captured.out

    def test_renders_skipped_stages(self, capsys: pytest.CaptureFixture) -> None:
        summary = CompositeSummary(
            name="test",
            succeeded=["a"],
            skipped=["b", "c"],
        )
        render_composite_summary(summary)
        captured = capsys.readouterr()
        assert "2 skipped (b, c)" in captured.out

    def test_renders_artifacts(self, capsys: pytest.CaptureFixture) -> None:
        summary = CompositeSummary(
            name="test",
            succeeded=["a"],
            artifacts=[Path("/tmp/x"), Path("/tmp/y")],
        )
        render_composite_summary(summary)
        captured = capsys.readouterr()
        assert "Artifacts written:" in captured.out
        assert "/tmp/x" in captured.out
        assert "/tmp/y" in captured.out

    def test_renders_warnings(self, capsys: pytest.CaptureFixture) -> None:
        summary = CompositeSummary(
            name="test",
            succeeded=["a"],
            warnings=["something to know about"],
        )
        render_composite_summary(summary)
        captured = capsys.readouterr()
        assert "Warnings:" in captured.out
        assert "something to know about" in captured.out

    def test_renders_failures(self, capsys: pytest.CaptureFixture) -> None:
        summary = CompositeSummary(
            name="test",
            succeeded=["a"],
            failed=["b"],
        )
        render_composite_summary(summary)
        captured = capsys.readouterr()
        assert "1 failed (b)" in captured.out
        assert not summary.overall_success


class TestResolveWinProbModelType:
    """Cover win-probability model-type resolution."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def test_returns_value_verbatim_when_not_auto(self) -> None:
        from gridiron_edge.cli._composites import resolve_win_prob_model_type

        assert resolve_win_prob_model_type("random_forest") == "random_forest"
        assert resolve_win_prob_model_type("xgboost") == "xgboost"
        assert resolve_win_prob_model_type("elo") == "elo"

    def test_resolves_auto_from_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from gridiron_edge.cli._composites import resolve_win_prob_model_type

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:00:00",
                    "source_run_id": "RUN_X",
                    "metrics": {"brier": 0.213},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )

        assert resolve_win_prob_model_type("auto") == "random_forest"

    def test_auto_raises_bad_parameter_when_manifest_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import typer

        from gridiron_edge.cli._composites import resolve_win_prob_model_type

        # tmp_path has no manifest at data/output/champions/champions.json
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )

        with pytest.raises(typer.BadParameter, match="requires a champion manifest"):
            resolve_win_prob_model_type("auto")

    def test_auto_raises_bad_parameter_when_win_prob_not_in_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        import typer

        from gridiron_edge.cli._composites import resolve_win_prob_model_type

        # Manifest exists but has no win_prob entry (e.g., only prop models).
        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "qb_pass_yards": {
                    "model_type": "elasticnet",
                    "promoted_at": "2026-07-01T14:10:00",
                    "source_run_id": "RUN_X",
                    "metrics": {"mae": 63.4},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )

        with pytest.raises(typer.BadParameter, match="requires a champion manifest"):
            resolve_win_prob_model_type("auto")
