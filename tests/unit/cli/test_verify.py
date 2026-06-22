# tests/unit/cli/test_verify.py

"""Tests for the verify composite command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.verify import (
    _build_stages,
    _summarize_pytest_output,
)


class TestStageBuilder:
    """Verify _build_stages produces the right shape for each mode."""

    def test_default_mode_includes_smoke_and_e2e(self) -> None:
        stages = _build_stages(fast=False, very_thorough=False)
        names = [s.name for s in stages]
        assert "e2e-tests" in names
        assert "smoke-pipeline" in names
        assert "slow-tests" not in names

    def test_fast_mode_excludes_smoke_and_e2e(self) -> None:
        stages = _build_stages(fast=True, very_thorough=False)
        names = [s.name for s in stages]
        assert "e2e-tests" not in names
        assert "smoke-pipeline" not in names

    def test_very_thorough_includes_slow_tests(self) -> None:
        stages = _build_stages(fast=False, very_thorough=True)
        names = [s.name for s in stages]
        assert "slow-tests" in names

    def test_baseline_comparison_always_present(self) -> None:
        for fast in (True, False):
            for very_thorough in (True, False):
                stages = _build_stages(fast=fast, very_thorough=very_thorough)
                names = [s.name for s in stages]
                assert "baseline-comparison" in names, (
                    f"baseline-comparison missing for fast={fast}, very_thorough={very_thorough}"
                )

    def test_smoke_pipeline_is_soft_fail(self) -> None:
        stages = {s.name: s for s in _build_stages(fast=False, very_thorough=False)}
        assert stages["smoke-pipeline"].soft_fail is True

    def test_baseline_comparison_is_soft_fail(self) -> None:
        stages = {s.name: s for s in _build_stages(fast=False, very_thorough=False)}
        assert stages["baseline-comparison"].soft_fail is True

    def test_quality_gates_is_hard_fail(self) -> None:
        stages = {s.name: s for s in _build_stages(fast=False, very_thorough=False)}
        assert stages["quality-gates"].soft_fail is False


class TestSummarizePytestOutput:
    """Cover the pytest output parser."""

    def test_handles_passed_summary(self) -> None:
        output = "===== 542 passed in 12.3s ====="
        result = _summarize_pytest_output(output)
        assert "542 passed" in result

    def test_handles_failed_summary(self) -> None:
        output = "===== 3 failed, 539 passed in 12.3s ====="
        result = _summarize_pytest_output(output)
        assert "3 failed" in result

    def test_handles_error_summary(self) -> None:
        output = "===== 2 errors in 4.5s ====="
        result = _summarize_pytest_output(output)
        assert "2 errors" in result

    def test_handles_empty_output(self) -> None:
        result = _summarize_pytest_output("")
        assert "no summary" in result


class TestSubprocessStages:
    """Cover the subprocess-invoking stages."""

    @patch("gridiron_edge.cli.verify._run_subprocess")
    def test_quality_gates_passes_when_both_clean(self, mock_sub: MagicMock) -> None:
        from gridiron_edge.cli.verify import _stage_quality_gates

        # ruff success, pyrefly success
        mock_sub.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]
        ctx = {"repo_root": Path("/tmp")}
        result = _stage_quality_gates(ctx)
        assert result.success
        assert "clean" in result.detail

    @patch("gridiron_edge.cli.verify._run_subprocess")
    def test_quality_gates_fails_on_ruff(self, mock_sub: MagicMock) -> None:
        from gridiron_edge.cli.verify import _stage_quality_gates

        mock_sub.side_effect = [(1, "", "ruff error")]
        ctx = {"repo_root": Path("/tmp")}
        result = _stage_quality_gates(ctx)
        assert not result.success
        assert "ruff failed" in result.detail

    @patch("gridiron_edge.cli.verify._run_subprocess")
    def test_quality_gates_fails_on_pyrefly(self, mock_sub: MagicMock) -> None:
        from gridiron_edge.cli.verify import _stage_quality_gates

        # ruff success, pyrefly failure
        mock_sub.side_effect = [
            (0, "", ""),
            (1, "", "pyrefly error"),
        ]
        ctx = {"repo_root": Path("/tmp")}
        result = _stage_quality_gates(ctx)
        assert not result.success
        assert "pyrefly failed" in result.detail

    @patch("gridiron_edge.cli.verify._run_subprocess")
    def test_unit_tests_extracts_summary(self, mock_sub: MagicMock) -> None:
        from gridiron_edge.cli.verify import _stage_unit_tests

        mock_sub.return_value = (
            0,
            "===== 542 passed in 12.3s =====",
            "",
        )
        ctx = {"repo_root": Path("/tmp")}
        result = _stage_unit_tests(ctx)
        assert result.success
        assert "542 passed" in result.detail


class TestBaselineComparisonStage:
    """Cover the baseline-comparison stage."""

    def test_returns_failure_when_no_reports_dir(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.verify import _stage_baseline_comparison

        ctx = {"repo_root": tmp_path}
        result = _stage_baseline_comparison(ctx)
        assert not result.success
        assert "no full-retrain report directory" in result.detail

    def test_returns_failure_when_no_reports(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.verify import _stage_baseline_comparison

        (tmp_path / "data" / "output" / "reports").mkdir(parents=True)
        ctx = {"repo_root": tmp_path}
        result = _stage_baseline_comparison(ctx)
        assert not result.success
        assert "no full-retrain reports" in result.detail

    def test_returns_success_when_reports_exist(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.verify import _stage_baseline_comparison

        reports_dir = tmp_path / "data" / "output" / "reports"
        reports_dir.mkdir(parents=True)
        report_path = reports_dir / "full-retrain-2026-06-21.md"
        report_path.write_text("# Sample report")

        ctx = {"repo_root": tmp_path}
        result = _stage_baseline_comparison(ctx)
        assert result.success
        assert result.success
        assert report_path in result.artifacts

    def test_detects_metric_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """baseline-comparison should surface metric drift."""

        from dataclasses import dataclass

        from gridiron_edge.cli.verify import _stage_baseline_comparison

        reports_dir: Path = tmp_path / "data" / "output" / "reports"
        reports_dir.mkdir(parents=True)

        report_path: Path = reports_dir / "full-retrain-2026-06-21-120000.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Full Retrain Baseline Report",
                    "",
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_logistic | 0.2200 | 0.0100 | 0.7000 | - | - | - |",
                ]
            )
        )

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo_root: Path) -> None:
                self.repo_root = repo_root

            def is_trained(
                self,
                model_name: str,
                model_type: str,
            ) -> bool:
                return True

            def read_metadata(
                self,
                model_name: str,
                model_type: str,
            ) -> FakeMeta:
                return FakeMeta(
                    metrics={
                        "brier": 0.2300,  # drifted
                        "ece": 0.0100,
                        "auc": 0.7000,
                    }
                )

        monkeypatch.setattr(
            "gridiron_edge.cli.verify.ArtifactStore",
            FakeArtifactStore,
        )

        ctx: dict[str, Path] = {"repo_root": tmp_path}

        result: StageResult = _stage_baseline_comparison(ctx)

        assert result.success
        assert "differ from baseline" in result.detail
        assert result.warnings
        assert "win_prob_logistic:brier" in result.warnings[0]
        assert report_path in result.artifacts

    def test_detects_matching_baseline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """baseline-comparison should report matches when metrics agree."""

        from dataclasses import dataclass

        from gridiron_edge.cli.verify import _stage_baseline_comparison

        reports_dir: Path = tmp_path / "data" / "output" / "reports"
        reports_dir.mkdir(parents=True)

        report_path: Path = reports_dir / "full-retrain-2026-06-21-120000.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Full Retrain Baseline Report",
                    "",
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_logistic | 0.2200 | 0.0100 | 0.7000 | - | - | - |",
                ]
            )
        )

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo_root: Path) -> None:
                self.repo_root = repo_root

            def is_trained(
                self,
                model_name: str,
                model_type: str,
            ) -> bool:
                return True

            def read_metadata(
                self,
                model_name: str,
                model_type: str,
            ) -> FakeMeta:
                return FakeMeta(
                    metrics={
                        "brier": 0.2200,
                        "ece": 0.0100,
                        "auc": 0.7000,
                    }
                )

        monkeypatch.setattr(
            "gridiron_edge.cli.verify.ArtifactStore",
            FakeArtifactStore,
        )

        ctx: dict[str, Path] = {"repo_root": tmp_path}

        result: StageResult = _stage_baseline_comparison(ctx)

        assert result.success
        assert "match baseline" in result.detail
        assert result.artifacts
        assert report_path in result.artifacts


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.verify._stage_quality_gates")
    @patch("gridiron_edge.cli.verify._stage_unit_tests")
    @patch("gridiron_edge.cli.verify._stage_integration_tests")
    @patch("gridiron_edge.cli.verify._stage_e2e_tests")
    @patch("gridiron_edge.cli.verify._stage_smoke_pipeline")
    @patch("gridiron_edge.cli.verify._stage_baseline_comparison")
    def test_runs_all_default_stages(
        self,
        mock_baseline: MagicMock,
        mock_smoke: MagicMock,
        mock_e2e: MagicMock,
        mock_integ: MagicMock,
        mock_unit: MagicMock,
        mock_quality: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.verify import verify_cmd

        for m in [
            mock_quality,
            mock_unit,
            mock_integ,
            mock_e2e,
            mock_smoke,
            mock_baseline,
        ]:
            m.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(verify_cmd)

        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

    @patch("gridiron_edge.cli.verify._stage_quality_gates")
    @patch("gridiron_edge.cli.verify._stage_unit_tests")
    @patch("gridiron_edge.cli.verify._stage_integration_tests")
    @patch("gridiron_edge.cli.verify._stage_baseline_comparison")
    def test_fast_mode_skips_e2e_and_smoke(
        self,
        mock_baseline: MagicMock,
        mock_integ: MagicMock,
        mock_unit: MagicMock,
        mock_quality: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.verify import verify_cmd

        for m in [
            mock_quality,
            mock_unit,
            mock_integ,
            mock_baseline,
        ]:
            m.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(verify_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--fast"])
        assert result.exit_code == 0, result.output

    def test_only_filter_with_unknown_stage_for_mode(self) -> None:
        """--only e2e-tests in --fast mode should fail validation."""
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.verify import verify_cmd

        app = typer.Typer()
        app.command()(verify_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--fast", "--only", "e2e-tests"])
        assert result.exit_code != 0
        assert "Unknown stage" in result.output


class TestParseCompositeKey:
    """Tests for verify composite-key parsing."""

    def test_random_forest(self) -> None:
        from gridiron_edge.cli.verify import _parse_composite_key

        assert _parse_composite_key("win_prob_random_forest") == (
            "win_prob",
            "random_forest",
        )

    def test_xgboost(self) -> None:
        from gridiron_edge.cli.verify import _parse_composite_key

        assert _parse_composite_key("total_xgboost") == (
            "total",
            "xgboost",
        )

    def test_elasticnet(self) -> None:
        from gridiron_edge.cli.verify import _parse_composite_key

        assert _parse_composite_key("qb_pass_yards_elasticnet") == (
            "qb_pass_yards",
            "elasticnet",
        )

    def test_unknown_returns_none(self) -> None:
        from gridiron_edge.cli.verify import _parse_composite_key

        assert _parse_composite_key("not_a_real_model") is None
