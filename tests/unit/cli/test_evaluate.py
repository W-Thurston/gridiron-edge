# tests/unit/cli/test_evaluate.py

"""Tests for evaluate CLI champion-manifest persistence.

This module focuses on the explicit manifest-write command surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from gridiron_edge.evaluation.backfill import (
    BackfillMode,
    BackfillResult,
    BackfillSeasonResult,
    BackfillSeasonStatus,
)


class TestSelectModelWriteManifestFlag:
    """Cover the --write-manifest flag on evaluate select-model."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _stub_ranking_upstream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stub the pieces of evaluate_select_model that produce the terminal
        ranking table. Not in scope for this test; we care about the
        --write-manifest side effect.
        """
        monkeypatch.setattr(
            "gridiron_edge.cli.evaluate._collect_latest_forecast_run_metrics",
            lambda names, *, repo: [
                {
                    "model_key": "win_prob_random_forest",
                    "n_games": 100,
                    "brier": 0.213,
                    "ece": 0.041,
                    "auc": 0.721,
                    "accuracy": 0.65,
                    "log_loss": 0.61,
                },
            ],
        )

    def test_flag_writes_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        self._stub_ranking_upstream(monkeypatch)

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }
        monkeypatch.setattr(
            "gridiron_edge.evaluation.select.latest_backfilled_run_id",
            lambda model_name, model_type, *, repo: "fake-run",
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions_from_runs",
            lambda pairs, *, backfill_run_ids, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        runner = CliRunner()
        result = runner.invoke(
            evaluate_app,
            ["select-model", "--write-manifest"],
        )

        assert result.exit_code == 0, result.output

        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["models"]["win_prob"]["model_type"] == "random_forest"
        assert "Manifest written:" in result.output

    def test_flag_preserves_existing_families(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        self._stub_ranking_upstream(monkeypatch)

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        prior_manifest = {
            "schema_version": 1,
            "updated_at": "2026-06-01T00:00:00+00:00",
            "models": {
                "rb_rush_yards": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-06-01T00:00:00",
                    "source_run_id": "OLD_RUN",
                    "metrics": {"mae": 25.0, "rmse": 32.0, "r2": 0.17, "coverage": 0.91},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(prior_manifest))

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }
        monkeypatch.setattr(
            "gridiron_edge.evaluation.select.latest_backfilled_run_id",
            lambda model_name, model_type, *, repo: "fake-run",
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions_from_runs",
            lambda pairs, *, backfill_run_ids, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        runner = CliRunner()
        result = runner.invoke(
            evaluate_app,
            ["select-model", "--write-manifest"],
        )

        assert result.exit_code == 0, result.output

        manifest = json.loads((manifest_dir / "champions.json").read_text())
        assert set(manifest["models"].keys()) == {"win_prob", "rb_rush_yards"}
        assert manifest["models"]["rb_rush_yards"]["source_run_id"] == "OLD_RUN"

    def test_no_flag_does_not_write_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        self._stub_ranking_upstream(monkeypatch)

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        runner = CliRunner()
        result = runner.invoke(evaluate_app, ["select-model"])

        assert result.exit_code == 0, result.output
        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        assert not manifest_path.exists()
        assert "Manifest written:" not in result.output


class TestPruneChampionsCommand:
    """Cover the evaluate prune-champions command."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _write_manifest(self, tmp_path: Path, models: dict) -> Path:
        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-08-05T12:51:53+00:00",
            "models": models,
        }
        path = manifest_dir / "champions.json"
        path.write_text(json.dumps(manifest))
        return path

    def _write_artifact(self, tmp_path: Path, model_name: str, model_type: str) -> None:
        artifact_dir = tmp_path / "data" / "models" / model_name / model_type
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "metadata.json").write_text("{}")

    def test_removes_orphaned_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        manifest_path = self._write_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "logistic",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "RUN1",
                    "metrics": {"brier": 0.213},
                },
                "qb_rush_yards": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-08-05T12:45:55Z",
                    "source_run_id": "RUN1",
                    "metrics": {"mae": 16.5},
                },
            },
        )
        self._write_artifact(tmp_path, "win_prob", "logistic")

        runner = CliRunner()
        result = runner.invoke(evaluate_app, ["prune-champions"])

        assert result.exit_code == 0, result.output
        assert "removed: qb_rush_yards (random_forest)" in result.output

        manifest = json.loads(manifest_path.read_text())
        assert set(manifest["models"]) == {"win_prob"}

    def test_noop_message_when_nothing_orphaned(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        self._write_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "logistic",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "RUN1",
                    "metrics": {"brier": 0.213},
                },
            },
        )
        self._write_artifact(tmp_path, "win_prob", "logistic")

        runner = CliRunner()
        result = runner.invoke(evaluate_app, ["prune-champions"])

        assert result.exit_code == 0, result.output
        assert "nothing to prune" in result.output


class TestEvaluateBackfill:
    @staticmethod
    def _result(*, generated: bool = True) -> BackfillResult:
        seasons = (
            (
                BackfillSeasonResult(
                    season="2024-2025",
                    status=BackfillSeasonStatus.PREDICTED,
                    generated_count=2,
                ),
            )
            if generated
            else ()
        )
        return BackfillResult(
            model_name="win_prob",
            model_type="random_forest",
            mode=BackfillMode.WALK_FORWARD,
            run_id="run-1" if generated else None,
            generated_count=2 if generated else 0,
            inserted_count=2 if generated else 0,
            existing_count=0,
            seasons=seasons,
        )

    @patch("gridiron_edge.evaluation.backfill.backfill_model")
    def test_renders_structured_result(self, backfill_model) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        backfill_model.return_value = self._result()
        result = CliRunner().invoke(
            evaluate_app,
            ["backfill", "--model-type", "random_forest"],
        )

        assert result.exit_code == 0, result.output
        assert "Mode: walk-forward" in result.output
        assert "Run ID: run-1" in result.output
        assert "Generated events: 2" in result.output
        assert "Inserted events: 2" in result.output
        assert "Predicted seasons: 2024-2025" in result.output

    @patch("gridiron_edge.evaluation.backfill.backfill_model")
    def test_zero_generation_is_successfully_visible(self, backfill_model) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        backfill_model.return_value = self._result(generated=False)
        result = CliRunner().invoke(
            evaluate_app,
            ["backfill", "--model-type", "random_forest"],
        )

        assert result.exit_code == 0, result.output
        assert "Run ID: none" in result.output
        assert "Generated events: 0" in result.output
        assert "Predicted seasons: none" in result.output

    @patch("gridiron_edge.evaluation.backfill.backfill_model")
    def test_invalid_mode_fails_before_service(self, backfill_model) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        result = CliRunner().invoke(
            evaluate_app,
            ["backfill", "--mode", "nonsense"],
        )

        assert result.exit_code != 0
        backfill_model.assert_not_called()

    @patch("gridiron_edge.evaluation.backfill.backfill_model")
    def test_validation_error_is_clean_cli_failure(self, backfill_model) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        backfill_model.side_effect = ValueError(
            "start_season must use consecutive years, got '2025-2027'."
        )
        result = CliRunner().invoke(
            evaluate_app,
            [
                "backfill",
                "--model-type",
                "random_forest",
                "--mode",
                "walk-forward",
                "--start-season",
                "2025-2027",
            ],
        )

        assert result.exit_code != 0
        assert "must use consecutive years" in result.output
        assert "Traceback" not in result.output


class TestModelReportCommand:
    """Cover the evaluate model-report command."""

    _ARGS: ClassVar[list[str]] = [
        "model-report",
        "--win-elo-run",
        "run-elo",
        "--win-logistic-run",
        "run-logistic",
        "--win-random-forest-run",
        "run-rf-win",
        "--win-xgboost-run",
        "run-xgb-win",
        "--total-random-forest-run",
        "run-rf-total",
        "--total-xgboost-run",
        "run-xgb-total",
    ]

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _fake_result(self):
        from types import SimpleNamespace

        report = SimpleNamespace(
            report_id="a" * 64,
            game_count=4,
            win_metrics=[
                SimpleNamespace(
                    model_type="elo",
                    brier=0.22,
                    ece=0.02,
                    auc=0.65,
                    calibration_slope=0.9,
                    calibration_intercept=0.01,
                    sharpness=0.03,
                    season_stability=0.01,
                ),
            ],
            total_metrics=[
                SimpleNamespace(
                    model_type="random_forest",
                    mae=10.0,
                    median_absolute_error=9.0,
                    rmse=13.0,
                    actual_coverage=0.9,
                    nominal_coverage=0.9,
                ),
            ],
        )
        return SimpleNamespace(report=report)

    def test_builds_prints_and_selects_the_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )
        fake_result = self._fake_result()
        with (
            patch(
                "gridiron_edge.evaluation.game_model_evaluation_report_builder."
                "build_and_write_game_model_evaluation_report",
                return_value=fake_result,
            ) as build,
            patch(
                "gridiron_edge.evaluation.game_model_evaluation_report_selection."
                "select_current_game_model_evaluation_report",
            ) as select,
        ):
            result = CliRunner().invoke(evaluate_app, self._ARGS)

        assert result.exit_code == 0, result.output
        assert "a" * 64 in result.output
        assert "Selected as current" in result.output
        build.assert_called_once()
        assert build.call_args.kwargs["win_run_ids"] == {
            "elo": "run-elo",
            "logistic": "run-logistic",
            "random_forest": "run-rf-win",
            "xgboost": "run-xgb-win",
        }
        assert build.call_args.kwargs["total_run_ids"] == {
            "random_forest": "run-rf-total",
            "xgboost": "run-xgb-total",
        }
        select.assert_called_once()
        assert select.call_args.args[0] == "a" * 64

    def test_no_select_flag_skips_selection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )
        with (
            patch(
                "gridiron_edge.evaluation.game_model_evaluation_report_builder."
                "build_and_write_game_model_evaluation_report",
                return_value=self._fake_result(),
            ),
            patch(
                "gridiron_edge.evaluation.game_model_evaluation_report_selection."
                "select_current_game_model_evaluation_report",
            ) as select,
        ):
            result = CliRunner().invoke(evaluate_app, [*self._ARGS, "--no-select"])

        assert result.exit_code == 0, result.output
        select.assert_not_called()
        assert "Selected as current" not in result.output

    def test_mismatched_game_set_is_a_clean_cli_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )
        with patch(
            "gridiron_edge.evaluation.game_model_evaluation_report_builder."
            "build_and_write_game_model_evaluation_report",
            side_effect=ValueError("do not share a common game set"),
        ):
            result = CliRunner().invoke(evaluate_app, self._ARGS)

        assert result.exit_code != 0
        assert "do not share a common game set" in result.output
        assert "Traceback" not in result.output


class TestExplainLogisticCommand:
    """Cover the evaluate explain-logistic command."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _fake_result(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            batch=SimpleNamespace(batch_id="a" * 64, evidence_id="b" * 64),
            event_count=16,
            max_reconciliation_error=1.2e-12,
        )

    def test_builds_and_persists_the_batch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )
        with patch(
            "gridiron_edge.evaluation.logistic_explanation_evidence_builder."
            "build_and_write_logistic_explanation_batch",
            return_value=self._fake_result(),
        ) as build:
            result = CliRunner().invoke(evaluate_app, ["explain-logistic", "--run-id", "run-1"])

        assert result.exit_code == 0, result.output
        assert "a" * 64 in result.output
        assert "16" in result.output
        build.assert_called_once()
        assert build.call_args.kwargs["run_id"] == "run-1"

    def test_value_error_is_a_clean_cli_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.evaluate import evaluate_app

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )
        with patch(
            "gridiron_edge.evaluation.logistic_explanation_evidence_builder."
            "build_and_write_logistic_explanation_batch",
            side_effect=ValueError("No win_prob/logistic persisted-estimator evidence found"),
        ):
            result = CliRunner().invoke(evaluate_app, ["explain-logistic", "--run-id", "run-1"])

        assert result.exit_code != 0
        assert "No win_prob/logistic persisted-estimator evidence found" in result.output
        assert "Traceback" not in result.output


class TestWeeklyRecap:
    """Cover `evaluate weekly-recap`: refusals, the dry run, writing, selection and replay."""

    @pytest.fixture
    def repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from types import SimpleNamespace

        from tests.fixtures.weekly_recap import recap_week

        week = recap_week()
        raw = tmp_path / "data/raw/NFL_wk_by_wk_nflverse.parquet"
        raw.parent.mkdir(parents=True)
        week.schedule.to_parquet(raw, index=False)
        logs = tmp_path / "data/cleaned/player_game_logs.parquet"
        logs.parent.mkdir(parents=True)
        week.logs.to_parquet(logs, index=False)
        settings = SimpleNamespace(repo_root=tmp_path)
        monkeypatch.setattr("gridiron_edge.core.settings.get_settings", lambda: settings)
        monkeypatch.setattr(
            "gridiron_edge.datasets.loaders.load_current_weekly_product",
            lambda repo, *, season, week: self.product,
        )
        monkeypatch.setattr("gridiron_edge.datasets.loaders.load_games", lambda repo: self.games)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            lambda *, repo, season, week: week_events,
        )
        week_events = week.events
        self.product = week.product
        self.games = week.games
        return tmp_path

    def _invoke(self, *extra: str):
        from gridiron_edge.cli.evaluate import evaluate_app

        return CliRunner().invoke(
            evaluate_app,
            ["weekly-recap", "--season", "2026-2027", "--week", "4", *extra],
        )

    def _recap_root(self, repo: Path) -> Path:
        return repo / "data/output/weekly_recaps"

    def test_refuses_a_week_with_an_unplayed_game(self, repo: Path) -> None:
        import pandas as pd

        raw = repo / "data/raw/NFL_wk_by_wk_nflverse.parquet"
        schedule = pd.read_parquet(raw)
        schedule.loc[0, ["away_score", "home_score", "result"]] = None
        schedule.to_parquet(raw, index=False)

        result = self._invoke()

        assert result.exit_code == 1
        assert "1 of 4 games have no result yet: 2026_04_AAA_BBB" in result.output
        assert not self._recap_root(repo).exists()

    def test_refuses_a_week_without_a_selected_product(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(repo: Path, *, season: str, week: int) -> None:
            raise FileNotFoundError("none")

        monkeypatch.setattr("gridiron_edge.datasets.loaders.load_current_weekly_product", missing)

        result = self._invoke()

        assert result.exit_code == 1
        assert "no weekly product is selected" in result.output

    def test_refuses_a_score_conflict(self, repo: Path) -> None:
        self.games.loc[self.games["GAME_ID"] == "2026_04_AAA_BBB", "HOME_SCORE"] = 99.0

        result = self._invoke()

        assert result.exit_code == 1
        assert "disagree on scores: 2026_04_AAA_BBB" in result.output

    def test_dry_run_prints_the_recap_and_writes_nothing(self, repo: Path) -> None:
        result = self._invoke("--dry-run")

        assert result.exit_code == 0, result.output
        assert "Brier" in result.output
        assert "not in product: 2026_04_EEE_FFF" in result.output
        # No evidence names a sidecar here, so the carried game reads not recorded.
        assert "starting QBs: 2 of 8 slots scored (0 from a sidecar capture)" in result.output
        assert "not recorded 2, not in product 2, unresolved 1, actual missing 1" in result.output
        assert "dry run, nothing written" in result.output
        assert not self._recap_root(repo).exists()

    def test_writes_selects_then_replays(self, repo: Path) -> None:
        import json

        first = self._invoke()
        assert first.exit_code == 0, first.output
        assert "previous selection: none" in first.output
        assert "new: recap" in first.output
        current = self._recap_root(repo) / "current.json"
        selection = json.loads(current.read_text())["selections"]["2026-2027_week_04"]
        stored = {
            path: path.read_bytes() for path in self._recap_root(repo).rglob("*") if path.is_file()
        }

        second = self._invoke()

        assert second.exit_code == 0, second.output
        assert f"replayed: recap {selection['recap_id']}" in second.output
        assert {
            path: path.read_bytes() for path in self._recap_root(repo).rglob("*") if path.is_file()
        } == stored

    def test_reselects_a_stored_recap_that_is_not_current(self, repo: Path) -> None:
        first = self._invoke()
        self.games.loc[self.games["GAME_ID"] == "2026_04_EEE_FFF", "VEGAS_LINE"] = -2.0
        changed = self._invoke()
        self.games.loc[self.games["GAME_ID"] == "2026_04_EEE_FFF", "VEGAS_LINE"] = -1.0

        again = self._invoke()

        assert "new: recap" in first.output
        assert "new: recap" in changed.output
        assert "reselected: recap" in again.output
