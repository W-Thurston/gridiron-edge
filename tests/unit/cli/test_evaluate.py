# tests/unit/cli/test_evaluate.py

"""Tests for the evaluate CLI, focused on W13 --write-manifest additions.

Broader evaluate-command coverage is out of scope for W13 Tier 2. This
file exercises only the new manifest-write flag introduced in Step 7.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


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
            "gridiron_edge.cli.evaluate._collect_model_metrics",
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
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
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
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
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
