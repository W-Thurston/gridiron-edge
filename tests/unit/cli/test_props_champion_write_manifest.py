# tests/unit/cli/test_props_champion_write_manifest.py

"""Tests for `gridiron props champion --write-manifest` (W13 Step 8).

Broader props CLI coverage is out of scope for W13 Tier 2. This file
exercises only the new --write-manifest flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner


class TestPropsChampionWriteManifestFlag:
    """Cover the --write-manifest flag on props champion."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _stub_champion_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub build_prop_evaluation_df + select_prop_champion so the
        champion_cmd terminal-output loop completes without needing a
        real archive. The manifest-write path is what we care about here.
        """
        # Non-empty eval DF so the display loop finds "archive rows".
        eval_df = pd.DataFrame(
            {
                "actual": [200.0] * 5,
                "predicted_mean": [210.0] * 5,
                "predicted_std": [40.0] * 5,
                "lo_90": [140.0] * 5,
                "hi_90": [280.0] * 5,
            }
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.prop_archive.build_prop_evaluation_df",
            lambda **kwargs: eval_df,
        )

        # Minimal report shape; the display path calls .accuracy.mae etc.
        @dataclass
        class FakeAccuracy:
            mae: float = 63.4
            rmse: float = 80.6
            r2: float = 0.05
            median_ae: float = 51.8
            n: int = 100

        @dataclass
        class FakeCoverage:
            actual_coverage: float = 0.938
            nominal_coverage: float = 0.90
            mean_interval_width: float = 140.0

        @dataclass
        class FakeBias:
            mean_error: float = 9.7
            pct_over_predicted: float = 0.528

        @dataclass
        class FakeReport:
            accuracy: FakeAccuracy
            coverage: FakeCoverage | None
            bias: FakeBias

        monkeypatch.setattr(
            "gridiron_edge.evaluation.prop_metrics.evaluate_prop_model",
            lambda **kwargs: FakeReport(
                accuracy=FakeAccuracy(),
                coverage=FakeCoverage(),
                bias=FakeBias(),
            ),
        )

    def test_flag_writes_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.props import props_app

        self._stub_champion_display(monkeypatch)

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        # Stub the three selectors used by write_champion_manifest.
        prop_result = {
            "qb_pass_yards": {
                "model_type": "elasticnet",
                "promoted_at": "2026-07-01T14:10:00",
                "metrics": {
                    "mae": 63.4,
                    "rmse": 80.6,
                    "r2": 0.05,
                    "coverage": 0.938,
                },
            },
        }
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: prop_result,
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            ["champion", "--model", "qb_pass_yards", "--write-manifest"],
        )

        assert result.exit_code == 0, result.output

        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["models"]["qb_pass_yards"]["model_type"] == "elasticnet"
        assert "Manifest written:" in result.output

    def test_flag_preserves_existing_families(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.props import props_app

        self._stub_champion_display(monkeypatch)

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
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-06-01T00:00:00",
                    "source_run_id": "OLD_RUN",
                    "metrics": {"brier": 0.22, "ece": 0.05, "auc": 0.70},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(prior_manifest))

        prop_result = {
            "qb_pass_yards": {
                "model_type": "elasticnet",
                "promoted_at": "2026-07-01T14:10:00",
                "metrics": {
                    "mae": 63.4,
                    "rmse": 80.6,
                    "r2": 0.05,
                    "coverage": 0.938,
                },
            },
        }
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: prop_result,
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            ["champion", "--model", "qb_pass_yards", "--write-manifest"],
        )

        assert result.exit_code == 0, result.output

        manifest = json.loads((manifest_dir / "champions.json").read_text())
        assert set(manifest["models"].keys()) == {"win_prob", "qb_pass_yards"}
        assert manifest["models"]["win_prob"]["source_run_id"] == "OLD_RUN"

    def test_no_flag_does_not_write_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.props import props_app

        self._stub_champion_display(monkeypatch)

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            self._fake_settings(tmp_path),
        )

        runner = CliRunner()
        result = runner.invoke(
            props_app,
            ["champion", "--model", "qb_pass_yards"],
        )

        assert result.exit_code == 0, result.output
        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        assert not manifest_path.exists()
        assert "Manifest written:" not in result.output
