# tests/unit/cli/test_edges.py

"""Tests for edges CLI, focused on W13 Tier 3 --model-type auto behavior.

Broader edges-command coverage lives in tests/integration/test_edges_cli.py.
This file exercises only the manifest-resolution path introduced in Step 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner


class TestReportModelTypeResolution:
    """Cover --model-type auto sentinel on `gridiron edges report`."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _stub_data_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub load_prediction_log / load_current_odds so the CLI exits
        cleanly at the "no predictions" branch without hitting real data.
        We only care about resolution behavior, not the report contents."""
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

    def test_explicit_model_type_passes_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            edges_app,
            [
                "report",
                "--week",
                "1",
                "--season",
                "2026-2027",
                "--model-type",
                "xgboost",
            ],
        )

        # Exits at "no predictions found" — the branch we control.
        # The important assertion: the header shows the resolved value.
        assert "model=xgboost" in result.output
        assert "win_prob/xgboost" in result.output

    def test_auto_resolves_from_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

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
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            edges_app,
            [
                "report",
                "--week",
                "1",
                "--season",
                "2026-2027",
            ],
        )

        assert "model=random_forest" in result.output
        assert "win_prob/random_forest" in result.output

    def test_auto_fails_when_manifest_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            edges_app,
            [
                "report",
                "--week",
                "1",
                "--season",
                "2026-2027",
            ],
        )

        assert result.exit_code != 0
        assert "requires a champion manifest" in result.output


class TestClvModelTypeResolution:
    """Cover --model-type auto sentinel on `gridiron edges clv`."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _stub_data_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

    def test_explicit_model_type_passes_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            edges_app,
            [
                "clv",
                "--model-type",
                "xgboost",
            ],
        )

        assert "model=xgboost" in result.output
        assert "win_prob/xgboost" in result.output

    def test_auto_resolves_from_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

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
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(edges_app, ["clv"])

        assert "model=random_forest" in result.output
        assert "win_prob/random_forest" in result.output

    def test_auto_fails_when_manifest_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(edges_app, ["clv"])

        assert result.exit_code != 0
        assert "requires a champion manifest" in result.output
