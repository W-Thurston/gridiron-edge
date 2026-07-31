# tests/unit/cli/test_edges.py

"""Tests for edges CLI model resolution retained by historical CLV."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner


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
