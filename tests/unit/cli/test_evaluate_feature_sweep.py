# tests/unit/cli/test_evaluate_feature_sweep.py

"""Wiring tests for the feature-sweep and feature-selection commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.evaluate import evaluate_app

runner = CliRunner()


@dataclass
class _Settings:
    repo_root: Path


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings", lambda: _Settings(repo_root=tmp_path)
    )
    monkeypatch.setattr("gridiron_edge.datasets.loaders.load_modeling_file", lambda _repo: object())


@pytest.mark.unit
def test_commands_are_registered() -> None:
    names = {c.name for c in evaluate_app.registered_commands}
    assert {"feature-sweep", "feature-selection"} <= names


@pytest.mark.unit
def test_feature_sweep_report_only_requires_checkpoints(tmp_path: Path) -> None:
    result = runner.invoke(
        evaluate_app,
        ["feature-sweep", "--checkpoint-dir", str(tmp_path / "none"), "--report-only"],
    )
    assert result.exit_code != 0
    assert "No checkpoints found" in result.output


@pytest.mark.unit
def test_feature_selection_requires_a_completed_sweep(tmp_path: Path) -> None:
    result = runner.invoke(
        evaluate_app, ["feature-selection", "--checkpoint-dir", str(tmp_path / "none")]
    )
    assert result.exit_code != 0
    assert "No sweep checkpoints found" in result.output
