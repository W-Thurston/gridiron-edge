# tests/unit/cli/test_normal_workflows_exclude_draftkings.py
"""Verify normal orchestration excludes the legacy DraftKings adapter."""

from __future__ import annotations

import inspect

from gridiron_edge.cli.main import ALL_STAGES, _run_pipeline_stages
from gridiron_edge.cli.weekly_predict import _ALL_STAGES, _build_stages


def test_data_pipeline_has_no_external_odds_stage() -> None:
    assert "fetch-odds" not in ALL_STAGES


def test_data_pipeline_does_not_import_or_invoke_draftkings() -> None:
    source = inspect.getsource(_run_pipeline_stages)
    assert "fetch_dk_odds" not in source
    assert "DraftKings" not in source


def test_weekly_predict_has_no_external_odds_stage() -> None:
    assert "fetch-odds" not in _ALL_STAGES
    assert [stage.name for stage in _build_stages()] == [
        "ensure-data-fresh",
        "predict-week",
        "render-outputs",
        "generate-edges",
    ]


def test_weekly_edge_stage_uses_existing_market_snapshot() -> None:
    stages = {stage.name: stage for stage in _build_stages()}
    assert stages["generate-edges"].depends_on == ("predict-week",)
    assert stages["generate-edges"].soft_fail is True
