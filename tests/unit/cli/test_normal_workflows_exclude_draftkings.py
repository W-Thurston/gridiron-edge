# tests/unit/cli/test_normal_workflows_exclude_draftkings.py
"""Verify normal orchestration excludes the legacy DraftKings adapter."""

from __future__ import annotations

import inspect
from pathlib import Path

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
        "compose-weekly-product",
        "render-outputs",
        "generate-edges",
    ]


def test_weekly_edge_stage_uses_selected_weekly_product() -> None:
    stages = {stage.name: stage for stage in _build_stages()}

    assert stages["compose-weekly-product"].depends_on == ("predict-week",)
    assert stages["generate-edges"].depends_on == ("compose-weekly-product",)


def test_weekly_edge_stage_has_no_direct_market_or_archive_dependencies() -> None:
    module_path = Path("src/gridiron_edge/cli/weekly_predict.py")
    source = module_path.read_text()

    retired_dependencies = (
        "load_prediction_log",
        "load_current_odds",
        "get_margin_std",
        "get_total_std",
        "build_edge_report",
        "rank_edges",
    )

    found = [name for name in retired_dependencies if name in source]

    assert found == []
