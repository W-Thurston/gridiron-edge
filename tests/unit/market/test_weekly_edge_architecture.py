# tests/unit/market/test_weekly_edge_architecture.py

"""Architecture guards for current-week edge consumers."""

from __future__ import annotations

from pathlib import Path

_CURRENT_WEEK_CONSUMERS = (
    Path("src/gridiron_edge/cli/edges.py"),
    Path("src/gridiron_edge/cli/weekly_predict.py"),
    Path("src/gridiron_edge/cli/verify_week.py"),
    Path("src/gridiron_edge/api/loaders.py"),
)

_RETIRED_CURRENT_WEEK_DEPENDENCIES = (
    "load_prediction_log",
    "get_margin_std",
    "get_total_std",
    "build_edge_report",
    "rank_edges",
)


def test_current_week_consumers_use_weekly_edge_service() -> None:
    expected_functions = {
        "src/gridiron_edge/cli/edges.py": "def report(",
        "src/gridiron_edge/cli/weekly_predict.py": ("def _stage_generate_edges("),
        "src/gridiron_edge/cli/verify_week.py": ("def _load_edge_result("),
        "src/gridiron_edge/api/loaders.py": ("def load_edges_for_week("),
    }

    for path in _CURRENT_WEEK_CONSUMERS:
        source = path.read_text()
        assert expected_functions[str(path)] in source
        assert "build_weekly_edge_result" in source


def test_weekly_predict_has_no_retired_edge_dependencies() -> None:
    source = Path("src/gridiron_edge/cli/weekly_predict.py").read_text()

    found = [name for name in _RETIRED_CURRENT_WEEK_DEPENDENCIES if name in source]

    assert found == []


def test_verify_week_has_no_direct_recommendation_dependencies() -> None:
    source = Path("src/gridiron_edge/cli/verify_week.py").read_text()

    retired = (
        "build_edge_report",
        "get_margin_std",
        "get_total_std",
    )
    found = [name for name in retired if name in source]

    assert found == []


def test_api_edge_loader_has_no_retired_edge_dependencies() -> None:
    source = Path("src/gridiron_edge/api/loaders.py").read_text()

    start = source.index("def load_edges_for_week(")
    end = source.index(
        "\ndef _parse_season_int(",
        start,
    )
    function_source = source[start:end]

    retired = (
        "resolve_current_champion",
        "load_prediction_log",
        "load_current_odds",
        "get_margin_std",
        "get_total_std",
        "build_edge_report",
        "rank_edges",
    )
    found = [name for name in retired if name in function_source]

    assert found == []


def test_standalone_report_uses_service_while_clv_remains_historical() -> None:
    source = Path("src/gridiron_edge/cli/edges.py").read_text()

    report_start = source.index("def report(")
    clv_start = source.index("def clv(")
    report_source = source[report_start:clv_start]
    clv_source = source[clv_start:]

    assert "build_weekly_edge_result" in report_source
    assert "build_edge_report" not in report_source
    assert "rank_edges" not in report_source

    assert "load_prediction_log" in clv_source
    assert "build_edge_report" in clv_source
    assert "rank_edges" in clv_source
