# tests/unit/cli/test_verify_week.py

"""Tests for read-only weekly readiness assembly."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# pyrefly: ignore [missing-import]
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.verify_week import (
    _render_weekly_readiness,
    _schedule_for_readiness,
    load_weekly_readiness,
    validate_season_label,
    verify_week_cmd,
)
from gridiron_edge.evaluation.weekly_readiness import (
    WeeklyReadiness,
    WeeklyReadinessBlocker,
)
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import EdgeResult

runner = CliRunner()


@pytest.mark.parametrize(
    "season",
    [
        "2026",
        "2026-27",
        "not-a-season",
        "2026-2028",
        "",
    ],
)
def test_rejects_invalid_season_labels(
    season: str,
) -> None:
    with pytest.raises(ValueError):
        validate_season_label(season)


def test_accepts_valid_season_label() -> None:
    assert validate_season_label("2026-2027") == "2026-2027"


@patch("gridiron_edge.market.weekly_edge_service.build_weekly_edge_result")
@patch("gridiron_edge.cli.verify_week.load_current_odds")
@patch("gridiron_edge.cli.verify_week.load_current_weekly_product")
@patch("gridiron_edge.cli.verify_week.load_schedule_upcoming_rich")
def test_selected_product_builds_readiness_without_writes(
    mock_schedule: MagicMock,
    mock_product: MagicMock,
    mock_markets: MagicMock,
    mock_edges: MagicMock,
    tmp_path: Path,
) -> None:
    generated_at = pd.Timestamp("2026-09-01T12:00:00Z")
    fetched_at = pd.Timestamp("2026-09-01T13:00:00Z")
    mock_schedule.return_value = pd.DataFrame(
        {"season": ["2026-2027"], "week": [1], "game_id": ["game-1"]}
    )
    mock_product.return_value = pd.DataFrame(
        {
            "season": ["2026-2027"],
            "week": [1],
            "game_id": ["game-1"],
            "home_win_prob": [0.55],
            "model_spread": [-1.5],
            "model_total": [44.5],
            "projected_home_score": [23.0],
            "projected_away_score": [21.5],
            "win_event_id": ["event-1"],
            "product_run_id": ["run-1"],
            "win_model_name": ["win_prob"],
            "win_model_type": ["elo"],
            "product_generated_at": [generated_at],
        }
    )
    mock_markets.return_value = pd.DataFrame(
        {
            "fetched_at": [fetched_at, fetched_at],
            "sportsbook": ["draftkings", "draftkings"],
            "season": ["2026-2027", "2026-2027"],
            "week": [1, 1],
            "game_id": ["game-1", "game-1"],
            "market": ["moneyline", "moneyline"],
            "side": ["home", "away"],
            "odds": [-110.0, -110.0],
            "line": [None, None],
        }
    )
    mock_edges.return_value = EdgeResult(
        rows=pd.DataFrame({"ev": [0.04]}),
        diagnostics=EdgeDiagnostics(
            season="2026-2027",
            week=1,
            prediction_game_count=1,
            market_game_count=1,
            matched_game_count=1,
            complete_moneyline_count=1,
            complete_spread_count=0,
            complete_total_count=0,
            eligible_market_count=1,
            calculated_edge_count=1,
            positive_edge_count=1,
            filtered_edge_count=1,
            state=EdgeResultState.POSITIVE_EDGES,
        ),
    )

    result = load_weekly_readiness(
        season="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert result.ready
    assert result.prediction_ready
    assert result.market_ready
    assert result.selected_win_prediction_count == 1
    assert result.positive_edge_count == 1


@patch("gridiron_edge.market.weekly_edge_service.build_weekly_edge_result")
@patch("gridiron_edge.cli.verify_week.load_current_odds")
@patch("gridiron_edge.cli.verify_week.load_current_weekly_product")
@patch("gridiron_edge.cli.verify_week.load_schedule_upcoming_rich")
def test_missing_selected_product_is_blocked(
    mock_schedule: MagicMock,
    mock_product: MagicMock,
    mock_markets: MagicMock,
    mock_edges: MagicMock,
    tmp_path: Path,
) -> None:
    mock_schedule.return_value = pd.DataFrame(
        {"season": ["2026-2027"], "week": [1], "game_id": ["game-1"]}
    )
    mock_product.side_effect = FileNotFoundError
    mock_markets.return_value = None
    mock_edges.return_value = EdgeResult(
        rows=pd.DataFrame(),
        diagnostics=EdgeDiagnostics(
            season="2026-2027",
            week=1,
            prediction_game_count=0,
            market_game_count=0,
            matched_game_count=0,
            complete_moneyline_count=0,
            complete_spread_count=0,
            complete_total_count=0,
            eligible_market_count=0,
            calculated_edge_count=0,
            positive_edge_count=0,
            filtered_edge_count=0,
            state=EdgeResultState.BLOCKED,
            blockers=(EdgeDiagnosticBlocker.NO_PREDICTIONS,),
        ),
    )

    result = load_weekly_readiness(
        season="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert not result.prediction_ready
    assert WeeklyReadinessBlocker.MISSING_WEEKLY_PRODUCT in result.blockers


def test_rejects_invalid_week(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="week must be between 1 and 22",
    ):
        load_weekly_readiness(
            season="2026-2027",
            week=0,
            repo=tmp_path,
        )


def _cli_readiness(
    *,
    blockers: tuple[
        WeeklyReadinessBlocker,
        ...,
    ] = (),
    positive_edge_count: int = 3,
) -> WeeklyReadiness:
    """Create a readiness result for CLI rendering tests."""
    return WeeklyReadiness(
        season="2026-2027",
        week=1,
        scheduled_game_count=2,
        selected_win_prediction_count=2,
        spread_value_count=2,
        total_prediction_count=2,
        projected_score_count=2,
        complete_provenance_count=2,
        market_game_count=2,
        prediction_market_match_count=2,
        eligible_market_count=6,
        positive_edge_count=positive_edge_count,
        prediction_generated_at=datetime(
            2026,
            9,
            1,
            12,
            tzinfo=UTC,
        ),
        market_fetched_at=datetime(
            2026,
            9,
            1,
            13,
            tzinfo=UTC,
        ),
        market_source="draftkings",
        blockers=blockers,
    )


def _command_app() -> typer.Typer:
    """Create an isolated Typer app for command tests."""
    app = typer.Typer()
    app.command()(verify_week_cmd)
    return app


class TestVerifyWeekCommand:
    """Tests for weekly readiness CLI rendering and exit behavior."""

    @patch("gridiron_edge.cli.verify_week.load_weekly_readiness")
    def test_complete_readiness_exits_successfully(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _cli_readiness()

        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "verify-week  2026-2027 week 1" in result.output
        assert "Ready" in result.output
        assert "No blockers" in result.output

        mock_load.assert_called_once_with(
            season="2026-2027",
            week=1,
        )

    def test_run_id_option_is_not_supported(self) -> None:
        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                "1",
                "--run-id",
                "run-1",
            ],
        )

        assert result.exit_code != 0

    @patch("gridiron_edge.cli.verify_week.load_weekly_readiness")
    def test_blocked_readiness_exits_nonzero(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _cli_readiness(
            blockers=(
                WeeklyReadinessBlocker.MISSING_FORECAST_SELECTION,
                WeeklyReadinessBlocker.MISSING_WIN_PREDICTIONS,
            ),
            positive_edge_count=0,
        )

        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 1
        assert "Blocked" in result.output
        assert "missing_forecast_selection" in result.output
        assert "missing_win_predictions" in result.output

    @patch("gridiron_edge.cli.verify_week.load_weekly_readiness")
    def test_zero_positive_edges_exits_successfully(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _cli_readiness(
            positive_edge_count=0,
        )

        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Positive edges                  0" in result.output
        assert "Ready" in result.output

    @patch("gridiron_edge.cli.verify_week.load_weekly_readiness")
    def test_output_contains_every_diagnostic_count(
        self,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _cli_readiness()

        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 0

        expected_labels = [
            "Scheduled games",
            "Selected win predictions",
            "Spread values",
            "Total predictions",
            "Projected scores",
            "Complete provenance",
            "Games with market data",
            "Prediction-market matches",
            "Eligible markets",
            "Positive edges",
            "Prediction generated at",
            "Market fetched at",
            "Market source",
        ]

        for label in expected_labels:
            assert label in result.output


class TestVerifyWeekValidation:
    """Tests for required CLI scope and validation."""

    def test_season_is_required(self) -> None:
        result = runner.invoke(
            _command_app(),
            [
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 2

    def test_week_is_required(self) -> None:
        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
            ],
        )

        assert result.exit_code == 2

    @pytest.mark.parametrize(
        "week",
        [
            "0",
            "23",
        ],
    )
    def test_week_range_is_validated(
        self,
        week: str,
    ) -> None:
        result = runner.invoke(
            _command_app(),
            [
                "--season",
                "2026-2027",
                "--week",
                week,
            ],
        )

        assert result.exit_code == 2

    @pytest.mark.parametrize(
        "season",
        [
            "not-a-season",
            "2026",
            "2026-2028",
        ],
    )
    def test_season_format_is_validated(
        self,
        season: str,
    ) -> None:
        result = runner.invoke(
            _command_app(),
            [
                "--season",
                season,
                "--week",
                "1",
            ],
        )

        assert result.exit_code == 2
        assert "Expected format" in result.output or ("ending year" in result.output)


def test_rendering_marks_unavailable_provenance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    readiness = WeeklyReadiness(
        season="2026-2027",
        week=1,
        scheduled_game_count=0,
        selected_win_prediction_count=0,
        spread_value_count=0,
        total_prediction_count=0,
        projected_score_count=0,
        complete_provenance_count=0,
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        prediction_generated_at=None,
        market_fetched_at=None,
        market_source=None,
        blockers=(WeeklyReadinessBlocker.MISSING_SCHEDULE,),
    )

    _render_weekly_readiness(readiness)

    output = capsys.readouterr().out

    assert output.count("unavailable") == 3
    assert "missing_schedule" in output


def test_verify_week_is_registered_on_main_app() -> None:
    from gridiron_edge.cli.main import app

    result = runner.invoke(
        app,
        [
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "verify-week" in result.output


def test_verify_week_module_contains_no_writer_imports() -> None:
    from gridiron_edge.cli import verify_week

    source = Path(verify_week.__file__).read_text()

    forbidden = [
        "write_forecast_events",
        "write_current_odds_snapshot",
        "append_to_odds_ledger",
        "append_to_prediction_log",
        "build_predictions_df",
        "enrich_predictions",
    ]

    for name in forbidden:
        assert name not in source


def test_rich_schedule_projects_to_readiness_identity() -> None:
    rich = pd.DataFrame(
        {
            "season": [
                "2026-2027",
                "2026-2027",
            ],
            "week": [1, 2],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_02_BAL_BUF",
            ],
            "stadium": [
                "SoFi Stadium",
                "Highmark Stadium",
            ],
            "spread_line": [
                pd.NA,
                -2.5,
            ],
        }
    )

    projected = _schedule_for_readiness(rich)

    assert projected.to_dict(orient="list") == {
        "YEAR": [
            "2026-2027",
            "2026-2027",
        ],
        "WEEK_NUM": [1, 2],
        "GAME_ID": [
            "2026_01_KC_LAC",
            "2026_02_BAL_BUF",
        ],
    }


@patch("gridiron_edge.cli.verify_week.load_current_odds")
@patch("gridiron_edge.cli.verify_week.load_current_weekly_product")
@patch("gridiron_edge.cli.verify_week.load_schedule_upcoming_rich")
def test_missing_rich_schedule_remains_visible(
    mock_schedule: MagicMock,
    mock_product: MagicMock,
    mock_markets: MagicMock,
    tmp_path: Path,
) -> None:
    mock_schedule.side_effect = FileNotFoundError
    mock_product.side_effect = FileNotFoundError
    mock_markets.return_value = None

    result = load_weekly_readiness(
        season="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert result.scheduled_game_count == 0
    assert WeeklyReadinessBlocker.MISSING_SCHEDULE in result.blockers


def test_verify_week_uses_rich_schedule_without_legacy_fallback() -> None:
    from gridiron_edge.cli import verify_week

    source = Path(verify_week.__file__).read_text()

    assert "load_schedule_upcoming_rich" in source
    assert "load_schedule_upcoming(" not in source


def test_verify_week_has_no_direct_edge_calculation_dependencies() -> None:
    from gridiron_edge.cli import verify_week

    source = Path(verify_week.__file__).read_text()
    retired = (
        "build_edge_report",
        "get_margin_std",
        "get_total_std",
    )
    found = [name for name in retired if name in source]

    assert found == []
    assert "build_weekly_edge_result" in source
