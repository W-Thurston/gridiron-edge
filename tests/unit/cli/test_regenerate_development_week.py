# tests/unit/cli/test_regenerate_development_week.py

"""Tests for the regenerate-development-week command.

Generation and composition are each already covered directly (generation by
``test_development_forecast.py``, composition by
``test_weekly_product_composition.py``); these tests cover this command's own
job: wiring the two together atomically in one process — sharing the exact
in-memory policy and events rather than reloading or reconstructing them —
and then reporting readiness.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

# pyrefly: ignore [missing-import]
import typer
from typer.testing import CliRunner

from gridiron_edge.cli._weekly_product_composition import ComposedWeeklyProduct
from gridiron_edge.cli.development_forecast import (
    DevelopmentForecastResult,
    regenerate_development_week_cmd,
)
from gridiron_edge.evaluation.weekly_readiness import WeeklyReadiness, WeeklyReadinessBlocker

SEASON = "2026-2027"
WEEK = 2
RUN_ID = "run-1"
GENERATED_AT = datetime(2026, 9, 25, 12, tzinfo=UTC)

runner = CliRunner()


def _command_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(regenerate_development_week_cmd)
    return app


def _generation_result() -> DevelopmentForecastResult:
    return DevelopmentForecastResult(
        run_id=RUN_ID,
        generated_at=GENERATED_AT,
        schedule=pd.DataFrame({"season": [SEASON], "week": [WEEK], "game_id": ["game-1"]}),
        execution=SimpleNamespace(
            policy=MagicMock(),
            events=pd.DataFrame({"event_id": ["event-1"]}),
            input_evidence=(MagicMock(),),
        ),
        artifacts=(Path("/repo/events.parquet"),),
    )


def _readiness(*, ready: bool) -> WeeklyReadiness:
    return WeeklyReadiness(
        season=SEASON,
        week=WEEK,
        scheduled_game_count=1,
        selected_win_prediction_count=1,
        spread_value_count=1,
        total_prediction_count=1,
        projected_score_count=1,
        complete_provenance_count=1,
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        blockers=() if ready else (WeeklyReadinessBlocker.MISSING_MARKET_DATA,),
    )


@patch("gridiron_edge.cli.verify_week.load_weekly_readiness")
@patch("gridiron_edge.cli.development_forecast._generate_development_forecast")
def test_success_composes_selects_and_reports_readiness(
    mock_generate: MagicMock,
    mock_readiness: MagicMock,
) -> None:
    generation = _generation_result()
    mock_generate.return_value = generation
    mock_readiness.return_value = _readiness(ready=True)

    composed = ComposedWeeklyProduct(
        product_id=f"weekly_{SEASON.replace('-', '_')}_wk{WEEK:02d}_{RUN_ID}",
        artifact=Path("/repo/weekly.parquet"),
        row_count=1,
    )
    with patch(
        "gridiron_edge.cli._weekly_product_composition.compose_and_select_weekly_product",
        return_value=composed,
    ) as mock_compose:
        result = runner.invoke(
            _command_app(),
            ["--season", SEASON, "--week", str(WEEK)],
        )

    assert result.exit_code == 0
    assert f"Run ID: {RUN_ID}" in result.output
    assert composed.product_id in result.output
    assert "prediction_ready: True" in result.output

    mock_compose.assert_called_once()
    _, kwargs = mock_compose.call_args
    assert kwargs["schedule"] is generation.schedule
    assert kwargs["events"] is generation.execution.events
    assert kwargs["policy"] is generation.execution.policy
    assert kwargs["run_id"] == RUN_ID
    assert kwargs["generated_at"] == GENERATED_AT
    assert kwargs["season"] == SEASON
    assert kwargs["week"] == WEEK

    mock_readiness.assert_called_once()
    _, readiness_kwargs = mock_readiness.call_args
    assert readiness_kwargs["season"] == SEASON
    assert readiness_kwargs["week"] == WEEK


@patch("gridiron_edge.cli.development_forecast._generate_development_forecast")
def test_generation_failure_exits_nonzero_without_composing(
    mock_generate: MagicMock,
) -> None:
    mock_generate.side_effect = ValueError("no retained games for that week")

    with patch(
        "gridiron_edge.cli._weekly_product_composition.compose_and_select_weekly_product"
    ) as mock_compose:
        result = runner.invoke(
            _command_app(),
            ["--season", SEASON, "--week", str(WEEK)],
        )

    assert result.exit_code != 0
    mock_compose.assert_not_called()


@patch("gridiron_edge.cli.development_forecast._generate_development_forecast")
def test_composition_failure_exits_nonzero(
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = _generation_result()

    with patch(
        "gridiron_edge.cli._weekly_product_composition.compose_and_select_weekly_product",
        side_effect=ValueError("Rich schedule has no rows for the requested season and week."),
    ):
        result = runner.invoke(
            _command_app(),
            ["--season", SEASON, "--week", str(WEEK)],
        )

    assert result.exit_code != 0
