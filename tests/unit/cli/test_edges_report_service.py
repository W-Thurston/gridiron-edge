# tests/unit/cli/test_edges_report_service.py

"""Tests for the standalone edge report service boundary."""

from __future__ import annotations

from unittest.mock import patch

from pandas import DataFrame
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.edges import _format_american_odds, edges_app
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import EdgeResult

runner = CliRunner()
_SERVICE = "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result"


def _diagnostics(
    *,
    state: EdgeResultState,
    blockers: tuple[EdgeDiagnosticBlocker, ...] = (),
    calculated: int = 0,
    positive: int = 0,
    filtered: int = 0,
) -> EdgeDiagnostics:
    return EdgeDiagnostics(
        season="2026-2027",
        week=1,
        prediction_game_count=0,
        market_game_count=0,
        matched_game_count=0,
        complete_moneyline_count=0,
        complete_spread_count=0,
        complete_total_count=0,
        eligible_market_count=0,
        calculated_edge_count=calculated,
        positive_edge_count=positive,
        filtered_edge_count=filtered,
        state=state,
        blockers=blockers,
    )


def _empty_result(
    *,
    state: EdgeResultState = EdgeResultState.BLOCKED,
    blockers: tuple[EdgeDiagnosticBlocker, ...] = (EdgeDiagnosticBlocker.NO_PREDICTIONS,),
    calculated: int = 0,
    positive: int = 0,
) -> EdgeResult:
    return EdgeResult(
        rows=DataFrame(),
        diagnostics=_diagnostics(
            state=state,
            blockers=blockers,
            calculated=calculated,
            positive=positive,
        ),
    )


def _row(*, stake: float | None = None) -> DataFrame:
    return DataFrame(
        [
            {
                "provider": "the_odds_api",
                "provider_event_id": "event-1",
                "sportsbook": "draftkings",
                "market_fetched_at": "2026-09-05T12:00:00+00:00",
                "sportsbook_updated_at": "2026-09-05T11:59:00+00:00",
                "commence_time": "2026-09-06T00:20:00+00:00",
                "game_id": "2026_01_KC_LAC",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market_type": "moneyline",
                "side": "home",
                "american_odds": 125,
                "ev": 0.08,
                "edge_strength": "strong",
                "kelly_stake": stake,
                "confidence_tier": "High",
            }
        ]
    )


def _positive_result(*, stake: float | None = None) -> EdgeResult:
    return EdgeResult(
        rows=_row(stake=stake),
        diagnostics=_diagnostics(
            state=EdgeResultState.POSITIVE_EDGES,
            calculated=1,
            positive=1,
            filtered=1,
        ),
    )


def test_report_calls_service_once_and_omits_bankroll() -> None:
    with patch(_SERVICE, return_value=_positive_result()) as service:
        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )

    assert result.exit_code == 0, result.output
    service.assert_called_once_with(
        season="2026-2027",
        week=1,
        bankroll=None,
        kelly_multiplier=0.25,
        min_ev=0.0,
    )
    assert "1 edge(s) found" in result.output
    assert "+125" in result.output


def test_report_forwards_explicit_bankroll_and_filter() -> None:
    with patch(_SERVICE, return_value=_positive_result(stake=18.0)) as service:
        result = runner.invoke(
            edges_app,
            [
                "report",
                "--week",
                "1",
                "--season",
                "2026-2027",
                "--bankroll",
                "2500",
                "--kelly-multiplier",
                "0.10",
                "--min-ev",
                "0.03",
            ],
        )

    assert result.exit_code == 0, result.output
    assert service.call_args.kwargs["bankroll"] == 2500.0
    assert service.call_args.kwargs["kelly_multiplier"] == 0.10
    assert service.call_args.kwargs["min_ev"] == 0.03


@pytest.mark.parametrize(
    ("blocker", "message"),
    [
        (
            EdgeDiagnosticBlocker.NO_PREDICTIONS,
            "No current weekly product is selected",
        ),
        (
            EdgeDiagnosticBlocker.NO_MARKET_DATA,
            "No current market snapshot is available",
        ),
        (
            EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE,
            "does not contain the requested season and week",
        ),
        (
            EdgeDiagnosticBlocker.MARKET_STALE,
            "market snapshot is stale",
        ),
        (
            EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES,
            "no matching game IDs",
        ),
        (
            EdgeDiagnosticBlocker.INCOMPLETE_MARKETS,
            "market families are incomplete",
        ),
    ],
)
def test_report_renders_blocker_reason(
    blocker: EdgeDiagnosticBlocker,
    message: str,
) -> None:
    with patch(
        _SERVICE,
        return_value=_empty_result(blockers=(blocker,)),
    ):
        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(
    ("state", "calculated", "message"),
    [
        (
            EdgeResultState.NO_CALCULABLE_EDGES,
            0,
            "No calculable edges were produced",
        ),
        (
            EdgeResultState.NO_POSITIVE_EDGES,
            1,
            "no positive expected-value edges",
        ),
    ],
)
def test_report_renders_analytical_empty_state(
    state: EdgeResultState,
    calculated: int,
    message: str,
) -> None:
    with patch(
        _SERVICE,
        return_value=_empty_result(
            state=state,
            blockers=(),
            calculated=calculated,
        ),
    ):
        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )

    assert result.exit_code == 0
    assert message in result.output


def test_report_distinguishes_threshold_filtered_empty_result() -> None:
    with patch(
        _SERVICE,
        return_value=_empty_result(
            state=EdgeResultState.POSITIVE_EDGES,
            blockers=(),
            calculated=2,
            positive=1,
        ),
    ):
        result = runner.invoke(
            edges_app,
            [
                "report",
                "--week",
                "1",
                "--season",
                "2026-2027",
                "--min-ev",
                "0.10",
            ],
        )

    assert result.exit_code == 0
    assert "Positive edges were calculated" in result.output
    assert "min_ev=10.0%" in result.output


def test_report_help_has_no_model_type_and_bankroll_is_optional() -> None:
    result = runner.invoke(edges_app, ["report", "--help"])

    assert result.exit_code == 0
    assert "--model-type" not in result.output
    assert "Optional bankroll" in result.output


def test_american_odds_formatter_preserves_sign() -> None:
    assert _format_american_odds(125) == "+125"
    assert _format_american_odds(-110) == "-110"


def test_table_renders_unavailable_stake() -> None:
    with patch(_SERVICE, return_value=_positive_result(stake=None)):
        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )

    assert result.exit_code == 0, result.output
    assert "—" in result.output
