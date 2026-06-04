# tests/integration/test_edges_cli.py
"""Integration tests for the edges CLI sub-commands."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from pandas import Timestamp
from typer.testing import CliRunner

from gridiron_edge.cli.edges import edges_app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_predictions(
    game_id: str = "2026_01_KC_LAC",
    home_win_prob: float = 0.70,
    model_spread: float = -7.0,
    model_total: float = 50.0,
) -> pd.DataFrame:
    """Build a single-row predictions DataFrame matching archive schema."""
    return pd.DataFrame(
        [
            {
                "predicted_at": pd.Timestamp("2026-09-04 12:00:00"),
                "is_backfilled": False,
                "model_version": "random_forest",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "away_elo": 1550.0,
                "home_elo": 1520.0,
                "away_win_prob": 1.0 - home_win_prob,
                "home_win_prob": home_win_prob,
                "model_spread": model_spread,
                "model_total": model_total,
                "projected_home_score": 28.0,
                "projected_away_score": 22.0,
                "margin_std": 13.54,
                "win_prob_lo": 0.50,
                "win_prob_hi": 0.85,
                "confidence_tier": "High",
            }
        ]
    )


def _make_long_odds(
    game_id: str = "2026_01_KC_LAC",
) -> pd.DataFrame:
    """Build long-format odds for one game, all three markets."""
    ts = pd.Timestamp("2026-09-05 12:00:00")
    rows: list[dict[str, Timestamp | float | int | str]] = [
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "moneyline",
            "side": "home",
            "odds": -200.0,
            "line": float("nan"),
        },
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "moneyline",
            "side": "away",
            "odds": 170.0,
            "line": float("nan"),
        },
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "spread",
            "side": "home",
            "odds": -110.0,
            "line": -3.5,
        },
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "spread",
            "side": "away",
            "odds": -110.0,
            "line": 3.5,
        },
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "total",
            "side": "over",
            "odds": -110.0,
            "line": 44.0,
        },
        {
            "fetched_at": ts,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "market": "total",
            "side": "under",
            "odds": -110.0,
            "line": 44.0,
        },
    ]
    return pd.DataFrame(rows)


_PREDICTIONS_PATH = "gridiron_edge.evaluation.archive.load_prediction_log"
_CURRENT_ODDS_PATH = "gridiron_edge.ingest.odds.store.load_current_odds"
_ODDS_LEDGER_PATH = "gridiron_edge.ingest.odds.store.load_odds_ledger"
_MARGIN_STD_PATH = "gridiron_edge.models.game_prediction.post_process.get_margin_std"


# ---------------------------------------------------------------------------
# TestReportCommand
# ---------------------------------------------------------------------------


class TestReportCommand:
    """Tests for 'gridiron edges report'."""

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_CURRENT_ODDS_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_report_runs(self, mock_preds, mock_odds, mock_std) -> None:
        """Command completes successfully with valid data."""
        mock_preds.return_value = _make_predictions()
        mock_odds.return_value = _make_long_odds()

        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )
        assert result.exit_code == 0

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_CURRENT_ODDS_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_report_no_predictions(self, mock_preds, mock_odds, mock_std) -> None:
        """Empty predictions -> graceful exit with message."""
        mock_preds.return_value = pd.DataFrame()
        mock_odds.return_value = _make_long_odds()

        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )
        assert result.exit_code == 0
        assert "No predictions found" in result.output

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_CURRENT_ODDS_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_report_no_odds(self, mock_preds, mock_odds, mock_std) -> None:
        """Empty odds -> graceful exit with message."""
        mock_preds.return_value = _make_predictions()
        mock_odds.return_value = pd.DataFrame()

        result = runner.invoke(
            edges_app,
            ["report", "--week", "1", "--season", "2026-2027"],
        )
        assert result.exit_code == 0
        assert "No current odds" in result.output


# ---------------------------------------------------------------------------
# TestClvCommand
# ---------------------------------------------------------------------------


class TestClvCommand:
    """Tests for 'gridiron edges clv'."""

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_ODDS_LEDGER_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_clv_runs(self, mock_preds, mock_ledger, mock_std) -> None:
        """Command completes successfully with valid data."""
        mock_preds.return_value = _make_predictions()
        mock_ledger.return_value = _make_long_odds()

        result = runner.invoke(
            edges_app,
            ["clv", "--season", "2026-2027"],
        )
        assert result.exit_code == 0

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_ODDS_LEDGER_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_clv_no_predictions(self, mock_preds, mock_ledger, mock_std) -> None:
        """Empty predictions -> graceful exit."""
        mock_preds.return_value = pd.DataFrame()
        mock_ledger.return_value = _make_long_odds()

        result = runner.invoke(
            edges_app,
            ["clv", "--season", "2026-2027"],
        )
        assert result.exit_code == 0
        assert "No predictions found" in result.output

    @patch(_MARGIN_STD_PATH, return_value=13.54)
    @patch(_ODDS_LEDGER_PATH)
    @patch(_PREDICTIONS_PATH)
    def test_clv_no_odds(self, mock_preds, mock_ledger, mock_std) -> None:
        """Empty odds ledger -> graceful exit."""
        mock_preds.return_value = _make_predictions()
        mock_ledger.return_value = pd.DataFrame()

        result = runner.invoke(
            edges_app,
            ["clv", "--season", "2026-2027"],
        )
        assert result.exit_code == 0
        assert "No odds ledger" in result.output
