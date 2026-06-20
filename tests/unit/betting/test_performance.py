# tests/unit/betting/test_performance.py
"""Unit tests for betting performance analytics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.betting.ledger import _BET_COLUMNS
from gridiron_edge.betting.performance import (
    clv_summary,
    ev_analysis,
    record,
    roi,
    streak_analysis,
    summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


def _make_bet(
    status: str = "won",
    pnl: float = 100.0,
    stake: float = 100.0,
    odds: int = -110,
    market_type: str = "moneyline",
    book: str = "draftkings",
    model_ev: float | None = None,
    model_prob: float | None = None,
    clv: float | None = None,
    confidence_tier: str | None = None,
    edge_strength: str | None = None,
    placed_at: datetime | None = None,
    game_id: str = "2026_01_KC_LAC",
    model_name: str | None = None,
    model_type: str | None = None,
) -> dict:
    """Build a single bet row as a dict."""
    if placed_at is None:
        placed_at = _BASE_TIME
    return {
        "bet_id": f"test-{id(status)}-{pnl}",
        "game_id": game_id,
        "placed_at": placed_at,
        "market_type": market_type,
        "side": "home",
        "line": None,
        "odds": odds,
        "stake": stake,
        "book": book,
        "model_name": model_name,
        "model_type": model_type,
        "model_prob": model_prob,
        "model_ev": model_ev,
        "edge_strength": edge_strength,
        "confidence_tier": confidence_tier,
        "status": status,
        "settled_at": placed_at + timedelta(hours=3) if status != "open" else None,
        "pnl": pnl if status != "open" else None,
        "closing_line": None,
        "closing_odds": None,
        "clv": clv,
    }


def _make_bets(*bet_dicts: dict) -> pd.DataFrame:
    """Build a DataFrame from bet dicts."""
    return pd.DataFrame(list(bet_dicts), columns=_BET_COLUMNS)


def _empty_bets() -> pd.DataFrame:
    return pd.DataFrame(columns=_BET_COLUMNS)


# ---------------------------------------------------------------------------
# TestRecord
# ---------------------------------------------------------------------------


class TestRecord:
    """Tests for W-L-P record calculation."""

    def test_overall_record(self) -> None:
        """3W 1L 1P -> correct counts and win_pct."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100),
            _make_bet("won", 80),
            _make_bet("won", 120),
            _make_bet("lost", -100),
            _make_bet("push", 0),
        )
        df = record(bets)
        assert df.iloc[0]["wins"] == 3
        assert df.iloc[0]["losses"] == 1
        assert df.iloc[0]["pushes"] == 1
        assert df.iloc[0]["total"] == 5
        assert df.iloc[0]["win_pct"] == pytest.approx(0.75)

    def test_split_by_market(self) -> None:
        """Split by market_type produces separate rows."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, market_type="moneyline"),
            _make_bet("lost", -100, market_type="moneyline"),
            _make_bet("won", 50, market_type="spread"),
        )
        df = record(bets, split_by="market_type")
        assert len(df) == 2
        ml_row = df[df["market_type"] == "moneyline"].iloc[0]
        assert ml_row["wins"] == 1
        assert ml_row["losses"] == 1

    def test_ignores_open_bets(self) -> None:
        """Open bets are not counted."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100),
            _make_bet("open", 0),
        )
        df = record(bets)
        assert df.iloc[0]["total"] == 1

    def test_empty_bets(self) -> None:
        """Empty DataFrame -> empty result."""
        df = record(_empty_bets())
        assert df.empty

    def test_win_pct_excludes_pushes(self) -> None:
        """win_pct denominator is wins + losses only."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100),
            _make_bet("push", 0),
            _make_bet("push", 0),
            _make_bet("push", 0),
        )
        df = record(bets)
        # 1 win, 0 losses -> win_pct = 1.0
        assert df.iloc[0]["win_pct"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestRoi
# ---------------------------------------------------------------------------


class TestRoi:
    """Tests for ROI calculation."""

    def test_overall_roi(self) -> None:
        """Known stakes and PnL -> correct ROI."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 150, stake=100),
            _make_bet("lost", -100, stake=100),
        )
        df = roi(bets)
        # staked=200, pnl=50, roi=25%
        assert df.iloc[0]["total_staked"] == pytest.approx(200.0)
        assert df.iloc[0]["total_pnl"] == pytest.approx(50.0)
        assert df.iloc[0]["roi_pct"] == pytest.approx(25.0)

    def test_split_by_book(self) -> None:
        """Split by book produces separate rows."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, book="draftkings"),
            _make_bet("lost", -100, book="fanduel"),
        )
        df = roi(bets, split_by="book")
        assert len(df) == 2

    def test_empty_bets(self) -> None:
        """Empty DataFrame -> empty result."""
        df = roi(_empty_bets())
        assert df.empty

    def test_all_losses(self) -> None:
        """All losses -> negative ROI."""
        bets: DataFrame = _make_bets(
            _make_bet("lost", -100, stake=100),
            _make_bet("lost", -100, stake=100),
        )
        df = roi(bets)
        assert df.iloc[0]["roi_pct"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# TestClvSummary
# ---------------------------------------------------------------------------


class TestClvSummary:
    """Tests for CLV summary."""

    def test_with_clv_data(self) -> None:
        """Known CLV values produce correct summary."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, clv=0.05),
            _make_bet("won", 80, clv=0.03),
            _make_bet("lost", -100, clv=-0.02),
        )
        result = clv_summary(bets)
        assert result["mean_clv"] == pytest.approx(0.02, abs=1e-4)
        assert result["median_clv"] == pytest.approx(0.03, abs=1e-4)
        assert result["pct_positive_clv"] == pytest.approx(66.6667, rel=1e-2)
        assert result["n_bets"] == 3

    def test_no_clv_data(self) -> None:
        """All clv=NaN -> NaN results."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, clv=None),
            _make_bet("lost", -100, clv=None),
        )
        result = clv_summary(bets)
        assert math.isnan(result["mean_clv"])
        assert result["n_bets"] == 0

    def test_empty(self) -> None:
        """Empty DataFrame -> NaN results."""
        result = clv_summary(_empty_bets())
        assert math.isnan(result["mean_clv"])
        assert result["n_bets"] == 0


# ---------------------------------------------------------------------------
# TestEvAnalysis
# ---------------------------------------------------------------------------


class TestEvAnalysis:
    """Tests for EV analysis."""

    def test_with_ev_data(self) -> None:
        """Known EV and actual results."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 150, stake=100, model_ev=0.10),
            _make_bet("lost", -100, stake=100, model_ev=0.05),
        )
        result = ev_analysis(bets)
        assert result["mean_ev_at_bet"] == pytest.approx(0.075)
        # actual roi per bet: 1.5, -1.0 -> mean = 0.25
        assert result["mean_actual_roi"] == pytest.approx(0.25)
        assert result["n_model_bets"] == 2

    def test_no_ev_data(self) -> None:
        """No model_ev populated -> NaN results."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, model_ev=None),
        )
        result = ev_analysis(bets)
        assert math.isnan(result["mean_ev_at_bet"])
        assert result["n_model_bets"] == 0

    def test_gap_calculation(self) -> None:
        """ev_vs_actual_gap is actual - expected."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, stake=100, model_ev=0.05),
        )
        result = ev_analysis(bets)
        # actual roi = 100/100 = 1.0, ev = 0.05, gap = 0.95
        assert result["ev_vs_actual_gap"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# TestStreakAnalysis
# ---------------------------------------------------------------------------


class TestStreakAnalysis:
    """Tests for streak analysis."""

    def test_current_win_streak(self) -> None:
        """W W W -> current streak = 3."""
        t = _BASE_TIME
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, placed_at=t),
            _make_bet("won", 100, placed_at=t + timedelta(hours=1)),
            _make_bet("won", 100, placed_at=t + timedelta(hours=2)),
        )
        result = streak_analysis(bets)
        assert result["current_streak"] == 3
        assert result["current_streak_type"] == "W"

    def test_current_loss_streak(self) -> None:
        """W W L L -> current streak = -2."""
        t = _BASE_TIME
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, placed_at=t),
            _make_bet("won", 100, placed_at=t + timedelta(hours=1)),
            _make_bet("lost", -100, placed_at=t + timedelta(hours=2)),
            _make_bet("lost", -100, placed_at=t + timedelta(hours=3)),
        )
        result = streak_analysis(bets)
        assert result["current_streak"] == -2
        assert result["current_streak_type"] == "L"

    def test_push_breaks_streak(self) -> None:
        """W W P W -> current streak = 1."""
        t = _BASE_TIME
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, placed_at=t),
            _make_bet("won", 100, placed_at=t + timedelta(hours=1)),
            _make_bet("push", 0, placed_at=t + timedelta(hours=2)),
            _make_bet("won", 100, placed_at=t + timedelta(hours=3)),
        )
        result = streak_analysis(bets)
        assert result["current_streak"] == 1

    def test_longest_streaks(self) -> None:
        """Mixed results -> correct longest streaks."""
        t = _BASE_TIME
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, placed_at=t),
            _make_bet("won", 100, placed_at=t + timedelta(hours=1)),
            _make_bet("won", 100, placed_at=t + timedelta(hours=2)),
            _make_bet("lost", -100, placed_at=t + timedelta(hours=3)),
            _make_bet("lost", -100, placed_at=t + timedelta(hours=4)),
            _make_bet("won", 100, placed_at=t + timedelta(hours=5)),
        )
        result = streak_analysis(bets)
        assert result["longest_win_streak"] == 3
        assert result["longest_loss_streak"] == 2

    def test_empty(self) -> None:
        """Empty DataFrame -> all zeros."""
        result = streak_analysis(_empty_bets())
        assert result["current_streak"] == 0
        assert result["longest_win_streak"] == 0


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for the combined summary."""

    def test_returns_all_keys(self) -> None:
        """Summary has all expected keys."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 100, stake=100, model_ev=0.05, clv=0.03),
        )
        result = summary(bets)
        expected_keys: set[str] = {
            "wins",
            "losses",
            "pushes",
            "total",
            "win_pct",
            "total_staked",
            "total_pnl",
            "roi_pct",
            "mean_clv",
            "pct_positive_clv",
            "n_clv_bets",
            "mean_ev_at_bet",
            "n_model_bets",
            "current_streak",
            "current_streak_type",
            "longest_win_streak",
            "longest_loss_streak",
        }
        assert set(result.keys()) == expected_keys

    def test_values_consistent(self) -> None:
        """Summary values match individual function calls."""
        bets: DataFrame = _make_bets(
            _make_bet("won", 150, stake=100),
            _make_bet("lost", -100, stake=100),
        )
        s = summary(bets)
        rec = record(bets)
        r = roi(bets)
        assert s["wins"] == int(rec.iloc[0]["wins"])
        assert s["total_pnl"] == pytest.approx(float(r.iloc[0]["total_pnl"]))
