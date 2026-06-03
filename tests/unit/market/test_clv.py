# tests/unit/market/test_clv.py
"""Unit tests for clv.py — closing line value analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.clv import (
    build_clv_report,
    closing_line_value,
    extract_closing_odds,
    extract_opening_odds,
    spread_clv,
    summarize_clv,
    total_clv,
)
from gridiron_edge.market.recommendations import _REPORT_COLUMNS

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_ledger_row(
    game_id: str,
    market: str,
    side: str,
    odds: float,
    line: float,
    fetched_at: str,
) -> dict:
    """Build a single odds-ledger row."""
    return {
        "fetched_at": pd.Timestamp(fetched_at),
        "sportsbook": "draftkings",
        "season": "2026-2027",
        "week": 1,
        "game_id": game_id,
        "game_date": "2026-09-05",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": market,
        "side": side,
        "odds": odds,
        "line": line,
    }


def _make_multi_pull_ledger(game_id: str = "2026_01_KC_LAC") -> pd.DataFrame:
    """Build a ledger with two pulls (opening and closing) for one game.

    Opening (10:00): ML home -140 / away +120, spread -3 (-110/-110), total 44 (-110/-110)
    Closing (17:00): ML home -160 / away +140, spread -4.5 (-110/-110), total 46 (-110/-110)
    """
    t1 = "2026-09-05 10:00:00"
    t2 = "2026-09-05 17:00:00"
    rows: list[dict] = [
        # Opening pull
        _make_ledger_row(game_id, "moneyline", "home", -140.0, float("nan"), t1),
        _make_ledger_row(game_id, "moneyline", "away", 120.0, float("nan"), t1),
        _make_ledger_row(game_id, "spread", "home", -110.0, -3.0, t1),
        _make_ledger_row(game_id, "spread", "away", -110.0, 3.0, t1),
        _make_ledger_row(game_id, "total", "over", -110.0, 44.0, t1),
        _make_ledger_row(game_id, "total", "under", -110.0, 44.0, t1),
        # Closing pull
        _make_ledger_row(game_id, "moneyline", "home", -160.0, float("nan"), t2),
        _make_ledger_row(game_id, "moneyline", "away", 140.0, float("nan"), t2),
        _make_ledger_row(game_id, "spread", "home", -110.0, -4.5, t2),
        _make_ledger_row(game_id, "spread", "away", -110.0, 4.5, t2),
        _make_ledger_row(game_id, "total", "over", -110.0, 46.0, t2),
        _make_ledger_row(game_id, "total", "under", -110.0, 46.0, t2),
    ]
    return pd.DataFrame(rows)


def _make_edge_report_row(
    game_id: str = "2026_01_KC_LAC",
    market_type: str = "moneyline",
    side: str = "home",
    ev: float = 0.05,
) -> dict:
    """Build one edge-report row with required columns."""
    base: dict[str, str] = dict.fromkeys(_REPORT_COLUMNS, "")
    base.update(
        {
            "game_id": game_id,
            "market_type": market_type,
            "side": side,
            "ev": ev,
            "edge_strength": "moderate",
        }
    )
    return base


# ---------------------------------------------------------------------------
# TestClosingLineValue
# ---------------------------------------------------------------------------


class TestClosingLineValue:
    """Tests for closing_line_value()."""

    def test_positive_clv(self) -> None:
        """Close moved toward bet -> positive CLV."""
        # Bet at 55%, close at 60% -> CLV = (0.60 - 0.55) / 0.55
        result = closing_line_value(0.55, 0.60)
        assert result == pytest.approx(0.05 / 0.55, abs=1e-9)
        assert result > 0

    def test_negative_clv(self) -> None:
        """Close moved away from bet -> negative CLV."""
        result = closing_line_value(0.60, 0.55)
        assert result == pytest.approx(-0.05 / 0.60, abs=1e-9)
        assert result < 0

    def test_zero_clv(self) -> None:
        """No movement -> zero CLV."""
        assert closing_line_value(0.55, 0.55) == pytest.approx(0.0, abs=1e-9)

    def test_large_positive_clv(self) -> None:
        """Big market move toward bet."""
        result = closing_line_value(0.40, 0.60)
        assert result == pytest.approx(0.20 / 0.40, abs=1e-9)

    def test_invalid_bet_prob_zero_raises(self) -> None:
        """bet_fair_prob of 0.0 -> ValueError."""
        with pytest.raises(ValueError, match="bet_fair_prob"):
            closing_line_value(0.0, 0.55)

    def test_invalid_close_prob_one_raises(self) -> None:
        """close_fair_prob of 1.0 -> ValueError."""
        with pytest.raises(ValueError, match="close_fair_prob"):
            closing_line_value(0.55, 1.0)


# ---------------------------------------------------------------------------
# TestSpreadClv
# ---------------------------------------------------------------------------


class TestSpreadClv:
    """Tests for spread_clv()."""

    def test_home_positive(self) -> None:
        """Bet home -3, close -7 -> +4 points of value."""
        # bet=-3, close=-7 -> -3 - (-7) = +4
        assert spread_clv(-3.0, -7.0, "home") == pytest.approx(4.0, abs=1e-9)

    def test_home_negative(self) -> None:
        """Bet home -7, close -3 -> -4 points."""
        assert spread_clv(-7.0, -3.0, "home") == pytest.approx(-4.0, abs=1e-9)

    def test_away_positive(self) -> None:
        """Bet away at home -7, close home -3 -> away gains +4 points."""
        # Home opened -7, closed -3 -> market shifted toward away.
        assert spread_clv(-7.0, -3.0, "away") == pytest.approx(4.0, abs=1e-9)

    def test_away_negative(self) -> None:
        """Bet away at home -3, close home -7 -> away loses 4 points."""
        assert spread_clv(-3.0, -7.0, "away") == pytest.approx(-4.0, abs=1e-9)

    def test_invalid_side_raises(self) -> None:
        """Invalid side -> ValueError."""
        with pytest.raises(ValueError, match="side"):
            spread_clv(-3.0, -7.0, "over")


# ---------------------------------------------------------------------------
# TestTotalClv
# ---------------------------------------------------------------------------


class TestTotalClv:
    """Tests for total_clv()."""

    def test_over_positive(self) -> None:
        """Bet over at 42, close at 45 -> +3 points."""
        assert total_clv(42.0, 45.0, "over") == pytest.approx(3.0, abs=1e-9)

    def test_under_positive(self) -> None:
        """Bet under at 48, close at 45 -> +3 points."""
        assert total_clv(48.0, 45.0, "under") == pytest.approx(3.0, abs=1e-9)

    def test_over_negative(self) -> None:
        """Total dropped -> over bettor loses value."""
        assert total_clv(45.0, 42.0, "over") == pytest.approx(-3.0, abs=1e-9)

    def test_under_negative(self) -> None:
        """Total rose -> under bettor loses value."""
        assert total_clv(45.0, 48.0, "under") == pytest.approx(-3.0, abs=1e-9)

    def test_invalid_side_raises(self) -> None:
        """Invalid side -> ValueError."""
        with pytest.raises(ValueError, match="side"):
            total_clv(45.0, 48.0, "home")


# ---------------------------------------------------------------------------
# TestExtractOpeningOdds
# ---------------------------------------------------------------------------


class TestExtractOpeningOdds:
    """Tests for extract_opening_odds()."""

    def test_takes_first_fetch(self) -> None:
        """Multiple pulls -> earliest wins."""
        ledger: DataFrame = _make_multi_pull_ledger()
        opening = extract_opening_odds(ledger)
        # ML home should be -140 (opening), not -160 (closing)
        ml_home_row = opening[(opening["market"] == "moneyline") & (opening["side"] == "home")]
        assert len(ml_home_row) == 1
        assert ml_home_row.iloc[0]["odds"] == -140.0

    def test_filter_game_ids(self) -> None:
        """Only specified games returned."""
        ledger1: DataFrame = _make_multi_pull_ledger("2026_01_KC_LAC")
        ledger2: DataFrame = _make_multi_pull_ledger("2026_01_BUF_MIA")
        combined: DataFrame = pd.concat([ledger1, ledger2], ignore_index=True)
        opening = extract_opening_odds(combined, game_ids=["2026_01_KC_LAC"])
        assert set(opening["game_id"].unique()) == {"2026_01_KC_LAC"}

    def test_empty_ledger(self) -> None:
        """Empty input -> empty output."""
        empty = pd.DataFrame(
            columns=[
                "fetched_at",
                "sportsbook",
                "season",
                "week",
                "game_id",
                "game_date",
                "away_team",
                "home_team",
                "market",
                "side",
                "odds",
                "line",
            ]
        )
        result = extract_opening_odds(empty)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestExtractClosingOdds
# ---------------------------------------------------------------------------


class TestExtractClosingOdds:
    """Tests for extract_closing_odds()."""

    def test_takes_last_fetch(self) -> None:
        """Multiple pulls -> latest wins."""
        ledger: DataFrame = _make_multi_pull_ledger()
        closing = extract_closing_odds(ledger)
        ml_home_row = closing[(closing["market"] == "moneyline") & (closing["side"] == "home")]
        assert len(ml_home_row) == 1
        assert ml_home_row.iloc[0]["odds"] == -160.0

    def test_filter_game_ids(self) -> None:
        """Only specified games returned."""
        ledger1: DataFrame = _make_multi_pull_ledger("2026_01_KC_LAC")
        ledger2: DataFrame = _make_multi_pull_ledger("2026_01_BUF_MIA")
        combined: DataFrame = pd.concat([ledger1, ledger2], ignore_index=True)
        closing = extract_closing_odds(combined, game_ids=["2026_01_BUF_MIA"])
        assert set(closing["game_id"].unique()) == {"2026_01_BUF_MIA"}

    def test_empty_ledger(self) -> None:
        """Empty input -> empty output."""
        empty = pd.DataFrame(
            columns=[
                "fetched_at",
                "sportsbook",
                "season",
                "week",
                "game_id",
                "game_date",
                "away_team",
                "home_team",
                "market",
                "side",
                "odds",
                "line",
            ]
        )
        result = extract_closing_odds(empty)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestBuildClvReport
# ---------------------------------------------------------------------------


class TestBuildClvReport:
    """Tests for build_clv_report()."""

    def test_augments_edge_report(self) -> None:
        """Adds opening_value, closing_value, clv columns."""
        edge_report = pd.DataFrame([_make_edge_report_row()])
        ledger: DataFrame = _make_multi_pull_ledger()
        result = build_clv_report(edge_report, ledger)
        assert "opening_value" in result.columns
        assert "closing_value" in result.columns
        assert "clv" in result.columns

    def test_ml_clv_computed(self) -> None:
        """Moneyline CLV uses no-vig probability movement."""
        edge_report = pd.DataFrame(
            [
                _make_edge_report_row(market_type="moneyline", side="home"),
            ]
        )
        ledger: DataFrame = _make_multi_pull_ledger()
        result = build_clv_report(edge_report, ledger)
        assert len(result) == 1
        row = result.iloc[0]
        # Opening: home -140/away +120 -> opening no-vig home prob
        # Closing: home -160/away +140 -> closing no-vig home prob
        # Home prob increased (line got stronger) -> positive CLV
        assert not np.isnan(row["clv"])
        assert row["clv"] > 0  # Market moved toward home

    def test_spread_clv_computed(self) -> None:
        """Spread CLV uses point movement."""
        edge_report = pd.DataFrame(
            [
                _make_edge_report_row(market_type="spread", side="home"),
            ]
        )
        ledger: DataFrame = _make_multi_pull_ledger()
        result = build_clv_report(edge_report, ledger)
        assert len(result) == 1
        row = result.iloc[0]
        # Opening spread: -3, Closing spread: -4.5
        # Home CLV = bet_spread - close_spread = -3 - (-4.5) = +1.5
        assert row["clv"] == pytest.approx(1.5, abs=1e-9)

    def test_empty_edges(self) -> None:
        """No edges -> empty report with clv columns."""
        empty = pd.DataFrame(columns=list(_REPORT_COLUMNS))
        ledger: DataFrame = _make_multi_pull_ledger()
        result = build_clv_report(empty, ledger)
        assert len(result) == 0
        assert "clv" in result.columns


# ---------------------------------------------------------------------------
# TestSummarizeClv
# ---------------------------------------------------------------------------


class TestSummarizeClv:
    """Tests for summarize_clv()."""

    def test_mean_clv(self) -> None:
        """Correct mean CLV."""
        df = pd.DataFrame({"clv": [0.10, 0.05, -0.02]})
        stats = summarize_clv(df)
        assert stats["mean_clv"] == pytest.approx((0.10 + 0.05 - 0.02) / 3, abs=1e-9)

    def test_pct_positive(self) -> None:
        """Correct percentage of positive CLV."""
        df = pd.DataFrame({"clv": [0.10, 0.05, -0.02]})
        stats = summarize_clv(df)
        # 2 of 3 are positive
        assert stats["pct_positive_clv"] == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_n_edges(self) -> None:
        """Correct edge count."""
        df = pd.DataFrame({"clv": [0.10, 0.05, -0.02]})
        stats = summarize_clv(df)
        assert stats["n_edges"] == 3.0

    def test_empty_report(self) -> None:
        """Empty report -> NaN stats."""
        df = pd.DataFrame(columns=["clv"])
        stats = summarize_clv(df)
        assert np.isnan(stats["mean_clv"])
        assert stats["n_edges"] == 0.0
