# tests/unit/market/test_recommendations.py
"""Unit tests for recommendations.py - odds pivot, join, edge report, ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp
import pytest

from gridiron_edge.market.edge import MoneylineEdge, SpreadEdge, TotalEdge
from gridiron_edge.market.recommendations import (
    _REPORT_COLUMNS,
    build_edge_report,
    compute_game_edges,
    join_predictions_to_odds,
    pivot_odds_to_wide,
    rank_edges,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_long_odds(
    *,
    game_id: str = "2026_01_KC_LAC",
    ml_home: float = -150.0,
    ml_away: float = 130.0,
    spread_home: float = -3.0,
    spread_odds_home: float = -110.0,
    spread_odds_away: float = -110.0,
    total_line: float = 45.0,
    over_odds: float = -110.0,
    under_odds: float = -110.0,
    fetched_at: str = "2026-09-05 12:00:00",
) -> pd.DataFrame:
    """Build a long-format odds DataFrame for a single game."""
    ts = pd.Timestamp(fetched_at)
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
            "odds": ml_home,
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
            "odds": ml_away,
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
            "odds": spread_odds_home,
            "line": spread_home,
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
            "odds": spread_odds_away,
            "line": -spread_home,
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
            "odds": over_odds,
            "line": total_line,
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
            "odds": under_odds,
            "line": total_line,
        },
    ]
    return pd.DataFrame(rows)


def _make_predictions(
    *,
    game_id: str = "2026_01_KC_LAC",
    home_win_prob: float = 0.65,
    model_spread: float = -4.5,
    model_total: float = 48.0,
    margin_std: float = 13.54,
    confidence_tier: str = "High",
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
                "projected_home_score": 23.0,
                "projected_away_score": 25.0,
                "margin_std": margin_std,
                "win_prob_lo": 0.45,
                "win_prob_hi": 0.80,
                "confidence_tier": confidence_tier,
            }
        ]
    )


# ---------------------------------------------------------------------------
# TestPivotOddsToWide
# ---------------------------------------------------------------------------


class TestPivotOddsToWide:
    """Tests for pivot_odds_to_wide()."""

    def test_standard_pivot(self) -> None:
        """All 6 rows for one game -> correct wide row."""
        odds: DataFrame = _make_long_odds()
        wide: DataFrame = pivot_odds_to_wide(odds)
        assert len(wide) == 1
        row: Series = wide.iloc[0]
        assert row["game_id"] == "2026_01_KC_LAC"
        assert row["ml_home"] == -150.0
        assert row["ml_away"] == 130.0
        assert row["spread_line_home"] == -3.0
        assert row["spread_odds_home"] == -110.0
        assert row["spread_odds_away"] == -110.0
        assert row["total_line"] == 45.0
        assert row["over_odds"] == -110.0
        assert row["under_odds"] == -110.0

    def test_multiple_games(self) -> None:
        """Two games -> two wide rows."""
        odds1: DataFrame = _make_long_odds(game_id="2026_01_KC_LAC")
        odds2: DataFrame = _make_long_odds(game_id="2026_01_BUF_MIA")
        combined: DataFrame = pd.concat([odds1, odds2], ignore_index=True)
        wide: DataFrame = pivot_odds_to_wide(combined)
        assert len(wide) == 2
        assert set(wide["game_id"]) == {"2026_01_KC_LAC", "2026_01_BUF_MIA"}

    def test_missing_spread_market(self) -> None:
        """Game with only ML and total -> spread columns are NaN."""
        odds: DataFrame = _make_long_odds()
        odds = odds[odds["market"] != "spread"].copy()
        wide: DataFrame = pivot_odds_to_wide(odds)
        assert len(wide) == 1
        assert np.isnan(wide.iloc[0]["spread_line_home"])
        assert np.isnan(wide.iloc[0]["spread_odds_home"])

    def test_missing_total_market(self) -> None:
        """Game with only ML and spread -> total columns are NaN."""
        odds: DataFrame = _make_long_odds()
        odds = odds[odds["market"] != "total"].copy()
        wide: DataFrame = pivot_odds_to_wide(odds)
        assert len(wide) == 1
        assert np.isnan(wide.iloc[0]["total_line"])
        assert np.isnan(wide.iloc[0]["over_odds"])

    def test_duplicate_fetches_last_wins(self) -> None:
        """Duplicate rows for the same market -> most recent fetch wins."""
        odds1: DataFrame = _make_long_odds(ml_home=-140.0, fetched_at="2026-09-05 10:00:00")
        odds2: DataFrame = _make_long_odds(ml_home=-155.0, fetched_at="2026-09-05 12:00:00")
        combined: DataFrame = pd.concat([odds1, odds2], ignore_index=True)
        wide: DataFrame = pivot_odds_to_wide(combined)
        assert len(wide) == 1
        assert wide.iloc[0]["ml_home"] == -155.0

    def test_empty_input(self) -> None:
        """Empty odds DataFrame -> empty wide DataFrame."""
        odds = pd.DataFrame(
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
        wide: DataFrame = pivot_odds_to_wide(odds)
        assert len(wide) == 0
        assert "ml_home" in wide.columns


# ---------------------------------------------------------------------------
# TestJoinPredictionsToOdds
# ---------------------------------------------------------------------------


class TestJoinPredictionsToOdds:
    """Tests for join_predictions_to_odds()."""

    def test_inner_join(self) -> None:
        """3 predictions, 2 with matching odds -> 2 joined rows."""
        preds: DataFrame = pd.concat(
            [
                _make_predictions(game_id="2026_01_KC_LAC"),
                _make_predictions(game_id="2026_01_BUF_MIA"),
                _make_predictions(game_id="2026_01_SF_SEA"),
            ],
            ignore_index=True,
        )
        odds: DataFrame = pd.concat(
            [
                _make_long_odds(game_id="2026_01_KC_LAC"),
                _make_long_odds(game_id="2026_01_BUF_MIA"),
            ],
            ignore_index=True,
        )
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        assert len(joined) == 2

    def test_all_match(self) -> None:
        """All predictions have odds -> all rows present."""
        preds: DataFrame = _make_predictions()
        odds: DataFrame = _make_long_odds()
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        assert len(joined) == 1
        assert "ml_home" in joined.columns
        assert "home_win_prob" in joined.columns

    def test_no_match(self) -> None:
        """Disjoint game_ids -> empty DataFrame."""
        preds: DataFrame = _make_predictions(game_id="2026_01_KC_LAC")
        odds: DataFrame = _make_long_odds(game_id="2026_01_BUF_MIA")
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        assert len(joined) == 0

    def test_accepts_wide_odds(self) -> None:
        """Already-pivoted odds work directly (no double pivot)."""
        preds: DataFrame = _make_predictions()
        wide_odds: DataFrame = pivot_odds_to_wide(_make_long_odds())
        joined: DataFrame = join_predictions_to_odds(preds, wide_odds)
        assert len(joined) == 1
        assert "ml_home" in joined.columns


# ---------------------------------------------------------------------------
# TestComputeGameEdges
# ---------------------------------------------------------------------------


class TestComputeGameEdges:
    """Tests for compute_game_edges()."""

    def test_all_three_markets(self) -> None:
        """Row with all data -> at least one edge per market if +EV."""
        preds: DataFrame = _make_predictions(
            home_win_prob=0.70,
            model_spread=-7.0,
            model_total=52.0,
        )
        odds: DataFrame = _make_long_odds(
            ml_home=-150,
            ml_away=130,
            spread_home=-3.0,
            total_line=45.0,
        )
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        row: Series = joined.iloc[0]
        edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = compute_game_edges(
            row, margin_std=13.0, total_std=13.0
        )
        # With model at 70% and market at ~60% (from -150/+130), expect ML edge
        types: set[type[MoneylineEdge] | type[SpreadEdge] | type[TotalEdge]] = {
            type(e) for e in edges
        }
        assert MoneylineEdge in types or SpreadEdge in types or TotalEdge in types
        assert len(edges) >= 1

    def test_ml_only(self) -> None:
        """Row with only ML data -> only MoneylineEdge possible."""
        preds: DataFrame = _make_predictions(home_win_prob=0.70)
        odds: DataFrame = _make_long_odds()
        # Remove spread and total from odds
        odds = odds[odds["market"] == "moneyline"].copy()
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        row: Series = joined.iloc[0]
        edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = compute_game_edges(
            row, margin_std=13.0, total_std=13.0
        )
        for e in edges:
            assert isinstance(e, MoneylineEdge)

    def test_no_edge_returns_empty(self) -> None:
        """Model agrees with market -> empty list."""
        # At -110/-110, fair prob is ~50%. Model at 50% -> no ML edge.
        # At spread=-7, model_spread=-7 -> no spread edge.
        # At total=45, model_total=45 -> no total edge.
        preds: DataFrame = _make_predictions(
            home_win_prob=0.50,
            model_spread=-7.0,
            model_total=45.0,
        )
        odds: DataFrame = _make_long_odds(
            ml_home=-110,
            ml_away=-110,
            spread_home=-7.0,
            total_line=45.0,
        )
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        row: Series = joined.iloc[0]
        edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = compute_game_edges(
            row, margin_std=13.0, total_std=13.0
        )
        assert edges == []

    def test_missing_model_total(self) -> None:
        """No model_total -> no TotalEdge attempted."""
        preds: DataFrame = _make_predictions(home_win_prob=0.70, model_total=float("nan"))
        odds: DataFrame = _make_long_odds()
        joined: DataFrame = join_predictions_to_odds(preds, odds)
        row: Series = joined.iloc[0]
        edges: list[MoneylineEdge | SpreadEdge | TotalEdge] = compute_game_edges(
            row, margin_std=13.0, total_std=13.0
        )
        for e in edges:
            assert not isinstance(e, TotalEdge)


# ---------------------------------------------------------------------------
# TestBuildEdgeReport
# ---------------------------------------------------------------------------


class TestBuildEdgeReport:
    """Tests for build_edge_report()."""

    def test_report_schema(self) -> None:
        """Output has all expected columns."""
        preds: DataFrame = _make_predictions(
            home_win_prob=0.70, model_spread=-7.0, model_total=52.0
        )
        odds: DataFrame = _make_long_odds(
            ml_home=-150, ml_away=130, spread_home=-3.0, total_line=45.0
        )
        report: DataFrame = build_edge_report(
            preds,
            odds,
            margin_std=13.0,
            total_std=13.0,
        )
        for col in _REPORT_COLUMNS:
            assert col in report.columns, f"Missing column: {col}"

        moneyline = report.loc[report["market_type"] == "moneyline"].iloc[0]

        assert 0.0 < moneyline["market_value"] < 1.0
        assert abs(moneyline["american_odds"]) >= 100

        spread = report.loc[report["market_type"] == "spread"].iloc[0]

        assert spread["market_value"] == pytest.approx(-3.0)
        assert spread["american_odds"] == -110

        total = report.loc[report["market_type"] == "total"].iloc[0]

        assert total["market_value"] == pytest.approx(45.0)
        assert total["american_odds"] == -110

        assert "american_odds" in report.columns

    def test_kelly_stake_calculation(self) -> None:
        """kelly_stake = bankroll * kelly_multiplier * kelly_frac, capped."""
        preds: DataFrame = _make_predictions(home_win_prob=0.70)
        odds: DataFrame = _make_long_odds(ml_home=-150, ml_away=130)
        report: DataFrame = build_edge_report(
            preds,
            odds,
            margin_std=13.0,
            total_std=13.0,
            bankroll=1000.0,
            kelly_multiplier=0.25,
        )
        if not report.empty:
            for _, row in report.iterrows():
                expected_stake = 1000.0 * 0.25 * row["kelly_frac"]
                max_stake: float = 1000.0 * 0.25
                assert row["kelly_stake"] == pytest.approx(min(expected_stake, max_stake), abs=1e-6)

    def test_edge_strength_populated(self) -> None:
        """classify_edge_strength is applied to every row."""
        preds: DataFrame = _make_predictions(
            home_win_prob=0.70, model_spread=-7.0, model_total=52.0
        )
        odds: DataFrame = _make_long_odds(
            ml_home=-150, ml_away=130, spread_home=-3.0, total_line=45.0
        )
        report: DataFrame = build_edge_report(preds, odds, margin_std=13.0, total_std=13.0)
        if not report.empty:
            for val in report["edge_strength"]:
                assert val in {"strong", "moderate", "lean", "no_edge"}

    def test_empty_when_no_odds(self) -> None:
        """No matching odds -> empty report."""
        preds: DataFrame = _make_predictions(game_id="2026_01_KC_LAC")
        odds: DataFrame = _make_long_odds(game_id="2026_01_BUF_MIA")
        report: DataFrame = build_edge_report(preds, odds, margin_std=13.0, total_std=13.0)
        assert len(report) == 0
        for col in _REPORT_COLUMNS:
            assert col in report.columns

    def test_preserves_price_used_for_each_edge(self) -> None:
        predictions: DataFrame = _make_predictions()
        odds: DataFrame = _make_long_odds(
            ml_home=-151.0,
            ml_away=131.0,
            spread_odds_home=-107.0,
            spread_odds_away=-113.0,
            over_odds=-104.0,
            under_odds=-116.0,
        )

        report: DataFrame = build_edge_report(
            predictions,
            odds,
            margin_std=13.5,
            total_std=14.0,
        )

        expected_prices: dict[tuple[str, str], int] = {
            ("moneyline", "home"): -151,
            ("moneyline", "away"): 131,
            ("spread", "home"): -107,
            ("spread", "away"): -113,
            ("total", "over"): -104,
            ("total", "under"): -116,
        }

        assert not report.empty

        for row in report.itertuples(index=False):
            assert row.american_odds == expected_prices[(row.market_type, row.side)]


# ---------------------------------------------------------------------------
# TestRankEdges
# ---------------------------------------------------------------------------


class TestRankEdges:
    """Tests for rank_edges()."""

    def _make_report(self) -> pd.DataFrame:
        """Helper: build a report with known EVs."""
        rows: list[dict[str, float | str]] = [
            dict.fromkeys(_REPORT_COLUMNS, "") | {"ev": 0.08, "edge_strength": "strong"},
            dict.fromkeys(_REPORT_COLUMNS, "") | {"ev": 0.03, "edge_strength": "moderate"},
            dict.fromkeys(_REPORT_COLUMNS, "") | {"ev": 0.01, "edge_strength": "lean"},
            dict.fromkeys(_REPORT_COLUMNS, "") | {"ev": -0.02, "edge_strength": "no_edge"},
        ]
        return pd.DataFrame(rows)

    def test_filters_negative_ev(self) -> None:
        """Rows with ev <= 0 are removed with default min_ev=0."""
        ranked: DataFrame = rank_edges(self._make_report())
        assert len(ranked) == 3
        assert all(ranked["ev"] > 0)

    def test_sorts_by_ev_descending(self) -> None:
        """Highest EV first."""
        ranked: DataFrame = rank_edges(self._make_report())
        evs: list[Any] = ranked["ev"].tolist()
        assert evs == sorted(evs, reverse=True)

    def test_custom_min_ev(self) -> None:
        """min_ev=0.03 filters lean edges."""
        ranked: DataFrame = rank_edges(self._make_report(), min_ev=0.03)
        assert len(ranked) == 1
        assert ranked.iloc[0]["ev"] == 0.08


class TestComputeGameEdgesHomeProbDerivation:
    """Verify defensive derivation of home_win_prob (Unit 11 / recommendations/H2)."""

    def test_uses_home_win_prob_when_present(self) -> None:
        row = pd.Series(
            {
                "home_win_prob": 0.65,
                "ml_home": -150,
                "ml_away": 130,
            }
        )
        edges = compute_game_edges(row, margin_std=10.0, total_std=10.0)
        ml_edges = [e for e in edges if hasattr(e, "model_prob")]
        assert len(ml_edges) == 1
        assert ml_edges[0].model_prob == pytest.approx(0.65)

    def test_derives_from_away_win_prob_when_home_missing(self) -> None:
        row = pd.Series(
            {
                "away_win_prob": 0.35,
                "ml_home": -150,
                "ml_away": 130,
            }
        )
        edges = compute_game_edges(row, margin_std=10.0, total_std=10.0)
        ml_edges = [e for e in edges if hasattr(e, "model_prob")]
        assert len(ml_edges) == 1
        assert ml_edges[0].model_prob == pytest.approx(0.65)

    def test_no_edge_when_neither_prob_present(self) -> None:
        row = pd.Series(
            {
                "ml_home": -150,
                "ml_away": 130,
            }
        )
        edges = compute_game_edges(row, margin_std=10.0, total_std=10.0)
        ml_edges = [e for e in edges if hasattr(e, "model_prob")]
        assert len(ml_edges) == 0
