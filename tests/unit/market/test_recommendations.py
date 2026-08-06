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
    join_predictions_to_current_odds,
    join_predictions_to_odds,
    pivot_current_odds_to_book_wide,
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


class TestPivotCurrentOddsToBookWide:
    """Tests for current-snapshot sportsbook-preserving preparation."""

    @staticmethod
    def _canonical_book_rows(
        *,
        sportsbook: str | None = "draftkings",
        provider_event_id: str | None = "event-1",
        ml_home: float = -150.0,
    ) -> DataFrame:
        rows = _make_long_odds(ml_home=ml_home).copy()
        rows["provider"] = "the_odds_api" if sportsbook is not None else "nflverse"
        rows["provider_event_id"] = provider_event_id
        rows["sportsbook"] = sportsbook
        rows["sportsbook_updated_at"] = pd.Timestamp("2026-09-05 11:59:00", tz="UTC")
        rows["commence_time"] = pd.Timestamp("2026-09-06 00:20:00", tz="UTC")
        rows["is_live"] = False
        rows["fetched_at"] = pd.Timestamp("2026-09-05 12:00:00", tz="UTC")
        return rows

    def test_preserves_two_sportsbooks_as_two_wide_rows(self) -> None:
        draftkings = self._canonical_book_rows(sportsbook="draftkings", ml_home=-150.0)
        fanduel = self._canonical_book_rows(sportsbook="fanduel", ml_home=-145.0)
        wide = pivot_current_odds_to_book_wide(pd.concat([draftkings, fanduel], ignore_index=True))
        assert len(wide) == 2
        assert set(wide["sportsbook"]) == {"draftkings", "fanduel"}
        prices = dict(zip(wide["sportsbook"], wide["ml_home"], strict=True))
        assert prices == {"draftkings": -150.0, "fanduel": -145.0}

    def test_preserves_provider_and_timestamp_provenance(self) -> None:
        wide = pivot_current_odds_to_book_wide(self._canonical_book_rows())
        row = wide.iloc[0]
        assert row["provider"] == "the_odds_api"
        assert row["provider_event_id"] == "event-1"
        assert row["sportsbook"] == "draftkings"
        assert row["fetched_at"] == pd.Timestamp("2026-09-05 12:00:00", tz="UTC")
        assert row["sportsbook_updated_at"] == pd.Timestamp(
            "2026-09-05 11:59:00",
            tz="UTC",
        )
        assert row["commence_time"] == pd.Timestamp("2026-09-06 00:20:00", tz="UTC")
        assert not row["is_live"]

    def test_preserves_null_sportsbook_consensus_group(self) -> None:
        wide = pivot_current_odds_to_book_wide(
            self._canonical_book_rows(
                sportsbook=None,
                provider_event_id=None,
            )
        )
        assert len(wide) == 1
        assert wide.loc[0, "provider"] == "nflverse"
        assert pd.isna(wide.loc[0, "sportsbook"])
        assert pd.isna(wide.loc[0, "provider_event_id"])

    def test_incomplete_market_remains_local_to_one_book(self) -> None:
        draftkings = self._canonical_book_rows(sportsbook="draftkings")
        fanduel = self._canonical_book_rows(sportsbook="fanduel")
        fanduel = fanduel.loc[fanduel["market"] != "spread", :]
        wide = pivot_current_odds_to_book_wide(
            pd.concat([draftkings, fanduel], ignore_index=True)
        ).set_index("sportsbook")
        assert wide.loc["draftkings", "spread_odds_home"] == -110.0
        assert pd.isna(wide.loc["fanduel", "spread_odds_home"])
        assert wide.loc["fanduel", "ml_home"] == -150.0

    def test_rejects_duplicate_current_book_side(self) -> None:
        rows = self._canonical_book_rows()
        duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
        with pytest.raises(
            ValueError,
            match="duplicate provider-event-book-market-side",
        ):
            pivot_current_odds_to_book_wide(duplicate)

    def test_rejects_mixed_group_provenance(self) -> None:
        rows = self._canonical_book_rows()
        rows.loc[0, "fetched_at"] = pd.Timestamp("2026-09-05 13:00:00", tz="UTC")
        with pytest.raises(ValueError, match="mixed fetched_at"):
            pivot_current_odds_to_book_wide(rows)

    def test_requires_canonical_provenance_columns(self) -> None:
        with pytest.raises(ValueError, match="provider_event_id"):
            pivot_current_odds_to_book_wide(
                self._canonical_book_rows().drop(columns="provider_event_id")
            )

    def test_empty_input_has_locked_book_wide_schema(self) -> None:
        result = pivot_current_odds_to_book_wide(DataFrame())
        assert result.empty
        assert list(result.columns) == [
            "provider",
            "provider_event_id",
            "sportsbook",
            "game_id",
            "fetched_at",
            "sportsbook_updated_at",
            "commence_time",
            "is_live",
            "ml_home",
            "ml_away",
            "spread_line_home",
            "spread_odds_home",
            "spread_odds_away",
            "total_line",
            "over_odds",
            "under_odds",
        ]


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


class TestJoinPredictionsToCurrentOdds:
    """Tests for the sportsbook-preserving current recommendation join."""

    @staticmethod
    def _book_rows(sportsbook: str, ml_home: float) -> DataFrame:
        return TestPivotCurrentOddsToBookWide._canonical_book_rows(
            sportsbook=sportsbook,
            provider_event_id="event-1",
            ml_home=ml_home,
        )

    def test_one_prediction_replicates_across_sportsbooks(self) -> None:
        odds = pd.concat(
            [
                self._book_rows("draftkings", -150.0),
                self._book_rows("fanduel", -145.0),
            ],
            ignore_index=True,
        )
        joined = join_predictions_to_current_odds(_make_predictions(), odds)
        assert len(joined) == 2
        assert set(joined["sportsbook"]) == {"draftkings", "fanduel"}
        prices = dict(zip(joined["sportsbook"], joined["ml_home"], strict=True))
        assert prices == {"draftkings": -150.0, "fanduel": -145.0}
        assert set(joined["home_win_prob"]) == {0.65}

    def test_incomplete_book_market_does_not_remove_other_book(self) -> None:
        draftkings = self._book_rows("draftkings", -150.0)
        fanduel = self._book_rows("fanduel", -145.0)
        fanduel = fanduel.loc[fanduel["market"] != "spread", :]
        joined = join_predictions_to_current_odds(
            _make_predictions(),
            pd.concat([draftkings, fanduel], ignore_index=True),
        ).set_index("sportsbook")
        assert joined.loc["draftkings", "spread_odds_home"] == -110.0
        assert pd.isna(joined.loc["fanduel", "spread_odds_home"])
        assert joined.loc["fanduel", "ml_home"] == -145.0

    def test_rejects_duplicate_prediction_game_ids(self) -> None:
        predictions = pd.concat(
            [_make_predictions(), _make_predictions()],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="one prediction row per game_id"):
            join_predictions_to_current_odds(
                predictions,
                self._book_rows("draftkings", -150.0),
            )

    def test_accepts_already_book_wide_input(self) -> None:
        wide = pivot_current_odds_to_book_wide(self._book_rows("draftkings", -150.0))
        joined = join_predictions_to_current_odds(_make_predictions(), wide)
        assert len(joined) == 1
        assert joined.loc[0, "sportsbook"] == "draftkings"

    def test_no_matching_game_returns_empty(self) -> None:
        joined = join_predictions_to_current_odds(
            _make_predictions(game_id="2026_01_BUF_MIA"),
            self._book_rows("draftkings", -150.0),
        )
        assert joined.empty


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

    def test_omitted_bankroll_preserves_fraction_without_dollar_stake(
        self,
    ) -> None:
        predictions = _make_predictions(
            home_win_prob=0.70,
            model_spread=-7.0,
            model_total=52.0,
        )
        odds = _make_long_odds(
            ml_home=-150,
            ml_away=130,
            spread_home=-3.0,
            total_line=45.0,
        )

        report = build_edge_report(
            predictions,
            odds,
            margin_std=13.0,
            total_std=13.0,
            bankroll=None,
            kelly_multiplier=0.25,
        )

        assert not report.empty
        assert report["kelly_frac"].notna().all()
        assert report["kelly_stake"].isna().all()

    def test_zero_bankroll_produces_zero_dollar_stake(
        self,
    ) -> None:
        predictions = _make_predictions(
            home_win_prob=0.70,
            model_spread=-7.0,
            model_total=52.0,
        )
        odds = _make_long_odds(
            ml_home=-150,
            ml_away=130,
            spread_home=-3.0,
            total_line=45.0,
        )

        report = build_edge_report(
            predictions,
            odds,
            margin_std=13.0,
            total_std=13.0,
            bankroll=0.0,
            kelly_multiplier=0.25,
        )

        assert not report.empty
        assert report["kelly_frac"].notna().all()
        assert (report["kelly_stake"] == 0.0).all()

    @pytest.mark.parametrize(
        ("bankroll", "kelly_multiplier"),
        [
            (-1.0, 0.25),
            (1000.0, -0.01),
            (1000.0, 1.01),
        ],
    )
    def test_rejects_invalid_sizing_inputs(
        self,
        bankroll: float,
        kelly_multiplier: float,
    ) -> None:
        with pytest.raises(ValueError):
            build_edge_report(
                _make_predictions(),
                _make_long_odds(),
                margin_std=13.0,
                total_std=13.0,
                bankroll=bankroll,
                kelly_multiplier=kelly_multiplier,
            )

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


class TestCurrentEdgeRowProvenance:
    """Tests for quote identity retained on current edge rows."""

    @staticmethod
    def _book_rows(
        *,
        sportsbook: str,
        event_id: str,
        ml_home: float,
        ml_away: float,
    ) -> DataFrame:
        rows = TestPivotCurrentOddsToBookWide._canonical_book_rows(
            sportsbook=sportsbook,
            provider_event_id=event_id,
            ml_home=ml_home,
        )
        moneyline_away = (rows["market"] == "moneyline") & (rows["side"] == "away")
        rows.loc[moneyline_away, "odds"] = ml_away
        return rows

    def test_current_report_carries_exact_book_provenance(self) -> None:
        odds = pd.concat(
            [
                self._book_rows(
                    sportsbook="draftkings",
                    event_id="event-1",
                    ml_home=-150.0,
                    ml_away=130.0,
                ),
                self._book_rows(
                    sportsbook="fanduel",
                    event_id="event-1",
                    ml_home=-140.0,
                    ml_away=120.0,
                ),
            ],
            ignore_index=True,
        )
        report = build_edge_report(
            _make_predictions(
                home_win_prob=0.75,
                model_spread=-7.0,
                model_total=52.0,
            ),
            odds,
            margin_std=13.0,
            current_snapshot=True,
            total_std=13.0,
        )
        assert not report.empty
        assert set(report["sportsbook"]) == {"draftkings", "fanduel"}
        assert set(report["provider"]) == {"the_odds_api"}
        assert set(report["provider_event_id"]) == {"event-1"}
        assert report["market_fetched_at"].notna().all()
        assert report["sportsbook_updated_at"].notna().all()
        assert report["commence_time"].notna().all()

    def test_moneyline_price_matches_the_same_output_sportsbook(self) -> None:
        odds = pd.concat(
            [
                self._book_rows(
                    sportsbook="draftkings",
                    event_id="event-1",
                    ml_home=-150.0,
                    ml_away=130.0,
                ),
                self._book_rows(
                    sportsbook="fanduel",
                    event_id="event-1",
                    ml_home=-140.0,
                    ml_away=120.0,
                ),
            ],
            ignore_index=True,
        )
        report = build_edge_report(
            _make_predictions(home_win_prob=0.75),
            odds,
            margin_std=None,
            current_snapshot=True,
            total_std=None,
        )
        moneyline = report.loc[report["market_type"] == "moneyline", :]
        actual = dict(
            zip(
                moneyline["sportsbook"],
                moneyline["american_odds"],
                strict=True,
            )
        )
        assert actual == {"draftkings": -150, "fanduel": -140}

    def test_legacy_report_uses_explicit_null_market_provenance(self) -> None:
        report = build_edge_report(
            _make_predictions(home_win_prob=0.75),
            _make_long_odds(ml_home=-150.0, ml_away=130.0),
            margin_std=None,
            total_std=None,
        )
        assert not report.empty
        for column in (
            "provider",
            "provider_event_id",
            "sportsbook",
            "market_fetched_at",
            "sportsbook_updated_at",
            "commence_time",
        ):
            assert report[column].isna().all()


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
    """Verify defensive derivation of home_win_prob."""

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
