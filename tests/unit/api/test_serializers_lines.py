"""Unit tests for line-shopping serialization."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.serializers.lines import serialize_line_shopping_list
from gridiron_edge.market.line_shopping import (
    evaluate_line_shopping_guidance,
)


def quotes() -> pd.DataFrame:
    base = {
        "fetched_at": pd.Timestamp("2026-08-05T22:05:33Z"),
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook_updated_at": pd.Timestamp("2026-08-05T22:05:03Z"),
        "commence_time": pd.Timestamp("2026-09-10T00:15:00Z"),
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_NE_SEA",
        "game_date": "2026-09-09",
        "away_team": "New England Patriots",
        "home_team": "Seattle Seahawks",
        "market": "spread",
        "side": "away",
    }
    return pd.DataFrame(
        [
            {**base, "sportsbook": "draftkings", "odds": -110.0, "line": 3.5},
            {**base, "sportsbook": "betrivers", "odds": -114.0, "line": 4.5},
        ]
    )


def product() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_SEA",
                "win_status": "available",
                "away_win_prob": 0.48,
                "home_win_prob": 0.52,
                "spread_status": "available",
                "model_spread": -1.0,
                "spread_uncertainty": 13.0,
                "total_status": "available",
                "model_total": 47.0,
                "total_uncertainty": 13.0,
                "product_id": "weekly-product",
                "product_run_id": "weekly-run",
            }
        ]
    )


def test_serializes_grouped_games_and_partial_coverage() -> None:
    evaluated = evaluate_line_shopping_guidance(product(), quotes())
    result = serialize_line_shopping_list(
        evaluated.offers,
        season="2026-2027",
        week=1,
        market=None,
        guidance=evaluated.guidance,
    )
    assert result.total == 1
    assert result.sportsbooks == ("betrivers", "draftkings")
    assert len(result.items[0].offers) == 2
    best_line = next(row for row in result.items[0].offers if row.is_best_line)
    assert best_line.sportsbook == "betrivers"
    assert best_line.line == 4.5
    assert best_line.model_status == "available"
    assert best_line.expected_value is not None
    assert best_line.product_id == "weekly-product"
    assert len(result.items[0].guidance) == 1
    assert result.items[0].guidance[0].reference_odds == -110


def test_serializes_empty_scope_without_fabricated_games() -> None:
    result = serialize_line_shopping_list(
        evaluate_line_shopping_guidance(product(), quotes().head(0)).offers,
        season="2026-2027",
        week=2,
        market="spread",
    )
    assert result.items == []
    assert result.total == 0
    assert result.sportsbooks == ()


def test_sorts_games_by_kickoff_then_team_identity() -> None:
    first = quotes().copy()
    first["game_id"] = "2026_01_ARI_LAC"
    first["away_team"] = "Arizona Cardinals"
    first["home_team"] = "Los Angeles Chargers"
    first["commence_time"] = pd.Timestamp("2026-09-13T20:25:00Z")

    second = quotes().copy()
    second["game_id"] = "2026_01_ATL_PIT"
    second["away_team"] = "Atlanta Falcons"
    second["home_team"] = "Pittsburgh Steelers"
    second["commence_time"] = pd.Timestamp("2026-09-13T17:00:00Z")

    third = quotes().copy()
    third["game_id"] = "2026_01_BAL_IND"
    third["away_team"] = "Baltimore Ravens"
    third["home_team"] = "Indianapolis Colts"
    third["commence_time"] = pd.NaT

    rows = pd.concat([first, second, third], ignore_index=True)
    classified = rows.assign(is_best_line=False, is_best_price=True)
    result = serialize_line_shopping_list(
        classified,
        season="2026-2027",
        week=1,
        market="spread",
    )

    assert [game.game_id for game in result.items] == [
        "2026_01_ATL_PIT",
        "2026_01_ARI_LAC",
        "2026_01_BAL_IND",
    ]
