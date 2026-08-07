from __future__ import annotations

import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import pytest

from gridiron_edge.market.line_shopping import classify_line_shopping_offers


def quotes(rows: list[dict[str, object]]) -> DataFrame:
    defaults: dict[str, object] = {
        "game_id": "2026_01_NE_SEA",
        "market": "spread",
        "side": "away",
        "line": 3.5,
        "odds": -110,
        "sportsbook": "draftkings",
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestBestPrice:
    def test_selects_greatest_american_price_at_exact_line(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"sportsbook": "draftkings", "odds": -110},
                    {"sportsbook": "fanduel", "odds": -105},
                    {"sportsbook": "bovada", "odds": -120},
                ]
            )
        )

        flags = result.set_index("sportsbook")["is_best_price"].to_dict()
        assert flags == {"bovada": False, "draftkings": False, "fanduel": True}

    def test_preserves_best_price_ties(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"sportsbook": "draftkings", "odds": 105},
                    {"sportsbook": "fanduel", "odds": 105},
                ]
            )
        )
        assert result["is_best_price"].tolist() == [True, True]

    def test_does_not_compare_prices_across_different_lines(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"sportsbook": "draftkings", "line": 3.5, "odds": 110},
                    {"sportsbook": "fanduel", "line": 4.5, "odds": -120},
                ]
            )
        )
        assert result["is_best_price"].tolist() == [True, True]


class TestBestLine:
    @pytest.mark.parametrize("side", ["away", "home"])
    def test_spread_selects_highest_line(self, side: str) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"sportsbook": "draftkings", "side": side, "line": -4.5},
                    {"sportsbook": "fanduel", "side": side, "line": -3.5},
                ]
            )
        )
        best = result.loc[result["is_best_line"], "sportsbook"].tolist()
        assert best == ["fanduel"]

    def test_total_over_selects_lowest_line(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"market": "total", "side": "over", "line": 44.0},
                    {"market": "total", "side": "over", "line": 45.0, "sportsbook": "fanduel"},
                ]
            )
        )
        assert result.loc[result["is_best_line"], "line"].tolist() == [44.0]

    def test_total_under_selects_highest_line(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"market": "total", "side": "under", "line": 44.0},
                    {"market": "total", "side": "under", "line": 45.0, "sportsbook": "fanduel"},
                ]
            )
        )
        assert result.loc[result["is_best_line"], "line"].tolist() == [45.0]

    def test_preserves_best_line_ties(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"sportsbook": "draftkings", "line": 4.5},
                    {"sportsbook": "fanduel", "line": 4.5},
                    {"sportsbook": "bovada", "line": 3.5},
                ]
            )
        )
        assert set(result.loc[result["is_best_line"], "sportsbook"]) == {
            "draftkings",
            "fanduel",
        }

    def test_moneyline_has_no_best_line(self) -> None:
        result = classify_line_shopping_offers(
            quotes(
                [
                    {"market": "moneyline", "side": "away", "line": None, "odds": 175},
                    {
                        "market": "moneyline",
                        "side": "away",
                        "line": None,
                        "odds": 180,
                        "sportsbook": "fanduel",
                    },
                ]
            )
        )
        assert not result["is_best_line"].any()
        assert result.loc[result["is_best_price"], "sportsbook"].tolist() == ["fanduel"]


class TestContract:
    def test_preserves_partial_coverage_and_provenance(self) -> None:
        source = quotes(
            [
                {"sportsbook": "draftkings", "provider_event_id": "event-dk"},
                {"sportsbook": "fanduel", "provider_event_id": "event-fd"},
            ]
        )
        result = classify_line_shopping_offers(source)
        assert len(result) == 2
        assert set(result["provider_event_id"]) == {"event-dk", "event-fd"}

    def test_is_deterministic_and_does_not_mutate_input(self) -> None:
        source = quotes(
            [
                {"sportsbook": "fanduel", "line": 4.5, "odds": -120},
                {"sportsbook": "draftkings", "line": 3.5, "odds": -105},
            ]
        )
        original = source.copy(deep=True)
        first = classify_line_shopping_offers(source)
        second = classify_line_shopping_offers(source.sample(frac=1, random_state=7))

        assert_frame_equal(source, original)
        assert_frame_equal(first, second)

    def test_empty_frame_returns_boolean_flag_columns(self) -> None:
        result = classify_line_shopping_offers(
            DataFrame(
                columns=[*["game_id", "market", "side", "line", "odds", "sportsbook"], "provider"]
            )
        )
        assert result.empty
        assert str(result["is_best_price"].dtype) == "bool"
        assert str(result["is_best_line"].dtype) == "bool"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"market": "props"}, "Unsupported line-shopping markets"),
            ({"side": "push"}, "Unsupported line-shopping sides"),
            ({"odds": 0}, "American odds cannot be zero"),
            ({"line": None}, "Spread and total offers require a line"),
            (
                {"market": "moneyline", "side": "away", "line": 0.5},
                "Moneyline offers must not contain a line",
            ),
        ],
    )
    def test_rejects_invalid_quote_values(
        self,
        row: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            classify_line_shopping_offers(quotes([row]))

    def test_rejects_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            classify_line_shopping_offers(DataFrame({"game_id": ["game"]}))
