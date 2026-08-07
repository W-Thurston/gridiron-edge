"""Exhaustive selected-product guidance for current Line Shopping offers."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.market.edge import expected_value
from gridiron_edge.market.line_shopping import (
    REFERENCE_ODDS,
    evaluate_line_shopping_guidance,
)
from gridiron_edge.market.odds_math import american_to_implied_prob

GAME_ID = "2026_01_NE_SEA"


def _product(**overrides: object) -> DataFrame:
    row: dict[str, object] = {
        "game_id": GAME_ID,
        "win_status": "available",
        "away_win_prob": 0.4,
        "home_win_prob": 0.6,
        "spread_status": "available",
        "model_spread": -2.0,
        "spread_uncertainty": 10.0,
        "total_status": "available",
        "model_total": 45.0,
        "total_uncertainty": 10.0,
        "product_id": "weekly-product",
        "product_run_id": "weekly-run",
    }
    row.update(overrides)
    return DataFrame([row])


def _quote(
    *,
    sportsbook: str,
    market: str,
    side: str,
    odds: int,
    line: float | None,
) -> dict[str, object]:
    return {
        "game_id": GAME_ID,
        "market": market,
        "side": side,
        "odds": odds,
        "line": line,
        "sportsbook": sportsbook,
    }


def _quotes() -> DataFrame:
    return DataFrame(
        [
            _quote(sportsbook="draftkings", market="moneyline", side="home", odds=-140, line=None),
            _quote(sportsbook="fanduel", market="moneyline", side="away", odds=150, line=None),
            _quote(sportsbook="draftkings", market="spread", side="home", odds=-110, line=-1.5),
            _quote(sportsbook="fanduel", market="spread", side="away", odds=-110, line=1.5),
            _quote(sportsbook="draftkings", market="total", side="over", odds=-110, line=44.0),
            _quote(sportsbook="fanduel", market="total", side="under", odds=-110, line=44.0),
        ]
    )


class TestEvaluateLineShoppingGuidance:
    def test_retains_every_exact_offer_and_input_frames(self) -> None:
        product = _product()
        quotes = _quotes()
        original_product = product.copy(deep=True)
        original_quotes = quotes.copy(deep=True)

        result = evaluate_line_shopping_guidance(product, quotes)

        assert len(result.offers) == len(quotes)
        assert len(result.guidance) == 6
        pd.testing.assert_frame_equal(product, original_product)
        pd.testing.assert_frame_equal(quotes, original_quotes)

    def test_evaluates_both_sides_instead_of_selecting_one_recommendation(self) -> None:
        result = evaluate_line_shopping_guidance(_product(), _quotes())

        moneyline = result.offers.loc[result.offers["market"] == "moneyline"]
        assert set(moneyline["side"]) == {"away", "home"}
        by_side = moneyline.set_index("side")
        assert by_side.loc["home", "model_probability"] == pytest.approx(0.6)
        assert by_side.loc["away", "model_probability"] == pytest.approx(0.4)
        assert by_side.loc["home", "expected_value"] == pytest.approx(expected_value(0.6, -140))
        assert by_side.loc["away", "expected_value"] == pytest.approx(expected_value(0.4, 150))

    def test_retains_negative_expected_value_offer(self) -> None:
        quotes = DataFrame(
            [
                _quote(
                    sportsbook="draftkings",
                    market="moneyline",
                    side="away",
                    odds=120,
                    line=None,
                )
            ]
        )

        offer = evaluate_line_shopping_guidance(_product(), quotes).offers.iloc[0]

        assert offer["expected_value"] < 0.0
        assert bool(offer["is_model_approved"]) is False
        assert bool(offer["is_best_model_approved_offer"]) is False

    def test_break_even_offer_is_not_approved(self) -> None:
        probability = american_to_implied_prob(REFERENCE_ODDS)
        quotes = DataFrame(
            [
                _quote(
                    sportsbook="draftkings",
                    market="moneyline",
                    side="home",
                    odds=REFERENCE_ODDS,
                    line=None,
                )
            ]
        )
        product = _product(
            home_win_prob=probability,
            away_win_prob=1.0 - probability,
        )

        offer = evaluate_line_shopping_guidance(product, quotes).offers.iloc[0]

        assert offer["expected_value"] == pytest.approx(0.0, abs=1e-12)
        assert bool(offer["is_model_approved"]) is False

    def test_spread_side_lines_use_home_spread_sign_convention(self) -> None:
        result = evaluate_line_shopping_guidance(_product(), _quotes())
        spread = result.offers.loc[result.offers["market"] == "spread"].set_index("side")

        home_probability = spread.loc["home", "model_probability"]
        away_probability = spread.loc["away", "model_probability"]
        assert home_probability + away_probability == pytest.approx(1.0)
        assert home_probability > 0.5
        assert away_probability < 0.5
        assert spread.loc["home", "expected_value"] == pytest.approx(
            expected_value(home_probability, -110)
        )
        assert spread.loc["away", "expected_value"] == pytest.approx(
            expected_value(away_probability, -110)
        )
        assert bool(spread.loc["home", "is_model_approved"]) is False
        assert bool(spread.loc["away", "is_model_approved"]) is False

    def test_playable_boundaries_use_reference_price_without_rounding(self) -> None:
        result = evaluate_line_shopping_guidance(_product(), _quotes())
        guidance = result.guidance.set_index(["market", "side"])
        z_value = 0.05971709978532289

        assert guidance.loc[("spread", "home"), "model_value"] == pytest.approx(-2.0)
        assert guidance.loc[("spread", "away"), "model_value"] == pytest.approx(2.0)
        assert guidance.loc[("spread", "home"), "playable_line"] == pytest.approx(
            -2.0 + 10.0 * z_value
        )
        assert guidance.loc[("spread", "away"), "playable_line"] == pytest.approx(
            2.0 + 10.0 * z_value
        )
        assert guidance.loc[("total", "over"), "playable_line"] == pytest.approx(
            45.0 - 10.0 * z_value
        )
        assert guidance.loc[("total", "under"), "playable_line"] == pytest.approx(
            45.0 + 10.0 * z_value
        )
        assert guidance.loc[("spread", "home"), "reference_odds"] == REFERENCE_ODDS

    def test_moneyline_guidance_exposes_fair_american_odds(self) -> None:
        guidance = evaluate_line_shopping_guidance(_product(), _quotes()).guidance
        moneyline = guidance.loc[guidance["market"] == "moneyline"].set_index("side")

        assert moneyline.loc["home", "fair_american_odds"] == -150
        assert moneyline.loc["away", "fair_american_odds"] == 150
        assert pd.isna(moneyline.loc["home", "playable_line"])
        assert pd.isna(moneyline.loc["home", "reference_odds"])

    def test_preserves_maximum_expected_value_ties(self) -> None:
        quotes = DataFrame(
            [
                _quote(
                    sportsbook="draftkings",
                    market="moneyline",
                    side="home",
                    odds=-140,
                    line=None,
                ),
                _quote(
                    sportsbook="fanduel",
                    market="moneyline",
                    side="home",
                    odds=-140,
                    line=None,
                ),
                _quote(
                    sportsbook="betmgm",
                    market="moneyline",
                    side="home",
                    odds=-150,
                    line=None,
                ),
            ]
        )

        offers = evaluate_line_shopping_guidance(_product(), quotes).offers
        preferred = offers.loc[offers["is_best_model_approved_offer"]]

        assert set(preferred["sportsbook"]) == {"draftkings", "fanduel"}

    @pytest.mark.parametrize(
        ("product_changes", "market", "expected_status"),
        [
            (
                {
                    "win_status": "forecast_unavailable",
                    "away_win_prob": pd.NA,
                    "home_win_prob": pd.NA,
                },
                "moneyline",
                "model_unavailable",
            ),
            (
                {
                    "spread_status": "source_unavailable",
                    "model_spread": pd.NA,
                    "spread_uncertainty": pd.NA,
                },
                "spread",
                "model_unavailable",
            ),
            (
                {
                    "total_status": "uncertainty_unavailable",
                    "total_uncertainty": pd.NA,
                },
                "total",
                "uncertainty_unavailable",
            ),
        ],
    )
    def test_preserves_unavailable_guidance_without_removing_quotes(
        self,
        product_changes: dict[str, object],
        market: str,
        expected_status: str,
    ) -> None:
        quote_frame = _quotes()
        quotes: DataFrame = quote_frame.loc[
            quote_frame["market"] == market,
            :,
        ].copy()

        result = evaluate_line_shopping_guidance(
            _product(**product_changes),
            quotes,
        )

        assert len(result.offers) == len(quotes)
        assert set(result.offers["model_status"]) == {expected_status}
        assert result.offers["model_probability"].isna().all()
        assert result.offers["expected_value"].isna().all()
        assert result.offers["is_model_approved"].isna().all()
        assert set(result.guidance["model_status"]) == {expected_status}

    def test_rejects_quotes_without_selected_product_identity(self) -> None:
        quotes = _quotes().copy()
        quotes.loc[0, "game_id"] = "2026_01_BUF_MIA"

        with pytest.raises(ValueError, match="no selected-product row"):
            evaluate_line_shopping_guidance(_product(), quotes)

    def test_rejects_duplicate_product_game_identity(self) -> None:
        product = pd.concat([_product(), _product()], ignore_index=True)

        with pytest.raises(ValueError, match="duplicate game_id"):
            evaluate_line_shopping_guidance(product, _quotes())
