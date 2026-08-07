"""Unit tests for current line-shopping response schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.lines import (
    LineOffer,
    LineOutcomeGuidance,
    LineShoppingGame,
    LineShoppingList,
)


def offer() -> LineOffer:
    return LineOffer(
        provider="the_odds_api",
        provider_event_id="event-1",
        sportsbook="draftkings",
        sportsbook_updated_at=datetime(2026, 8, 5, 22, tzinfo=UTC),
        market_fetched_at=datetime(2026, 8, 5, 22, 5, tzinfo=UTC),
        commence_time=datetime(2026, 9, 10, 0, 15, tzinfo=UTC),
        is_live=False,
        market="spread",
        side="away",
        line=3.5,
        american_odds=-110,
        is_best_line=False,
        is_best_price=True,
    )


def test_line_offer_preserves_exact_quote_identity() -> None:
    row = offer()
    assert row.sportsbook == "draftkings"
    assert row.line == 3.5
    assert row.american_odds == -110
    assert row.is_best_price is True


def test_line_offer_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        # pyrefly: ignore [unexpected-keyword]
        LineOffer(**offer().model_dump(), unexpected="x")


def test_line_shopping_list_composes_games_and_offers() -> None:
    game = LineShoppingGame(
        game_id="2026_01_NE_SEA",
        season="2026-2027",
        week=1,
        game_date="2026-09-09",
        away_team="New England Patriots",
        home_team="Seattle Seahawks",
        offers=[offer()],
    )
    response = LineShoppingList(
        season="2026-2027",
        week=1,
        items=[game],
        total=1,
        sportsbooks=("draftkings",),
        market_fetched_at=(datetime(2026, 8, 5, 22, 5, tzinfo=UTC),),
    )
    assert response.items[0].offers[0].provider_event_id == "event-1"
    assert response.total == 1


def test_schemas_are_frozen() -> None:
    row = offer()
    with pytest.raises(ValidationError):
        # pyrefly: ignore [read-only]
        row.sportsbook = "fanduel"


def test_line_offer_and_outcome_guidance_preserve_model_contract() -> None:
    row = offer().model_copy(
        update={
            "model_status": "available",
            "model_value": -1.5,
            "model_probability": 0.56,
            "expected_value": 0.069,
            "is_model_approved": True,
            "is_best_model_approved_offer": True,
            "product_id": "weekly-product",
            "product_run_id": "weekly-run",
        }
    )
    guidance = LineOutcomeGuidance(
        side="away",
        model_status="available",
        model_value=-1.5,
        playable_line=2.3,
        reference_odds=-110,
        product_id="weekly-product",
        product_run_id="weekly-run",
    )
    assert row.is_model_approved is True
    assert row.expected_value == pytest.approx(0.069)
    assert guidance.playable_line == pytest.approx(2.3)
