"""Unit tests for portfolio schemas."""

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.portfolio import BetRow, RecordBetRequest, TransactionRow


def test_historical_bet_fields_are_supported() -> None:
    row = BetRow(
        source_bet_id="DK-1",
        description="16-pick parlay",
        funding_type="bonus",
        paid_amount=0.20,
        potential_payout=10.0,
    )
    assert row.source_bet_id == "DK-1"
    assert row.game_id is None
    assert row.funding_type == "bonus"


def test_transaction_evidence_fields_are_supported() -> None:
    row = TransactionRow(source_transaction_id="TXN-1", balance_after=41.96)
    assert row.source_transaction_id == "TXN-1"
    assert row.balance_after == 41.96


def test_live_request_contract_remains_single_game() -> None:
    request = RecordBetRequest(
        game_id="2026_01_KC_LAC",
        market_type="moneyline",
        side="away",
        odds=175,
        stake=25.0,
        book="fanduel",
    )
    assert request.market_type == "moneyline"
    with pytest.raises(ValidationError):
        RecordBetRequest(
            game_id="x", market_type="parlay", side="away", odds=100, stake=1.0, book="draftkings"
        )
