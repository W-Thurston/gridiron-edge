"""Unit tests for historical portfolio serialization."""

import pandas as pd

from gridiron_edge.api.serializers.portfolio import serialize_bets, serialize_transactions


def test_serializes_parlay_and_funding_evidence() -> None:
    result = serialize_bets(
        pd.DataFrame(
            [
                {
                    "bet_id": "b1",
                    "source_bet_id": "DK-1",
                    "game_id": None,
                    "description": "16-pick parlay",
                    "placed_at": pd.Timestamp("2026-09-16T18:00:35Z"),
                    "market_type": "parlay",
                    "side": None,
                    "line": None,
                    "odds": 47787,
                    "stake": 1.0,
                    "book": "draftkings",
                    "funding_type": "cash",
                    "paid_amount": None,
                    "potential_payout": 478.87,
                    "status": "open",
                    "pnl": None,
                }
            ]
        )
    )
    row = result.items[0]
    assert row.source_bet_id == "DK-1"
    assert row.game_id is None
    assert row.description == "16-pick parlay"
    assert row.potential_payout == 478.87


def test_serializes_transaction_source_balance() -> None:
    result = serialize_transactions(
        pd.DataFrame(
            [
                {
                    "txn_id": "t1",
                    "source_transaction_id": "TXN-1",
                    "timestamp": pd.Timestamp("2026-09-16T18:00:00Z"),
                    "txn_type": "bet_placed",
                    "amount": 1.0,
                    "balance_after": 41.96,
                    "reference_id": None,
                    "note": None,
                }
            ]
        )
    )
    row = result.items[0]
    assert row.source_transaction_id == "TXN-1"
    assert row.balance_after == 41.96
