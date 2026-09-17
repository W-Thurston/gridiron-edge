# tests/integration/api/test_portfolio_routes.py
"""Integration tests for recorded and historical Portfolio wagers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.betting.bankroll import _TXN_COLUMNS, _write_txn_log
from gridiron_edge.betting.ledger import _BET_COLUMNS, _write_ledger


@dataclass
class _Settings:
    repo_root: Path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return an API client whose repository is isolated to ``tmp_path``."""
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _Settings(tmp_path)
    return TestClient(app)


def test_records_manual_wager_without_fabricated_provenance(
    client: TestClient,
    tmp_path: Path,
) -> None:
    response = client.post(
        "/portfolio/bets",
        json={
            "game_id": "2026_01_KC_LAC",
            "market_type": "moneyline",
            "side": "away",
            "odds": 175,
            "stake": 25.0,
            "book": "fanduel",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["bet"]["source_bet_id"] is None
    assert body["bet"]["funding_type"] == "cash"
    assert body["bankroll_transaction_id"]
    assert body["message"] == ("Wager recorded in Gridiron Edge. No sportsbook wager was placed.")
    transactions = pd.read_parquet(tmp_path / "data/betting/bankroll_txn.parquet")
    assert transactions.iloc[0]["reference_id"] == body["bet"]["bet_id"]


def test_returns_historical_parlay_and_transaction_evidence(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bet = dict.fromkeys(_BET_COLUMNS)
    bet.update(
        {
            "bet_id": "b1",
            "source_bet_id": "DK-1",
            "description": "16-pick parlay",
            "placed_at": pd.Timestamp("2026-09-16T18:00:35Z"),
            "market_type": "parlay",
            "odds": 47787,
            "stake": 1.0,
            "book": "draftkings",
            "funding_type": "cash",
            "potential_payout": 478.87,
            "status": "open",
        }
    )
    _write_ledger(pd.DataFrame([bet], columns=_BET_COLUMNS), tmp_path)

    transaction = dict.fromkeys(_TXN_COLUMNS)
    transaction.update(
        {
            "txn_id": "t1",
            "source_transaction_id": "TXN-1",
            "timestamp": pd.Timestamp("2026-09-16T18:00:00Z"),
            "txn_type": "bet_placed",
            "amount": 1.0,
            "balance_after": 41.96,
        }
    )
    _write_txn_log(
        pd.DataFrame([transaction], columns=_TXN_COLUMNS),
        tmp_path,
    )

    bets_response = client.get("/portfolio/bets")
    transactions_response = client.get("/portfolio/transactions")

    assert bets_response.status_code == 200
    assert transactions_response.status_code == 200
    bet_body = bets_response.json()["items"][0]
    transaction_body = transactions_response.json()["items"][0]
    assert bet_body["game_id"] is None
    assert bet_body["description"] == "16-pick parlay"
    assert bet_body["source_bet_id"] == "DK-1"
    assert transaction_body["source_transaction_id"] == "TXN-1"
    assert transaction_body["balance_after"] == 41.96


@pytest.mark.parametrize(
    "dimension",
    [
        "market_type",
        "funding_type",
        "side",
        "book",
        "model_name",
        "model_type",
        "confidence_tier",
    ],
)
def test_accepts_supported_split_dimensions(
    client: TestClient,
    dimension: str,
) -> None:
    response = client.get(
        "/portfolio/splits",
        params={"dimension": dimension},
    )

    assert response.status_code == 200
    assert response.json()["dimension"] == dimension


def test_rejects_unsupported_split_dimension(client: TestClient) -> None:
    response = client.get(
        "/portfolio/splits",
        params={"dimension": "unknown_column"},
    )

    assert response.status_code == 422


def test_rejects_partial_recommendation_identity(client: TestClient) -> None:
    response = client.post(
        "/portfolio/bets",
        json={
            "game_id": "2026_01_KC_LAC",
            "market_type": "moneyline",
            "side": "away",
            "odds": 175,
            "stake": 25.0,
            "book": "fanduel",
            "recommended_bet_result_id": "result-1",
        },
    )

    assert response.status_code == 422
