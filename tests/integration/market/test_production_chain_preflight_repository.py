# tests/integration/market/test_production_chain_preflight_repository.py
"""Real-repository readiness assessment for Market Unit 26."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gridiron_edge.market.production_chain_preflight import (
    ProductionMarketFamily,
    ProofComponentState,
    assess_production_chain_preflight,
)


def test_historical_week_one_evidence_is_classified_truthfully() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    result = assess_production_chain_preflight(
        repo=repo_root,
        season="2026-2027",
        week=1,
        assessed_at=datetime(2026, 8, 17, 18, 57, tzinfo=UTC),
    )

    expected_markets = (
        ProductionMarketFamily.MONEYLINE,
        ProductionMarketFamily.SPREAD,
        ProductionMarketFamily.TOTAL,
    )
    families = (result.moneyline, result.spread, result.total)
    assert tuple(family.market for family in families) == expected_markets

    for family in families:
        assert family.component("selected_product").state is ProofComponentState.AVAILABLE
        assert family.component("forecast_provenance").state is ProofComponentState.AVAILABLE
        quote_snapshot = family.component("quote_snapshot")
        assert quote_snapshot.state is ProofComponentState.AVAILABLE
        assert quote_snapshot.observation_count == 18
        assert quote_snapshot.distinct_timestamp_count == 1
        assert len(quote_snapshot.timestamps) == 1
        history = family.component("repeated_quote_history")
        assert history.state is ProofComponentState.AVAILABLE
        assert history.distinct_timestamp_count == 34
        assert family.component("selected_collection_plan").state is ProofComponentState.AVAILABLE
        collection_execution = family.component("collection_execution")
        assert collection_execution.state is ProofComponentState.AVAILABLE
        assert collection_execution.observation_count == 34
        assert collection_execution.distinct_timestamp_count == 34
        candidate = family.component("candidate_issuance")
        assert candidate.state is ProofComponentState.AVAILABLE
        assert candidate.reason == (
            "One immutable candidate issuance exactly matches the selected product scope."
        )
        assert candidate.evidence_ids == (
            "e945987f2903435ac8c798ea5085bd5a39d3ffe2a7741cf5123cabf221e427c0",
        )
        assert candidate.observation_count == 1680
        assert len(candidate.timestamps) == 1

        policy = family.component("recommendation_policy")
        assert policy.state is ProofComponentState.AVAILABLE
        assert policy.reason == (
            "One immutable recommendation policy is referenced by the exact issuance evaluation."
        )
        assert policy.evidence_ids == (
            "33255cc82cced0c438fd067ff30261cc4344cfbec2c72f7914124b8a9d3ccb6f",
        )

        recommendation = family.component("recommendation_result")
        assert recommendation.state is ProofComponentState.AVAILABLE
        assert recommendation.reason == (
            "One immutable recommendation evaluation exactly "
            "matches the candidate issuance and week."
        )
        assert recommendation.evidence_ids == (
            "100a70a86b73d19c55d5e810355529c553cf690a324ab108bf62ee06ab13b709",
            "33255cc82cced0c438fd067ff30261cc4344cfbec2c72f7914124b8a9d3ccb6f",
        )
        assert recommendation.observation_count == 698
        assert family.component("backend_serialization").state is ProofComponentState.AVAILABLE
        assert family.component("frontend_presentation").state is ProofComponentState.AVAILABLE
        assert family.component("completed_outcome").state is ProofComponentState.NOT_YET_ELIGIBLE
        assert family.component("market_closeout").state is ProofComponentState.NOT_YET_ELIGIBLE
        assert family.component("clv").state is ProofComponentState.NOT_YET_ELIGIBLE
        assert (
            family.component("realized_performance").state is ProofComponentState.NOT_YET_ELIGIBLE
        )

    assert not result.all_families_proven
    recorded_wager = family.component("recorded_wager")
    assert recorded_wager.state is ProofComponentState.UNAVAILABLE
    assert recorded_wager.reason == (
        "No matching recorded wager evidence was found; recording is optional."
    )
