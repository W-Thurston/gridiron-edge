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
        assert quote_snapshot.state is ProofComponentState.UNAVAILABLE
        assert quote_snapshot.observation_count == 0
        assert quote_snapshot.distinct_timestamp_count == 0
        assert quote_snapshot.timestamps == ()
        history = family.component("repeated_quote_history")
        assert history.state is ProofComponentState.AVAILABLE
        assert history.distinct_timestamp_count == 2
        assert family.component("selected_collection_plan").state is ProofComponentState.AVAILABLE
        assert (
            family.component("collection_execution").state is ProofComponentState.NOT_YET_ELIGIBLE
        )
        candidate = family.component("candidate_issuance")
        assert candidate.state is ProofComponentState.CONFLICTING
        assert candidate.reason == (
            "Multiple immutable candidate issuances match the selected product scope."
        )
        assert len(candidate.evidence_ids) == 3
        assert candidate.observation_count == 5040
        assert len(candidate.timestamps) == 3

        policy = family.component("recommendation_policy")
        assert policy.state is ProofComponentState.UNAVAILABLE
        assert policy.reason == (
            "No exact candidate issuance is available to anchor policy evidence."
        )
        assert policy.evidence_ids == ()

        recommendation = family.component("recommendation_result")
        assert recommendation.state is ProofComponentState.UNAVAILABLE
        assert recommendation.reason == (
            "No exact candidate issuance is available to anchor recommendation results."
        )
        assert recommendation.evidence_ids == ()
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
