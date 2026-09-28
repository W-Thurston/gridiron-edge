# tests/unit/market/test_production_chain_preflight.py
"""Tests for the immutable production-chain preflight contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.market.production_chain_preflight import (
    PRODUCTION_CHAIN_COMPONENT_IDS,
    PRODUCTION_CHAIN_PREFLIGHT_SCHEMA_VERSION,
    MarketFamilyProductionPreflight,
    ProductionChainComponent,
    ProductionChainPreflight,
    ProductionClvKind,
    ProductionMarketFamily,
    ProofComponentState,
    validate_production_chain_preflight,
)

NOW = datetime(2026, 8, 17, 18, 57, tzinfo=UTC)


def _components() -> tuple[ProductionChainComponent, ...]:
    return tuple(
        ProductionChainComponent(component_id, ProofComponentState.UNAVAILABLE, "Absent.")
        for component_id in PRODUCTION_CHAIN_COMPONENT_IDS
    )


def _family(market: ProductionMarketFamily) -> MarketFamilyProductionPreflight:
    return MarketFamilyProductionPreflight(market, _components())


def _preflight() -> ProductionChainPreflight:
    return ProductionChainPreflight(
        PRODUCTION_CHAIN_PREFLIGHT_SCHEMA_VERSION,
        "2026-2027",
        1,
        NOW,
        _family(ProductionMarketFamily.MONEYLINE),
        _family(ProductionMarketFamily.SPREAD),
        _family(ProductionMarketFamily.TOTAL),
    )


def _replace_component(
    family: MarketFamilyProductionPreflight,
    component: ProductionChainComponent,
) -> MarketFamilyProductionPreflight:
    return replace(
        family,
        components=tuple(
            component if value.component_id == component.component_id else value
            for value in family.components
        ),
    )


def _selected_product_frame(
    *,
    role: ForecastRole,
) -> pd.DataFrame:
    generated_at = datetime(
        2026,
        8,
        17,
        18,
        tzinfo=UTC,
    )

    return pd.DataFrame(
        {
            "product_id": ["product"],
            "product_run_id": ["run"],
            "product_generated_at": [generated_at],
            "season": ["2026-2027"],
            "week": [1],
            "game_id": ["2026_01_NE_SEA"],
            "win_status": ["available"],
            "win_selection_status": ["selected"],
            "win_role": [role.value],
            "win_event_id": ["win-event"],
            "win_run_id": ["run"],
            "win_generated_at": [generated_at],
            "spread_status": ["available"],
            "spread_source_event_id": ["win-event"],
            "spread_model_name": ["win_prob"],
            "spread_model_type": ["logistic"],
            "spread_calibration_key": ["win_prob_logistic"],
            "spread_calibration_updated_at": [generated_at],
            "total_status": ["available"],
            "total_selection_status": ["selected"],
            "total_role": [role.value],
            "total_event_id": ["total-event"],
            "total_run_id": ["run"],
            "total_generated_at": [generated_at],
        }
    )


def test_valid_independent_families() -> None:
    validate_production_chain_preflight(_preflight())


def test_wrong_family_slot_is_rejected() -> None:
    value = replace(_preflight(), moneyline=_family(ProductionMarketFamily.SPREAD))
    with pytest.raises(ValueError, match="wrong slot"):
        validate_production_chain_preflight(value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reversed"])
def test_component_inventory_is_exact(mutation: str) -> None:
    family = _family(ProductionMarketFamily.MONEYLINE)
    if mutation == "missing":
        components = family.components[:-1]
    elif mutation == "duplicate":
        components = (*family.components[:-1], family.components[0])
    else:
        components = tuple(reversed(family.components))
    with pytest.raises(ValueError, match="missing, duplicated, or out of order"):
        validate_production_chain_preflight(
            replace(_preflight(), moneyline=replace(family, components=components))
        )


def test_empty_reason_is_rejected() -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent("selected_product", ProofComponentState.AVAILABLE, ""),
    )
    with pytest.raises(ValueError, match="reason"):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


@pytest.mark.parametrize("ids", [("b", "a"), ("a", "a"), ("",)])
def test_evidence_ids_are_sorted_unique_and_nonempty(ids: tuple[str, ...]) -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent(
            "selected_product", ProofComponentState.AVAILABLE, "Present.", ids
        ),
    )
    with pytest.raises(ValueError, match="Evidence identities"):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


@pytest.mark.parametrize(
    "timestamps",
    [
        (NOW, NOW),
        (NOW, NOW - timedelta(minutes=1)),
        (datetime(2026, 8, 17, 18, 57),),
    ],
)
def test_timestamps_are_utc_sorted_and_unique(timestamps: tuple[datetime, ...]) -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent(
            "selected_product", ProofComponentState.AVAILABLE, "Present.", timestamps=timestamps
        ),
    )
    with pytest.raises(ValueError):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


def test_counts_are_consistent() -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent(
            "quote_snapshot",
            ProofComponentState.AVAILABLE,
            "Present.",
            observation_count=1,
            distinct_timestamp_count=2,
        ),
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


def test_available_repeated_history_requires_two_timestamps() -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent(
            "repeated_quote_history",
            ProofComponentState.AVAILABLE,
            "Repeated.",
            timestamps=(NOW,),
            observation_count=2,
            distinct_timestamp_count=1,
        ),
    )
    with pytest.raises(ValueError, match="two exact timestamps"):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


@pytest.mark.parametrize(
    ("market", "kind"),
    [
        (ProductionMarketFamily.MONEYLINE, ProductionClvKind.SPREAD_POINTS),
        (ProductionMarketFamily.SPREAD, ProductionClvKind.MONEYLINE_PRICE),
        (ProductionMarketFamily.TOTAL, ProductionClvKind.SPREAD_POINTS),
    ],
)
def test_market_family_requires_its_clv_kind(
    market: ProductionMarketFamily, kind: ProductionClvKind
) -> None:
    family = _replace_component(
        _family(market),
        ProductionChainComponent("clv", ProofComponentState.AVAILABLE, "Present.", clv_kind=kind),
    )
    value = _preflight()
    value = replace(value, **{market.value: family})
    with pytest.raises(ValueError, match="CLV kind"):
        validate_production_chain_preflight(value)


def test_closeout_requires_provider_sportsbook_and_kickoff_when_available() -> None:
    family = _replace_component(
        _family(ProductionMarketFamily.MONEYLINE),
        ProductionChainComponent(
            "market_closeout",
            ProofComponentState.AVAILABLE,
            "Present.",
            timestamps=(NOW,),
            provider=None,
            sportsbook="book",
            kickoff=NOW,
        ),
    )
    with pytest.raises(ValueError, match="provider, sportsbook, and kickoff"):
        validate_production_chain_preflight(replace(_preflight(), moneyline=family))


def _closeout_result(
    *,
    reference_id: str,
    fetched_at: datetime,
    kickoff: datetime,
):
    from gridiron_edge.market.market_closeout import (
        MarketCloseoutReference,
        MarketCloseoutReferenceKind,
        MarketCloseoutResult,
        MarketCloseoutStatus,
    )

    reference = MarketCloseoutReference(
        reference_id=reference_id,
        reference_kind=MarketCloseoutReferenceKind.CANDIDATE_ISSUANCE,
        provider="the_odds_api",
        provider_event_id=f"event-{reference_id}",
        sportsbook="draftkings",
        game_id=f"game-{reference_id}",
        market="moneyline",
        side="home",
        reference_fetched_at=fetched_at,
        reference_sportsbook_updated_at=fetched_at,
        reference_kickoff=kickoff,
        reference_is_live=False,
        reference_american_price=-110,
        reference_line=None,
    )
    return MarketCloseoutResult(
        reference=reference,
        status=MarketCloseoutStatus.AVAILABLE,
        closeout_fetched_at=fetched_at,
        closeout_sportsbook_updated_at=fetched_at,
        closeout_kickoff=kickoff,
        closeout_is_live=False,
        closeout_american_price=-110,
        closeout_line=None,
    )


def _empty_moneyline_evaluation():
    import pandas as pd

    from gridiron_edge.market.candidate_issuance import (
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        CandidateIssuance,
        candidate_issuance_id,
    )
    from gridiron_edge.market.market_family_evaluation import evaluate_market_families

    issuance_id = candidate_issuance_id(
        product_id="p", product_run_id="r", season="2026-2027", week=1, evaluated_at=NOW
    )
    issuance = CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION, issuance_id, "p", "r", NOW, "2026-2027", 1, NOW, ()
    )
    games = pd.DataFrame(columns=["GAME_ID", "AWAY_SCORE", "HOME_SCORE"])
    return evaluate_market_families(
        issuance=issuance, closeouts=(), games=games, history_boundaries=()
    ).moneyline


def test_postgame_closeout_allows_staggered_kickoffs_across_games() -> None:
    """A late game's legitimate pre-kickoff closeout fetch naturally lands
    after an early game's kickoff. Aggregating every game's closeout
    timestamps together must not be compared against any single game's
    kickoff -- each candidate's own fetch time already precedes its own
    kickoff, which is all that is required."""
    from gridiron_edge.market.production_chain_preflight import (
        _postgame_family_from_evidence,
    )

    early_kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
    late_kickoff = datetime(2026, 9, 14, 0, 20, tzinfo=UTC)
    closeouts = (
        _closeout_result(
            reference_id="a" * 64,
            fetched_at=early_kickoff - timedelta(minutes=5),
            kickoff=early_kickoff,
        ),
        _closeout_result(
            reference_id="b" * 64,
            fetched_at=late_kickoff - timedelta(minutes=5),
            kickoff=late_kickoff,
        ),
    )

    evidence = _postgame_family_from_evidence(
        market=ProductionMarketFamily.MONEYLINE,
        evaluation=_empty_moneyline_evaluation(),
        closeouts=closeouts,
        completed_outcome_count=2,
        scheduled_game_count=2,
    )

    assert evidence.market_closeout.state is ProofComponentState.AVAILABLE
    # The late game's fetch time is after the early game's kickoff -- exactly
    # the aggregate comparison that must not be treated as a violation.
    assert evidence.market_closeout.timestamps[-1] > early_kickoff


def test_postgame_closeout_rejects_observation_at_or_after_its_own_kickoff() -> None:
    from gridiron_edge.market.production_chain_preflight import (
        _postgame_family_from_evidence,
    )

    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
    closeouts = (_closeout_result(reference_id="c" * 64, fetched_at=kickoff, kickoff=kickoff),)

    with pytest.raises(ValueError, match="does not precede its own kickoff"):
        _postgame_family_from_evidence(
            market=ProductionMarketFamily.MONEYLINE,
            evaluation=_empty_moneyline_evaluation(),
            closeouts=closeouts,
            completed_outcome_count=1,
            scheduled_game_count=1,
        )


def test_inputs_are_not_mutated() -> None:
    value = _preflight()
    before = repr(value)
    validate_production_chain_preflight(value)
    assert repr(value) == before


def test_candidate_component_requires_exact_selected_scope(tmp_path) -> None:
    import pandas as pd

    from gridiron_edge.market.production_chain_preflight import (
        _candidate_issuance_component,
        _SelectedProductEvidence,
    )

    selected = _SelectedProductEvidence(
        product_id="selected-product",
        run_id="selected-run",
        generated_at=NOW,
        selected_at=NOW,
        frame=pd.DataFrame({"season": ["2026-2027"], "week": [1]}),
    )
    component = _candidate_issuance_component(
        repo=tmp_path,
        selected=selected,
        season="2026-2027",
        week=1,
    )
    assert component.state is ProofComponentState.UNAVAILABLE
    assert component.evidence_ids == ()


def test_candidate_component_marks_malformed_artifact_invalid(tmp_path) -> None:
    import pandas as pd

    from gridiron_edge.market.production_chain_preflight import (
        _candidate_issuance_component,
        _SelectedProductEvidence,
    )

    directory = tmp_path / "data/output/candidate_issuance/issuances"
    directory.mkdir(parents=True)
    (directory / ("a" * 64 + ".json")).write_text("{}", encoding="utf-8")
    selected = _SelectedProductEvidence(
        product_id="selected-product",
        run_id="selected-run",
        generated_at=NOW,
        selected_at=NOW,
        frame=pd.DataFrame({"season": ["2026-2027"], "week": [1]}),
    )
    component = _candidate_issuance_component(
        repo=tmp_path,
        selected=selected,
        season="2026-2027",
        week=1,
    )
    assert component.state is ProofComponentState.INVALID
    assert component.evidence_ids == ("a" * 64,)


def test_policy_and_results_require_exact_candidate_anchor(tmp_path) -> None:
    from gridiron_edge.market.production_chain_preflight import (
        _recommendation_policy_component,
        _recommendation_result_component,
    )

    candidate = ProductionChainComponent(
        "candidate_issuance",
        ProofComponentState.UNAVAILABLE,
        "Absent.",
    )
    policy = _recommendation_policy_component(repo=tmp_path, candidate=candidate)
    result = _recommendation_result_component(
        repo=tmp_path,
        candidate=candidate,
        season="2026-2027",
        week=1,
    )
    assert policy.state is ProofComponentState.UNAVAILABLE
    assert result.state is ProofComponentState.UNAVAILABLE
    assert policy.evidence_ids == ()
    assert result.evidence_ids == ()


def test_collection_execution_is_not_yet_eligible_before_first_poll(
    tmp_path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from gridiron_edge.market.collection_execution import (
        CollectionDueResult,
        CollectionDueStatus,
    )
    from gridiron_edge.market.production_chain_preflight import (
        _collection_execution_component,
    )

    plan = SimpleNamespace(season="2026-2027", week=1)
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.load_current_collection_plan",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.load_results",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.evaluate_collection_due",
        lambda *_args, **_kwargs: CollectionDueResult(
            CollectionDueStatus.NOT_DUE,
            None,
        ),
    )

    component = _collection_execution_component(
        repo=tmp_path,
        season="2026-2027",
        week=1,
        assessed_at=NOW,
    )
    assert component.state is ProofComponentState.NOT_YET_ELIGIBLE
    assert component.evidence_ids == ()


def test_collection_execution_marks_unresolved_claim_incomplete(
    tmp_path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from gridiron_edge.market.collection_execution import (
        CollectionDueResult,
        CollectionDueStatus,
    )
    from gridiron_edge.market.production_chain_preflight import (
        _collection_execution_component,
    )

    plan = SimpleNamespace(season="2026-2027", week=1)
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.load_current_collection_plan",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.load_results",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.evaluate_collection_due",
        lambda *_args, **_kwargs: CollectionDueResult(
            CollectionDueStatus.CLAIMED,
            None,
        ),
    )

    component = _collection_execution_component(
        repo=tmp_path,
        season="2026-2027",
        week=1,
        assessed_at=NOW,
    )
    assert component.state is ProofComponentState.INCOMPLETE


def test_postgame_assembly_short_circuits_before_kickoff(tmp_path) -> None:
    import pandas as pd

    from gridiron_edge.market.production_chain_preflight import (
        _assemble_postgame_evidence,
        _SelectedProductEvidence,
    )

    selected = _SelectedProductEvidence(
        product_id="product",
        run_id="run",
        generated_at=NOW,
        selected_at=NOW,
        frame=pd.DataFrame({"season": ["2026-2027"], "week": [1]}),
    )
    evidence = _assemble_postgame_evidence(
        repo=tmp_path,
        selected=selected,
        season="2026-2027",
        week=1,
        assessed_at=NOW,
        earliest_kickoff=NOW + timedelta(days=1),
    )
    for family in evidence.values():
        assert family.completed_outcome.state is ProofComponentState.NOT_YET_ELIGIBLE
        assert family.market_closeout.state is ProofComponentState.NOT_YET_ELIGIBLE
        assert family.clv.state is ProofComponentState.NOT_YET_ELIGIBLE
        assert family.realized_performance.state is ProofComponentState.NOT_YET_ELIGIBLE


@pytest.mark.parametrize(
    "market",
    [
        ProductionMarketFamily.MONEYLINE,
        ProductionMarketFamily.SPREAD,
        ProductionMarketFamily.TOTAL,
    ],
)
def test_development_forecast_provenance_is_not_production_ready(
    market: ProductionMarketFamily,
) -> None:
    from gridiron_edge.market.production_chain_preflight import (
        _forecast_component,
        _SelectedProductEvidence,
    )

    selected = _SelectedProductEvidence(
        product_id="product",
        run_id="run",
        generated_at=NOW,
        selected_at=NOW,
        frame=_selected_product_frame(
            role=ForecastRole.DEVELOPMENT,
        ),
    )

    component = _forecast_component(
        market,
        selected,
    )

    assert component.component_id == "forecast_provenance"
    assert component.state is ProofComponentState.INCOMPLETE
    assert component.reason == (f"Selected {market.value} forecast provenance is incomplete.")


@pytest.mark.parametrize(
    "market",
    [
        ProductionMarketFamily.MONEYLINE,
        ProductionMarketFamily.SPREAD,
        ProductionMarketFamily.TOTAL,
    ],
)
def test_live_forecast_provenance_remains_production_ready(
    market: ProductionMarketFamily,
) -> None:
    from gridiron_edge.market.production_chain_preflight import (
        _forecast_component,
        _SelectedProductEvidence,
    )

    selected = _SelectedProductEvidence(
        product_id="product",
        run_id="run",
        generated_at=NOW,
        selected_at=NOW,
        frame=_selected_product_frame(
            role=ForecastRole.LIVE,
        ),
    )

    component = _forecast_component(
        market,
        selected,
    )

    assert component.component_id == "forecast_provenance"
    assert component.state is ProofComponentState.AVAILABLE
