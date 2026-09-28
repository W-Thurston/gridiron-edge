# src/gridiron_edge/market/production_chain_preflight.py
"""Read-only readiness assessment for a full production recommendation chain.

This module reports whether independently required Moneyline, Spread, and Total
proof evidence is available. It never issues candidates, creates policies,
persists recommendation results, records or settles wagers, or selects closeout
observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import json
from pathlib import Path
from typing import Final, TypedDict, cast

import pandas as pd
from pandas import DataFrame

from gridiron_edge.betting.ledger import load_bets
from gridiron_edge.datasets.loaders import load_games
from gridiron_edge.evaluation.live_forecast_closeout import (
    load_live_forecast_closeout,
)
from gridiron_edge.ingest.odds.store import (
    load_current_odds,
    load_odds_ledger,
)
from gridiron_edge.market.candidate_issuance import CandidateIssuance
from gridiron_edge.market.candidate_issuance_store import (
    candidate_issuance_root,
    read_candidate_issuance,
)
from gridiron_edge.market.collection_execution import (
    CollectionDueStatus,
    evaluate_collection_due,
)
from gridiron_edge.market.collection_plan_store import load_current_collection_plan
from gridiron_edge.market.collection_receipt_store import load_results
from gridiron_edge.market.history_boundaries import select_quote_history_boundaries
from gridiron_edge.market.market_closeout import (
    MarketCloseoutResult,
    MarketCloseoutStatus,
    close_candidate_issuance,
)
from gridiron_edge.market.market_family_evaluation import (
    EvaluationEvidenceStatus,
    MarketFamilyEvaluation,
    evaluate_market_families,
)
from gridiron_edge.market.recommendation_policy_store import (
    read_recommendation_policy,
    recommendation_policy_root,
)
from gridiron_edge.market.recommended_bet_result_store import (
    read_recommended_bet_evaluation,
    recommended_bet_result_root,
)

PRODUCTION_CHAIN_PREFLIGHT_SCHEMA_VERSION: Final[int] = 1
PRODUCTION_CHAIN_COMPONENT_IDS: Final[tuple[str, ...]] = (
    "selected_product",
    "forecast_provenance",
    "quote_snapshot",
    "repeated_quote_history",
    "selected_collection_plan",
    "collection_execution",
    "candidate_issuance",
    "recommendation_policy",
    "recommendation_result",
    "backend_serialization",
    "frontend_presentation",
    "recorded_wager",
    "completed_outcome",
    "market_closeout",
    "clv",
    "realized_performance",
)


class ProofComponentState(StrEnum):
    """Availability state of one production-chain proof component."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INCOMPLETE = "incomplete"
    CONFLICTING = "conflicting"
    INVALID = "invalid"
    NOT_YET_ELIGIBLE = "not_yet_eligible"


class ProductionMarketFamily(StrEnum):
    """A separately accepted production recommendation market family."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class ProductionClvKind(StrEnum):
    """Market-specific CLV evidence required by one proof family."""

    MONEYLINE_PRICE = "moneyline_price"
    SPREAD_POINTS = "spread_points"
    TOTAL_POINTS = "total_points"


class _SelectedProductEvidence(TypedDict):
    product_id: str
    run_id: str
    generated_at: datetime
    selected_at: datetime
    frame: DataFrame


class _QuoteEvidence(TypedDict):
    snapshot_rows: int
    snapshot_timestamps: tuple[datetime, ...]
    snapshot_providers: tuple[str, ...]
    history_rows: int
    history_distinct_timestamps: int
    history_timestamps: tuple[datetime, ...]
    repeated_identities: int


class _CollectionPlanEvidence(TypedDict):
    available: bool
    evidence_ids: tuple[str, ...]
    selection_timestamps: tuple[datetime, ...]
    plan_start: datetime
    earliest_kickoff: datetime | None


_EXPECTED_CLV_KIND: Final[dict[ProductionMarketFamily, ProductionClvKind]] = {
    ProductionMarketFamily.MONEYLINE: ProductionClvKind.MONEYLINE_PRICE,
    ProductionMarketFamily.SPREAD: ProductionClvKind.SPREAD_POINTS,
    ProductionMarketFamily.TOTAL: ProductionClvKind.TOTAL_POINTS,
}


@dataclass(frozen=True, slots=True)
class ProductionChainComponent:
    """One immutable readiness statement for one production-chain boundary."""

    component_id: str
    state: ProofComponentState
    reason: str
    evidence_ids: tuple[str, ...] = ()
    timestamps: tuple[datetime, ...] = ()
    observation_count: int | None = None
    distinct_timestamp_count: int | None = None
    provider: str | None = None
    sportsbook: str | None = None
    kickoff: datetime | None = None
    clv_kind: ProductionClvKind | None = None


@dataclass(frozen=True, slots=True)
class _PostgameFamilyEvidence:
    completed_outcome: ProductionChainComponent
    market_closeout: ProductionChainComponent
    clv: ProductionChainComponent
    realized_performance: ProductionChainComponent


@dataclass(frozen=True, slots=True)
class MarketFamilyProductionPreflight:
    """Independent readiness assessment for one market family."""

    market: ProductionMarketFamily
    components: tuple[ProductionChainComponent, ...]

    def component(self, component_id: str) -> ProductionChainComponent:
        """Return one component by its stable identifier."""
        matches = tuple(item for item in self.components if item.component_id == component_id)
        if len(matches) != 1:
            raise ValueError(f"Component {component_id!r} did not resolve exactly once.")
        return matches[0]


@dataclass(frozen=True, slots=True)
class ProductionChainPreflight:
    """Read-only readiness assessment for one exact production week."""

    schema_version: int
    season: str
    week: int
    assessed_at: datetime
    moneyline: MarketFamilyProductionPreflight
    spread: MarketFamilyProductionPreflight
    total: MarketFamilyProductionPreflight

    @property
    def all_families_proven(self) -> bool:
        """Return whether every required component is available in every family."""
        return all(
            all(item.state is ProofComponentState.AVAILABLE for item in family.components)
            for family in (self.moneyline, self.spread, self.total)
        )


def validate_production_chain_preflight(preflight: ProductionChainPreflight) -> None:
    """Validate exact family structure, evidence shape, and local chronology."""
    if preflight.schema_version != PRODUCTION_CHAIN_PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Unsupported production-chain preflight schema version.")
    if not preflight.season.strip() or preflight.week <= 0:
        raise ValueError("Preflight requires a nonempty season and positive week.")
    _require_utc(preflight.assessed_at, "assessed_at")

    families = (
        (ProductionMarketFamily.MONEYLINE, preflight.moneyline),
        (ProductionMarketFamily.SPREAD, preflight.spread),
        (ProductionMarketFamily.TOTAL, preflight.total),
    )
    for expected_market, family in families:
        if family.market is not expected_market:
            raise ValueError(f"{expected_market.value} family occupies the wrong slot.")
        _validate_family(family)


def _validate_family(family: MarketFamilyProductionPreflight) -> None:
    observed = tuple(item.component_id for item in family.components)
    if observed != PRODUCTION_CHAIN_COMPONENT_IDS:
        raise ValueError("Production-chain components are missing, duplicated, or out of order.")
    for component in family.components:
        _validate_component(component)

    repeated = family.component("repeated_quote_history")
    if repeated.state is ProofComponentState.AVAILABLE and (
        repeated.observation_count is None
        or repeated.observation_count < 2
        or repeated.distinct_timestamp_count is None
        or repeated.distinct_timestamp_count < 2
        or len(repeated.timestamps) < 2
    ):
        raise ValueError("Available repeated quote history requires two exact timestamps.")

    clv = family.component("clv")
    if (
        clv.state is ProofComponentState.AVAILABLE
        and clv.clv_kind is not _EXPECTED_CLV_KIND[family.market]
    ):
        raise ValueError(f"{family.market.value} CLV kind does not match its market family.")

    closeout = family.component("market_closeout")
    if closeout.state is ProofComponentState.AVAILABLE and (
        not closeout.provider or not closeout.sportsbook or closeout.kickoff is None
    ):
        raise ValueError("Available closeout requires provider, sportsbook, and kickoff.")
    if clv.state is ProofComponentState.AVAILABLE and (
        clv.provider != closeout.provider or clv.sportsbook != closeout.sportsbook
    ):
        raise ValueError("CLV provider and sportsbook must match closeout evidence.")


def _validate_component(component: ProductionChainComponent) -> None:
    if component.component_id not in PRODUCTION_CHAIN_COMPONENT_IDS:
        raise ValueError(f"Unknown production-chain component: {component.component_id}")
    if not component.reason.strip():
        raise ValueError("Production-chain component reason must be nonempty.")
    if (
        component.evidence_ids != tuple(sorted(component.evidence_ids))
        or len(component.evidence_ids) != len(set(component.evidence_ids))
        or any(not value.strip() for value in component.evidence_ids)
    ):
        raise ValueError("Evidence identities must be nonempty, sorted, and unique.")
    if component.timestamps != tuple(sorted(component.timestamps)) or len(
        component.timestamps
    ) != len(set(component.timestamps)):
        raise ValueError("Component timestamps must be sorted and unique.")
    for value in component.timestamps:
        _require_utc(value, "component timestamp")
    if component.kickoff is not None:
        _require_utc(component.kickoff, "kickoff")
    for label, value in (
        ("observation_count", component.observation_count),
        ("distinct_timestamp_count", component.distinct_timestamp_count),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{label} must be nonnegative.")
    if (
        component.observation_count is not None
        and component.distinct_timestamp_count is not None
        and component.distinct_timestamp_count > component.observation_count
    ):
        raise ValueError("Distinct timestamps cannot exceed observation rows.")


def assess_production_chain_preflight(
    *, repo: Path, season: str, week: int, assessed_at: datetime
) -> ProductionChainPreflight:
    """Assess current repository evidence without creating or mutating artifacts."""
    _require_utc(assessed_at, "assessed_at")
    selected = _selected_product(repo, season, week)
    quotes = _quote_evidence(repo, season, week)
    plan = _collection_plan_evidence(repo, season, week, assessed_at)
    earliest_kickoff = plan["earliest_kickoff"]
    postgame = _assemble_postgame_evidence(
        repo=repo,
        selected=selected,
        season=season,
        week=week,
        assessed_at=assessed_at,
        earliest_kickoff=earliest_kickoff,
    )

    families = {
        market: _assess_family(
            repo=repo,
            market=market,
            selected=selected,
            quotes=quotes[market],
            plan=plan,
            assessed_at=assessed_at,
            earliest_kickoff=earliest_kickoff,
            postgame=postgame[market],
        )
        for market in ProductionMarketFamily
    }
    result = ProductionChainPreflight(
        PRODUCTION_CHAIN_PREFLIGHT_SCHEMA_VERSION,
        season,
        week,
        assessed_at,
        families[ProductionMarketFamily.MONEYLINE],
        families[ProductionMarketFamily.SPREAD],
        families[ProductionMarketFamily.TOTAL],
    )
    validate_production_chain_preflight(result)
    return result


def _assess_family(
    *,
    repo: Path,
    market: ProductionMarketFamily,
    selected: _SelectedProductEvidence,
    quotes: _QuoteEvidence,
    plan: _CollectionPlanEvidence,
    assessed_at: datetime,
    earliest_kickoff: datetime | None,
    postgame: _PostgameFamilyEvidence,
) -> MarketFamilyProductionPreflight:
    selected_ids = tuple(sorted((selected["product_id"], selected["run_id"])))
    selected_times = tuple(sorted((selected["generated_at"], selected["selected_at"])))

    forecast = _forecast_component(market, selected)
    snapshot_count = quotes["snapshot_rows"]
    snapshot_timestamps = quotes["snapshot_timestamps"]
    history_count = quotes["history_rows"]
    history_timestamp_count = quotes["history_distinct_timestamps"]
    history_timestamps = quotes["history_timestamps"]
    repeated_identities = quotes["repeated_identities"]

    components = [
        ProductionChainComponent(
            "selected_product",
            ProofComponentState.AVAILABLE,
            "The explicit current selection resolves one immutable weekly product.",
            selected_ids,
            selected_times,
        ),
        forecast,
        ProductionChainComponent(
            "quote_snapshot",
            ProofComponentState.AVAILABLE if snapshot_count else ProofComponentState.UNAVAILABLE,
            "At least one exact market observation exists."
            if snapshot_count
            else "No exact market observation exists.",
            tuple(sorted(quotes["snapshot_providers"])),
            snapshot_timestamps,
            snapshot_count,
            len(snapshot_timestamps),
        ),
        ProductionChainComponent(
            "repeated_quote_history",
            ProofComponentState.AVAILABLE
            if repeated_identities
            else (
                ProofComponentState.INCOMPLETE
                if snapshot_count
                else ProofComponentState.UNAVAILABLE
            ),
            "At least one exact identity has repeated timestamps."
            if repeated_identities
            else (
                "Canonical quote history exists, but no exact identity has two distinct timestamps."
            ),
            (),
            history_timestamps,
            history_count,
            history_timestamp_count,
        ),
        ProductionChainComponent(
            "selected_collection_plan",
            ProofComponentState.AVAILABLE if plan["available"] else ProofComponentState.UNAVAILABLE,
            "The global current plan selects this exact season and week."
            if plan["available"]
            else "No selected collection plan exists for this scope.",
            tuple(sorted(plan["evidence_ids"])),
            tuple(sorted(plan["selection_timestamps"])),
        ),
        _collection_execution_component(
            repo=repo,
            season=str(selected["frame"]["season"].iloc[0]),
            week=int(selected["frame"]["week"].iloc[0]),
            assessed_at=assessed_at,
        ),
        _candidate_issuance_component(
            repo=repo,
            selected=selected,
            season=str(selected["frame"]["season"].iloc[0]),
            week=int(selected["frame"]["week"].iloc[0]),
        ),
        _recommendation_policy_component(
            repo=repo,
            candidate=_candidate_issuance_component(
                repo=repo,
                selected=selected,
                season=str(selected["frame"]["season"].iloc[0]),
                week=int(selected["frame"]["week"].iloc[0]),
            ),
        ),
        _recommendation_result_component(
            repo=repo,
            candidate=_candidate_issuance_component(
                repo=repo,
                selected=selected,
                season=str(selected["frame"]["season"].iloc[0]),
                week=int(selected["frame"]["week"].iloc[0]),
            ),
            season=str(selected["frame"]["season"].iloc[0]),
            week=int(selected["frame"]["week"].iloc[0]),
        ),
        ProductionChainComponent(
            "backend_serialization",
            ProofComponentState.AVAILABLE,
            "The shipped backend contract serializes persisted recommendation evidence.",
            ("RecommendationPresentation",),
        ),
        ProductionChainComponent(
            "frontend_presentation",
            ProofComponentState.AVAILABLE,
            "The shipped frontend contract presents persisted recommendation evidence.",
            ("RecommendationStatus",),
        ),
        ProductionChainComponent(
            "recorded_wager",
            ProofComponentState.UNAVAILABLE,
            "No matching recorded wager evidence was found; recording is optional.",
        ),
    ]
    components.extend(
        (
            postgame.completed_outcome,
            postgame.market_closeout,
            postgame.clv,
            postgame.realized_performance,
        )
    )
    return MarketFamilyProductionPreflight(market, tuple(components))


def _assemble_postgame_evidence(
    *,
    repo: Path,
    selected: _SelectedProductEvidence,
    season: str,
    week: int,
    assessed_at: datetime,
    earliest_kickoff: datetime | None,
) -> dict[ProductionMarketFamily, _PostgameFamilyEvidence]:
    """Assemble outcomes, closeouts, CLV, and returns once per assessment."""
    if earliest_kickoff is None or assessed_at < earliest_kickoff:
        return {market: _future_postgame_family() for market in ProductionMarketFamily}

    issuance = _exact_candidate_issuance(
        repo=repo,
        selected=selected,
        season=season,
        week=week,
    )
    if issuance is None:
        return {
            market: _unavailable_postgame_family(
                "No exact candidate issuance is available for postgame proof assembly."
            )
            for market in ProductionMarketFamily
        }

    try:
        forecast_closeout = load_live_forecast_closeout(
            repo=repo,
            season=season,
            week=week,
        )
        observations = load_odds_ledger(
            season=season,
            week=week,
            repo=repo,
        )
        market_closeouts = close_candidate_issuance(issuance, observations)
        family_evaluation = evaluate_market_families(
            issuance=issuance,
            closeouts=market_closeouts,
            games=load_games(repo),
            history_boundaries=select_quote_history_boundaries(observations),
            wagers=load_bets(season=season, week=week, repo=repo),
        )
    except (FileNotFoundError, OSError, ValueError):
        return {market: _invalid_postgame_family() for market in ProductionMarketFamily}

    evaluations = {
        ProductionMarketFamily.MONEYLINE: family_evaluation.moneyline,
        ProductionMarketFamily.SPREAD: family_evaluation.spread,
        ProductionMarketFamily.TOTAL: family_evaluation.total,
    }
    closeouts_by_market = {
        market: tuple(
            result for result in market_closeouts if result.reference.market == market.value
        )
        for market in ProductionMarketFamily
    }
    return {
        market: _postgame_family_from_evidence(
            market=market,
            evaluation=evaluations[market],
            closeouts=closeouts_by_market[market],
            completed_outcome_count=forecast_closeout.completed_outcome_count,
            scheduled_game_count=forecast_closeout.scheduled_game_count,
        )
        for market in ProductionMarketFamily
    }


def _exact_candidate_issuance(
    *,
    repo: Path,
    selected: _SelectedProductEvidence,
    season: str,
    week: int,
) -> CandidateIssuance | None:
    directory = candidate_issuance_root(repo) / "issuances"
    matches = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                issuance = read_candidate_issuance(path)
            except (OSError, ValueError):
                continue
            if (
                issuance.season == season
                and issuance.week == week
                and issuance.product_id == selected["product_id"]
                and issuance.product_run_id == selected["run_id"]
            ):
                matches.append(issuance)
    return matches[0] if len(matches) == 1 else None


def _future_postgame_family() -> _PostgameFamilyEvidence:
    return _PostgameFamilyEvidence(
        ProductionChainComponent(
            "completed_outcome",
            ProofComponentState.NOT_YET_ELIGIBLE,
            "The selected week has not reached its earliest kickoff.",
        ),
        ProductionChainComponent(
            "market_closeout",
            ProofComponentState.NOT_YET_ELIGIBLE,
            "Closeout cannot exist before the selected week begins.",
        ),
        ProductionChainComponent(
            "clv",
            ProofComponentState.NOT_YET_ELIGIBLE,
            "CLV cannot exist before an eligible closeout exists.",
        ),
        ProductionChainComponent(
            "realized_performance",
            ProofComponentState.NOT_YET_ELIGIBLE,
            "Realized performance cannot exist before completed outcomes.",
        ),
    )


def _unavailable_postgame_family(reason: str) -> _PostgameFamilyEvidence:
    return _PostgameFamilyEvidence(
        ProductionChainComponent("completed_outcome", ProofComponentState.UNAVAILABLE, reason),
        ProductionChainComponent("market_closeout", ProofComponentState.UNAVAILABLE, reason),
        ProductionChainComponent("clv", ProofComponentState.UNAVAILABLE, reason),
        ProductionChainComponent("realized_performance", ProofComponentState.UNAVAILABLE, reason),
    )


def _invalid_postgame_family() -> _PostgameFamilyEvidence:
    reason = "Postgame evidence could not be strictly assembled from repository artifacts."
    return _PostgameFamilyEvidence(
        ProductionChainComponent("completed_outcome", ProofComponentState.INVALID, reason),
        ProductionChainComponent("market_closeout", ProofComponentState.INVALID, reason),
        ProductionChainComponent("clv", ProofComponentState.INVALID, reason),
        ProductionChainComponent("realized_performance", ProofComponentState.INVALID, reason),
    )


def _postgame_family_from_evidence(
    *,
    market: ProductionMarketFamily,
    evaluation: MarketFamilyEvaluation,
    closeouts: tuple[MarketCloseoutResult, ...],
    completed_outcome_count: int,
    scheduled_game_count: int,
) -> _PostgameFamilyEvidence:
    coverage = evaluation.coverage
    if completed_outcome_count == 0:
        outcome_state = ProofComponentState.UNAVAILABLE
    elif completed_outcome_count < scheduled_game_count:
        outcome_state = ProofComponentState.INCOMPLETE
    else:
        outcome_state = ProofComponentState.AVAILABLE
    outcome = ProductionChainComponent(
        "completed_outcome",
        outcome_state,
        "Completed game outcomes were reconciled to the selected weekly product.",
        observation_count=completed_outcome_count,
        distinct_timestamp_count=scheduled_game_count,
    )

    candidates = tuple(
        result
        for result in closeouts
        if result.reference.reference_id and result.status is MarketCloseoutStatus.AVAILABLE
    )
    for result in candidates:
        if (
            result.closeout_fetched_at is not None
            and result.closeout_kickoff is not None
            and result.closeout_fetched_at >= result.closeout_kickoff
        ):
            raise ValueError(
                "Closeout observation for "
                f"{result.reference.reference_id} does not precede its own kickoff."
            )
    closeout_state = (
        ProofComponentState.AVAILABLE if candidates else ProofComponentState.UNAVAILABLE
    )
    selected_closeout = candidates[0] if candidates else None
    closeout = ProductionChainComponent(
        "market_closeout",
        closeout_state,
        "Validated latest-eligible pregame closeout evidence exists."
        if candidates
        else "No validated latest-eligible pregame closeout evidence exists.",
        tuple(sorted(result.reference.reference_id for result in candidates)),
        tuple(
            sorted(
                {
                    result.closeout_fetched_at
                    for result in candidates
                    if result.closeout_fetched_at is not None
                }
            )
        ),
        observation_count=coverage.closeout_available_count,
        provider=None if selected_closeout is None else selected_closeout.reference.provider,
        sportsbook=None if selected_closeout is None else selected_closeout.reference.sportsbook,
        kickoff=None if selected_closeout is None else selected_closeout.closeout_kickoff,
    )

    expected_kind = {
        ProductionMarketFamily.MONEYLINE: ProductionClvKind.MONEYLINE_PRICE,
        ProductionMarketFamily.SPREAD: ProductionClvKind.SPREAD_POINTS,
        ProductionMarketFamily.TOTAL: ProductionClvKind.TOTAL_POINTS,
    }[market]
    clv_results = tuple(result for result in candidates if result.clv is not None)
    wrong_kind = any(
        result.clv_kind is None or result.clv_kind.value != expected_kind.value
        for result in clv_results
    )
    if wrong_kind:
        clv_state = ProofComponentState.CONFLICTING
    elif clv_results:
        clv_state = ProofComponentState.AVAILABLE
    else:
        clv_state = ProofComponentState.UNAVAILABLE
    selected_clv = clv_results[0] if clv_results else None
    clv = ProductionChainComponent(
        "clv",
        clv_state,
        "Validated market-specific CLV evidence exists."
        if clv_results and not wrong_kind
        else "No validated market-specific CLV evidence exists.",
        tuple(sorted(result.reference.reference_id for result in clv_results)),
        tuple(
            sorted(
                {
                    result.closeout_fetched_at
                    for result in clv_results
                    if result.closeout_fetched_at is not None
                }
            )
        ),
        observation_count=coverage.clv_available_count,
        provider=None if selected_clv is None else selected_clv.reference.provider,
        sportsbook=None if selected_clv is None else selected_clv.reference.sportsbook,
        kickoff=None if selected_clv is None else selected_clv.closeout_kickoff,
        clv_kind=expected_kind if clv_results and not wrong_kind else None,
    )

    returns = evaluation.realized_return
    if returns.status is EvaluationEvidenceStatus.AVAILABLE:
        performance_state = ProofComponentState.AVAILABLE
    elif returns.status is EvaluationEvidenceStatus.CONFLICTING_EVIDENCE:
        performance_state = ProofComponentState.CONFLICTING
    elif returns.status is EvaluationEvidenceStatus.INSUFFICIENT_EVIDENCE:
        performance_state = ProofComponentState.INCOMPLETE
    else:
        performance_state = ProofComponentState.UNAVAILABLE
    performance = ProductionChainComponent(
        "realized_performance",
        performance_state,
        returns.reason or "Settled wager return evidence is available.",
        observation_count=returns.available_count,
    )
    return _PostgameFamilyEvidence(outcome, closeout, clv, performance)


def _selected_product(
    repo: Path,
    season: str,
    week: int,
) -> _SelectedProductEvidence:
    root = repo / "data/output/weekly_products"
    current = _json(root / "current.json")
    index = _json(root / "index.json")
    key = f"{season}_week_{week:02d}"
    selections = cast(dict[str, object], current["selections"])
    selection = selections.get(key)
    if not isinstance(selection, dict):
        raise ValueError("Selected weekly product scope was not found.")
    product_id = str(selection["product_id"])
    products = cast(dict[str, object], index["products"])
    metadata = products.get(product_id)
    if not isinstance(metadata, dict):
        raise ValueError("Selected product is absent from the immutable index.")
    frame = pd.read_parquet(root / str(metadata["artifact"]))
    generated_at = _utc_datetime(metadata["generated_at"])
    selected_at = _utc_datetime(selection["selected_at"])
    if selected_at < generated_at:
        raise ValueError("Weekly product was selected before generation.")
    if len(frame) != int(metadata["row_count"]):
        raise ValueError("Selected product row count disagrees with its index.")
    if (
        not frame["product_id"].eq(product_id).all()
        or not frame["product_run_id"].eq(metadata["run_id"]).all()
    ):
        raise ValueError("Selected product identity disagrees with its rows.")
    return {
        "product_id": product_id,
        "run_id": str(metadata["run_id"]),
        "generated_at": generated_at,
        "selected_at": selected_at,
        "frame": frame,
    }


def _forecast_component(
    market: ProductionMarketFamily,
    selected: _SelectedProductEvidence,
) -> ProductionChainComponent:
    frame = selected["frame"]
    assert isinstance(frame, DataFrame)
    run_id = selected["run_id"]
    if market is ProductionMarketFamily.MONEYLINE:
        valid = (
            frame["win_status"].eq("available").all()
            and frame["win_selection_status"].eq("selected").all()
            and frame["win_role"].eq("live").all()
            and frame["win_event_id"].notna().all()
            and frame["win_event_id"].is_unique
            and frame["win_run_id"].eq(run_id).all()
        )
        ids = tuple(sorted(frame["win_event_id"].astype(str)))
        times = _column_timestamps(frame, "win_generated_at")
    elif market is ProductionMarketFamily.SPREAD:
        valid = (
            frame["spread_status"].eq("available").all()
            and frame["spread_source_event_id"].notna().all()
            and frame["spread_source_event_id"].eq(frame["win_event_id"]).all()
            and frame["win_role"].eq("live").all()
            and frame["spread_model_name"].notna().all()
            and frame["spread_model_type"].notna().all()
            and frame["spread_calibration_key"].notna().all()
            and frame["spread_calibration_updated_at"].notna().all()
        )
        ids = tuple(sorted(frame["spread_source_event_id"].astype(str)))
        times = _column_timestamps(frame, "spread_calibration_updated_at")
    else:
        valid = (
            frame["total_status"].eq("available").all()
            and frame["total_selection_status"].eq("selected").all()
            and frame["total_role"].eq("live").all()
            and frame["total_event_id"].notna().all()
            and frame["total_event_id"].is_unique
            and frame["total_run_id"].eq(run_id).all()
        )
        ids = tuple(sorted(frame["total_event_id"].astype(str)))
        times = _column_timestamps(frame, "total_generated_at")
    return ProductionChainComponent(
        "forecast_provenance",
        ProofComponentState.AVAILABLE if valid else ProofComponentState.INCOMPLETE,
        f"All selected {market.value} forecast provenance is complete."
        if valid
        else f"Selected {market.value} forecast provenance is incomplete.",
        ids,
        times,
    )


def _quote_evidence(
    repo: Path,
    season: str,
    week: int,
) -> dict[ProductionMarketFamily, _QuoteEvidence]:
    """Assess latest-snapshot coverage separately from append-preserved history."""
    current = load_current_odds(repo=repo)
    snapshot = DataFrame() if current is None else current
    history = load_odds_ledger(
        season=season,
        week=week,
        repo=repo,
    )

    return {
        market: _market_quote_evidence(
            market=market,
            snapshot=snapshot,
            history=history,
            season=season,
            week=week,
        )
        for market in ProductionMarketFamily
    }


def _market_quote_evidence(
    *,
    market: ProductionMarketFamily,
    snapshot: DataFrame,
    history: DataFrame,
    season: str,
    week: int,
) -> _QuoteEvidence:
    snapshot_rows = _scoped_market_rows(
        snapshot,
        market=market,
        season=season,
        week=week,
    )
    history_rows = _scoped_market_rows(
        history,
        market=market,
        season=season,
        week=week,
    )

    snapshot_timestamps = _frame_timestamps(snapshot_rows)
    history_timestamps = _frame_timestamps(history_rows)
    identity = [
        column
        for column in (
            "provider",
            "sportsbook",
            "game_id",
            "market",
            "side",
        )
        if column in history_rows
    ]
    repeated_identities = 0
    if not history_rows.empty and identity:
        timestamp_column = _timestamp_column(history_rows)
        depths = history_rows.groupby(
            identity,
            dropna=False,
        )[timestamp_column].nunique()
        repeated_identities = (depths >= 2).sum()

    providers = (
        tuple(sorted(snapshot_rows["provider"].dropna().astype(str).unique()))
        if "provider" in snapshot_rows
        else ()
    )
    return _QuoteEvidence(
        snapshot_rows=len(snapshot_rows),
        snapshot_timestamps=snapshot_timestamps,
        snapshot_providers=providers,
        history_rows=len(history_rows),
        history_distinct_timestamps=len(history_timestamps),
        history_timestamps=history_timestamps,
        repeated_identities=repeated_identities,
    )


def _scoped_market_rows(
    frame: DataFrame,
    *,
    market: ProductionMarketFamily,
    season: str,
    week: int,
) -> DataFrame:
    if frame.empty:
        return frame.copy()
    rows = frame
    if "season" in rows:
        rows = rows.loc[rows["season"].astype(str) == season]
    if "week" in rows:
        rows = rows.loc[pd.to_numeric(rows["week"], errors="coerce") == week]
    return rows.loc[rows["market"].astype(str) == market.value].copy()


def _timestamp_column(frame: DataFrame) -> str:
    for column in ("fetched_at", "market_fetched_at"):
        if column in frame:
            return column
    raise ValueError("Quote evidence has no observation timestamp column.")


def _frame_timestamps(frame: DataFrame) -> tuple[datetime, ...]:
    if frame.empty:
        return ()
    column = _timestamp_column(frame)
    timestamps = pd.to_datetime(frame[column], utc=True)
    return tuple(sorted(value.to_pydatetime() for value in timestamps.drop_duplicates()))


def _collection_plan_evidence(
    repo: Path,
    season: str,
    week: int,
    assessed_at: datetime,
) -> _CollectionPlanEvidence:
    root = repo / "data/odds/collection_plans"
    current_path = root / "current.json"
    plan_path = root / f"season={season}" / f"week={week:02d}.json"
    if not current_path.exists() or not plan_path.exists():
        return {
            "available": False,
            "evidence_ids": (),
            "selection_timestamps": (),
            "plan_start": assessed_at,
            "earliest_kickoff": None,
        }
    current, plan = _json(current_path), _json(plan_path)
    current_week = current.get("week")
    if not isinstance(current_week, int):
        raise ValueError("Current collection-plan week must be an integer.")
    available = str(current.get("season")) == season and current_week == week
    selected_at = _utc_datetime(current["selected_at"])
    created_at = _utc_datetime(plan["created_at"])
    plan_start = _utc_datetime(plan["plan_start"])
    kickoff_groups = cast(list[dict[str, object]], plan["kickoff_groups"])
    kickoffs = tuple(sorted(_utc_datetime(group["commence_time"]) for group in kickoff_groups))
    return {
        "available": available,
        "evidence_ids": (f"{season}:week:{week:02d}",),
        "selection_timestamps": tuple(sorted((created_at, selected_at, plan_start))),
        "plan_start": plan_start,
        "earliest_kickoff": kickoffs[0] if kickoffs else None,
    }


def _collection_execution_component(  # noqa: PLR0911
    *,
    repo: Path,
    season: str,
    week: int,
    assessed_at: datetime,
) -> ProductionChainComponent:
    """Assess selected-plan execution through owned receipt artifacts."""
    try:
        plan = load_current_collection_plan(repo=repo)
    except (FileNotFoundError, OSError, ValueError):
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.INVALID,
            ("The selected collection plan cannot be strictly loaded for execution assessment."),
        )

    if (plan.season, plan.week) != (season, week):
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.CONFLICTING,
            "The selected collection-plan scope does not match the weekly product.",
        )

    try:
        results = load_results(season=season, week=week, repo=repo)
        due = evaluate_collection_due(
            plan,
            evaluated_at=assessed_at,
            grace_period=timedelta(minutes=15),
            repo=repo,
        )
    except (OSError, ValueError):
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.INVALID,
            "Collection execution receipts cannot be strictly validated.",
        )

    result_ids = tuple(
        sorted(f"{result.scheduled_at.isoformat()}:{result.status.value}" for result in results)
    )
    timestamps = tuple(
        sorted(
            {
                timestamp
                for result in results
                for timestamp in (result.started_at, result.completed_at)
            }
        )
    )
    completed_count = sum(result.status.value == "completed" for result in results)

    if due.status is CollectionDueStatus.NOT_DUE and not results:
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.NOT_YET_ELIGIBLE,
            "The selected plan has not reached its first scheduled poll.",
        )
    if due.status is CollectionDueStatus.CLAIMED:
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.INCOMPLETE,
            "The earliest due collection poll has a claim but no terminal result.",
            result_ids,
            timestamps,
            observation_count=len(results),
        )
    if due.status in {CollectionDueStatus.DUE, CollectionDueStatus.MISSED}:
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.INCOMPLETE,
            "The earliest unresolved collection poll requires terminal execution evidence.",
            result_ids,
            timestamps,
            observation_count=len(results),
        )
    if due.status is CollectionDueStatus.PLAN_UNAVAILABLE:
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.UNAVAILABLE,
            "The selected plan does not contain executable schedule evidence.",
            result_ids,
            timestamps,
            observation_count=len(results),
        )
    if results:
        return ProductionChainComponent(
            "collection_execution",
            ProofComponentState.AVAILABLE,
            "Validated terminal receipts exist for selected-plan collection polls.",
            result_ids,
            timestamps,
            observation_count=len(results),
            distinct_timestamp_count=completed_count,
        )
    return ProductionChainComponent(
        "collection_execution",
        ProofComponentState.UNAVAILABLE,
        "No validated selected-plan execution receipts exist.",
    )


def _candidate_issuance_component(
    *,
    repo: Path,
    selected: _SelectedProductEvidence,
    season: str,
    week: int,
) -> ProductionChainComponent:
    """Resolve exact valid issuance evidence for the selected weekly product."""
    directory = candidate_issuance_root(repo) / "issuances"
    if not directory.is_dir():
        return ProductionChainComponent(
            "candidate_issuance",
            ProofComponentState.UNAVAILABLE,
            "No immutable candidate issuance exists for the selected scope.",
        )

    matches = []
    invalid_ids: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            issuance = read_candidate_issuance(path)
        except (OSError, ValueError):
            invalid_ids.append(path.stem)
            continue
        if (
            issuance.season == season
            and issuance.week == week
            and issuance.product_id == selected["product_id"]
            and issuance.product_run_id == selected["run_id"]
        ):
            matches.append(issuance)

    if len(matches) == 1:
        issuance = matches[0]
        return ProductionChainComponent(
            "candidate_issuance",
            ProofComponentState.AVAILABLE,
            "One immutable candidate issuance exactly matches the selected product scope.",
            (issuance.issuance_id,),
            (issuance.evaluated_at,),
            observation_count=len(issuance.rows),
        )
    if len(matches) > 1:
        return ProductionChainComponent(
            "candidate_issuance",
            ProofComponentState.CONFLICTING,
            "Multiple immutable candidate issuances match the selected product scope.",
            tuple(sorted(issuance.issuance_id for issuance in matches)),
            tuple(sorted({issuance.evaluated_at for issuance in matches})),
            observation_count=sum(len(issuance.rows) for issuance in matches),
        )
    if invalid_ids:
        return ProductionChainComponent(
            "candidate_issuance",
            ProofComponentState.INVALID,
            (
                "Candidate issuance artifacts exist but none can "
                "be strictly validated for the selected scope."
            ),
            tuple(sorted(invalid_ids)),
        )
    return ProductionChainComponent(
        "candidate_issuance",
        ProofComponentState.UNAVAILABLE,
        "No immutable candidate issuance matches the selected product scope.",
    )


def _recommendation_policy_component(
    *,
    repo: Path,
    candidate: ProductionChainComponent,
) -> ProductionChainComponent:
    """Resolve exact valid policy evidence referenced by recommendation results."""
    if candidate.state is not ProofComponentState.AVAILABLE:
        return ProductionChainComponent(
            "recommendation_policy",
            ProofComponentState.UNAVAILABLE,
            "No exact candidate issuance is available to anchor policy evidence.",
        )

    root = recommendation_policy_root(repo)
    valid = []
    invalid_ids: list[str] = []
    for path in sorted(root.glob("schema=*/*.json")) if root.is_dir() else ():
        try:
            valid.append(read_recommendation_policy(path))
        except (OSError, ValueError):
            invalid_ids.append(path.stem)

    result_root = recommended_bet_result_root(repo)
    referenced_policy_ids: set[str] = set()
    if result_root.is_dir():
        for path in sorted(result_root.glob("schema=*/evaluations/*.json")):
            try:
                evaluation = read_recommended_bet_evaluation(path)
            except (OSError, ValueError):
                continue
            if evaluation.issuance_id in candidate.evidence_ids:
                referenced_policy_ids.add(evaluation.policy_id)

    matches = [policy for policy in valid if policy.policy_id in referenced_policy_ids]
    if len(matches) == 1:
        policy = matches[0]
        return ProductionChainComponent(
            "recommendation_policy",
            ProofComponentState.AVAILABLE,
            "One immutable recommendation policy is referenced by the exact issuance evaluation.",
            (policy.policy_id,),
            (policy.created_at,),
        )
    if len(matches) > 1:
        return ProductionChainComponent(
            "recommendation_policy",
            ProofComponentState.CONFLICTING,
            "Multiple recommendation policies are referenced by evaluations of the exact issuance.",
            tuple(sorted(policy.policy_id for policy in matches)),
            tuple(sorted({policy.created_at for policy in matches})),
        )
    if invalid_ids:
        return ProductionChainComponent(
            "recommendation_policy",
            ProofComponentState.INVALID,
            "Recommendation-policy artifacts exist but cannot be strictly validated.",
            tuple(sorted(invalid_ids)),
        )
    return ProductionChainComponent(
        "recommendation_policy",
        ProofComponentState.UNAVAILABLE,
        "No exact evaluated recommendation policy exists for the candidate issuance.",
    )


def _recommendation_result_component(
    *,
    repo: Path,
    candidate: ProductionChainComponent,
    season: str,
    week: int,
) -> ProductionChainComponent:
    """Resolve exact valid evaluation evidence for one selected-scope issuance."""
    if candidate.state is not ProofComponentState.AVAILABLE:
        return ProductionChainComponent(
            "recommendation_result",
            ProofComponentState.UNAVAILABLE,
            "No exact candidate issuance is available to anchor recommendation results.",
        )

    root = recommended_bet_result_root(repo)
    matches = []
    invalid_ids: list[str] = []
    if root.is_dir():
        for path in sorted(root.glob("schema=*/evaluations/*.json")):
            try:
                evaluation = read_recommended_bet_evaluation(path)
            except (OSError, ValueError):
                invalid_ids.append(path.stem)
                continue
            scoped_results = tuple(
                result
                for result in evaluation.results
                if result.season == season and result.week == week
            )
            if evaluation.issuance_id in candidate.evidence_ids and len(scoped_results) == len(
                evaluation.results
            ):
                matches.append(evaluation)

    if len(matches) == 1:
        evaluation = matches[0]
        return ProductionChainComponent(
            "recommendation_result",
            ProofComponentState.AVAILABLE,
            (
                "One immutable recommendation evaluation exactly "
                "matches the candidate issuance and week."
            ),
            tuple(sorted((evaluation.evaluation_id, evaluation.policy_id))),
            (evaluation.evaluated_at,),
            observation_count=len(evaluation.results),
        )
    if len(matches) > 1:
        return ProductionChainComponent(
            "recommendation_result",
            ProofComponentState.CONFLICTING,
            "Multiple recommendation evaluations match the exact candidate issuance and week.",
            tuple(sorted(evaluation.evaluation_id for evaluation in matches)),
            tuple(sorted({evaluation.evaluated_at for evaluation in matches})),
            observation_count=sum(len(evaluation.results) for evaluation in matches),
        )
    if invalid_ids:
        return ProductionChainComponent(
            "recommendation_result",
            ProofComponentState.INVALID,
            "Recommended-bet evaluation artifacts exist but cannot be strictly validated.",
            tuple(sorted(invalid_ids)),
        )
    return ProductionChainComponent(
        "recommendation_result",
        ProofComponentState.UNAVAILABLE,
        "No recommendation evaluation matches the exact candidate issuance and week.",
    )


def _directory_component(
    component_id: str, path: Path, missing_reason: str
) -> ProductionChainComponent:
    files = (
        tuple(sorted(str(item.relative_to(path)) for item in path.rglob("*.json")))
        if path.is_dir()
        else ()
    )
    return ProductionChainComponent(
        component_id,
        ProofComponentState.AVAILABLE if files else ProofComponentState.UNAVAILABLE,
        "Persisted evidence exists and requires exact-scope proof assembly."
        if files
        else missing_reason,
        files,
    )


def _column_timestamps(frame: DataFrame, column: str) -> tuple[datetime, ...]:
    values = pd.to_datetime(frame[column], utc=True)
    return tuple(sorted(value.to_pydatetime() for value in values.drop_duplicates()))


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _utc_datetime(value: object) -> datetime:
    if not isinstance(value, str | datetime):
        raise ValueError("Artifact timestamp must be an ISO string or datetime.")
    result = pd.Timestamp(value).to_pydatetime()
    return _require_utc(result, "artifact timestamp")


def _require_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value
