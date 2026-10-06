# src/gridiron_edge/evaluation/weekly_recap_markets.py

"""The weekly recap's market sections: what was published as edges, and how recorded wagers did.

Nothing here computes an edge. The sections read persisted evidence only (``DECISIONS.md``
D89):

* **Edge evidence.** A week whose selected product has a candidate issuance reads
  ``production_chain``: the issuance, its latest unambiguous recommendation results, and each
  candidate's market closeout and graded outcome. Candidates were never recommendations, so they
  get CLV and win/loss/push counts only, never dollars. Without an issuance the week reads the
  latest weekly-cycle edge receipt decided at or before its last kickoff:
  ``withheld_unvalidated`` when that receipt records the ``model_edges_unvalidated`` blocker,
  otherwise ``not_recorded`` (a published edge report is a mutable file, so it is not evidence).
  Every edge receipt in that window is listed.
* **Recorded wagers.** The ledger rows whose ``game_id`` is in the week: record, stake, P&L and
  ROI through :mod:`gridiron_edge.betting.performance`, overall and per market type. An ``open``
  bet on a finished game is ``unsettled``. CLV comes from :func:`close_recorded_wagers` where a
  bet's quote reference matches the ledger; every other bet counts as ``clv_unavailable``.

Pure: no I/O. The recap builder supplies the issuances, results, receipts, bets and quotes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import math
from statistics import median
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.betting.performance import record, roi
from gridiron_edge.evaluation.game_model_evaluation_report import frame_content_digest
from gridiron_edge.market.candidate_issuance import (
    CandidateIssuance,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_row_id,
)
from gridiron_edge.market.candidate_outcome import CandidateOutcome, grade_candidate_outcome
from gridiron_edge.market.edge_diagnostics import EdgeDiagnosticBlocker
from gridiron_edge.market.market_closeout import (
    MarketCloseoutResult,
    MarketCloseoutStatus,
    close_candidate_rows,
    close_recorded_wagers,
)
from gridiron_edge.market.recommended_bet_result_store import LoadedRecommendedBetResult
from gridiron_edge.weekly_cycle.receipts import ATTEMPT_EDGES, OUTCOME_BLOCKED, CycleReceipt

CANDIDATE_MARKETS: Final[tuple[str, ...]] = ("moneyline", "spread", "total")
# Bet markets the quote ledger can close; parlays and specials have no single quote.
_CLOSABLE_BET_MARKETS: Final[frozenset[str]] = frozenset(CANDIDATE_MARKETS)
# The edge stage records its blockers as "edge calculation blocked: a, b (note)".
_BLOCKED_PREFIX: Final[str] = "edge calculation blocked: "
# Ledger columns that identify a bet internally; a re-import regenerates them.
_INTERNAL_BET_COLUMNS: Final[tuple[str, ...]] = ("bet_id",)


class EdgeEvidenceStatus(StrEnum):
    """What a week's persisted state says about the edges it published."""

    PRODUCTION_CHAIN = "production_chain"
    WITHHELD_UNVALIDATED = "withheld_unvalidated"
    NOT_RECORDED = "not_recorded"


class CandidateRecommendationState(StrEnum):
    """A candidate row's latest recommendation result state, or that it has none."""

    QUALIFIED = "qualified"
    RECOMMENDED = "recommended"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    CONFLICTING = "conflicting"
    # No persisted evaluation holds a result for this candidate row.
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class MarketInputs:
    """The persisted evidence the market sections read for one week.

    ``issuances`` are the stored candidate issuances of the week's selected product and run;
    ``recommendations`` the week's latest unambiguous recommendation results; ``receipts`` the
    week's weekly-cycle receipts; ``bets`` the ledger rows whose ``game_id`` is in the week;
    ``quotes`` the week's quote-ledger partition.
    """

    issuances: tuple[CandidateIssuance, ...]
    recommendations: tuple[LoadedRecommendedBetResult, ...]
    receipts: tuple[CycleReceipt, ...]
    bets: DataFrame
    quotes: DataFrame


@dataclass(frozen=True, slots=True)
class StatusCount:
    """How many rows read one status."""

    status: str
    count: int


@dataclass(frozen=True, slots=True)
class EdgeReceiptEntry:
    """One weekly-cycle edge receipt decided at or before the week's last kickoff."""

    decided_at: datetime
    outcome: str
    detail: str
    product_id: str | None
    market_fetched_at: str | None


@dataclass(frozen=True, slots=True)
class CandidateFamily:
    """One market family's candidates: their closeout, CLV and graded outcomes.

    Rows are candidate rows (the same side at several books and snapshots counts once per row);
    ``sides`` counts the distinct game and side they cover. CLV is in the family's own unit
    (``clv_kind``) over rows whose closeout is ``available``.
    """

    market: str
    candidates: int
    sides: int
    closeout: tuple[StatusCount, ...]
    clv_kind: str | None
    clv_count: int
    median_clv: float | None
    share_positive_clv: float | None
    wins: int
    losses: int
    pushes: int
    outcome_unavailable: int


@dataclass(frozen=True, slots=True)
class ProductionChain:
    """The week's candidate issuance and what the recommendation policy made of it."""

    issuance_id: str
    issued_at: datetime
    rows: int
    candidates: int
    not_candidates: int
    unavailable: int
    evaluation_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    recommendations: tuple[StatusCount, ...]
    families: tuple[CandidateFamily, ...]


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    """What the week published as edges, with the receipts that say so."""

    status: EdgeEvidenceStatus
    chain: ProductionChain | None
    receipts_through: datetime
    receipts: tuple[EdgeReceiptEntry, ...]


@dataclass(frozen=True, slots=True)
class WagerSummary:
    """Recorded wagers' record and money; ROI is P&L over settled stake, in percent."""

    bets: int
    settled: int
    unsettled: int
    wins: int
    losses: int
    pushes: int
    staked: float
    settled_staked: float
    pnl: float
    roi_pct: float | None
    bonus_funded: int
    clv_available: int
    clv_unavailable: int
    linked_recommendations: int


@dataclass(frozen=True, slots=True)
class WagerMarket:
    """One bet market type's summary, and its CLV in that market's own unit."""

    market_type: str
    summary: WagerSummary
    clv_kind: str | None
    median_clv: float | None
    share_positive_clv: float | None


@dataclass(frozen=True, slots=True)
class RecordedWagers:
    """The week's recorded wagers overall and per market type."""

    total: WagerSummary
    markets: tuple[WagerMarket, ...]


# ---------------------------------------------------------------------------
# Edge evidence
# ---------------------------------------------------------------------------


def receipt_blockers(detail: str) -> tuple[str, ...]:
    """Return the blockers an edge receipt's detail names, or none when it names no block."""
    if not detail.startswith(_BLOCKED_PREFIX):
        return ()
    listed = detail.removeprefix(_BLOCKED_PREFIX).split(" (", 1)[0]
    return tuple(part.strip() for part in listed.split(",") if part.strip())


def _edge_inputs(receipt: CycleReceipt) -> Mapping[str, object]:
    inputs = receipt.inputs.get("edge_inputs")
    return inputs if isinstance(inputs, Mapping) else {}


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def edge_receipt_window(
    receipts: Iterable[CycleReceipt], *, through: datetime
) -> tuple[CycleReceipt, ...]:
    """Return the edge receipts decided at or before ``through``, oldest first."""
    window = [
        receipt
        for receipt in receipts
        if receipt.attempted == ATTEMPT_EDGES and receipt.decided <= through
    ]
    return tuple(sorted(window, key=lambda receipt: receipt.decided))


def edge_evidence(
    inputs: MarketInputs,
    *,
    last_kickoff: datetime,
    scores: Mapping[str, tuple[float, float]],
) -> EdgeEvidence:
    """Read the week's edge evidence; at most one issuance may be given."""
    if len(inputs.issuances) > 1:
        raise ValueError("At most one candidate issuance can stand for a week's product.")
    window = edge_receipt_window(inputs.receipts, through=last_kickoff)
    entries = tuple(
        EdgeReceiptEntry(
            decided_at=receipt.decided,
            outcome=receipt.outcome,
            detail=receipt.detail,
            product_id=_text(_edge_inputs(receipt).get("product_id")),
            market_fetched_at=_text(_edge_inputs(receipt).get("market_fetched_at")),
        )
        for receipt in window
    )
    chain = None
    if inputs.issuances:
        chain = production_chain(
            inputs.issuances[0], inputs.recommendations, inputs.quotes, scores=scores
        )
        status = EdgeEvidenceStatus.PRODUCTION_CHAIN
    elif (
        window
        and window[-1].outcome == OUTCOME_BLOCKED
        and EdgeDiagnosticBlocker.MODEL_EDGES_UNVALIDATED.value
        in receipt_blockers(window[-1].detail)
    ):
        status = EdgeEvidenceStatus.WITHHELD_UNVALIDATED
    else:
        status = EdgeEvidenceStatus.NOT_RECORDED
    return EdgeEvidence(status=status, chain=chain, receipts_through=last_kickoff, receipts=entries)


def _status_counts(values: Iterable[str]) -> tuple[StatusCount, ...]:
    return tuple(StatusCount(status, count) for status, count in sorted(Counter(values).items()))


def _clv_values(results: Iterable[MarketCloseoutResult]) -> tuple[list[float], str | None]:
    available = [
        result
        for result in results
        if result.status is MarketCloseoutStatus.AVAILABLE and result.clv is not None
    ]
    kinds = {result.clv_kind.value for result in available if result.clv_kind is not None}
    if len(kinds) > 1:
        raise ValueError("One market's closeouts carry more than one CLV unit.")
    return [float(result.clv) for result in available if result.clv is not None], (
        kinds.pop() if kinds else None
    )


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _share_positive(values: Sequence[float]) -> float | None:
    return sum(value > 0 for value in values) / len(values) if values else None


def production_chain(
    issuance: CandidateIssuance,
    recommendations: Sequence[LoadedRecommendedBetResult],
    quotes: DataFrame,
    *,
    scores: Mapping[str, tuple[float, float]],
) -> ProductionChain:
    """Summarize one issuance: its rows, recommendation states, and each family's closeout."""
    candidates = tuple(
        row for row in issuance.rows if row.state is CandidateIssuanceState.CANDIDATE
    )
    results = [item for item in recommendations if item.result.issuance_id == issuance.issuance_id]
    closeouts = {
        result.reference.reference_id: result
        for result in close_candidate_rows(issuance, candidates, quotes)
    }
    return ProductionChain(
        issuance_id=issuance.issuance_id,
        issued_at=issuance.evaluated_at,
        rows=len(issuance.rows),
        candidates=len(candidates),
        not_candidates=sum(
            row.state is CandidateIssuanceState.NOT_CANDIDATE for row in issuance.rows
        ),
        unavailable=sum(row.state is CandidateIssuanceState.UNAVAILABLE for row in issuance.rows),
        evaluation_ids=tuple(sorted({item.evaluation_id for item in results})),
        policy_ids=tuple(sorted({item.result.policy_id for item in results})),
        recommendations=_recommendation_states(issuance, candidates, results),
        families=tuple(
            _candidate_family(
                market,
                [row for row in candidates if row.market == market],
                issuance_id=issuance.issuance_id,
                closeouts=closeouts,
                scores=scores,
            )
            for market in CANDIDATE_MARKETS
        ),
    )


def _recommendation_states(
    issuance: CandidateIssuance,
    candidates: Sequence[CandidateIssuanceRow],
    results: Sequence[LoadedRecommendedBetResult],
) -> tuple[StatusCount, ...]:
    """Count each candidate row's recommendation state; a row with no result is not evaluated."""
    by_reference = {
        (
            item.result.candidate_reference_derivation_version,
            item.result.candidate_reference_id,
        ): CandidateRecommendationState(item.result.result_state.value)
        for item in results
    }
    versions = sorted({version for version, _ in by_reference})
    states: list[CandidateRecommendationState] = []
    for row in candidates:
        keys = [
            (version, candidate_issuance_row_id(issuance.issuance_id, row, version=version))
            for version in versions
        ]
        states.append(
            next(
                (by_reference[key] for key in keys if key in by_reference),
                CandidateRecommendationState.NOT_EVALUATED,
            )
        )
    return _status_counts(state.value for state in states)


def _candidate_family(
    market: str,
    rows: Sequence[CandidateIssuanceRow],
    *,
    issuance_id: str,
    closeouts: Mapping[str, MarketCloseoutResult],
    scores: Mapping[str, tuple[float, float]],
) -> CandidateFamily:
    results = [closeouts[candidate_issuance_row_id(issuance_id, row)] for row in rows]
    clv, kind = _clv_values(results)
    outcomes = Counter(grade_candidate_outcome(row, scores.get(row.game_id)) for row in rows)
    return CandidateFamily(
        market=market,
        candidates=len(rows),
        sides=len({(row.game_id, row.side) for row in rows}),
        closeout=_status_counts(result.status.value for result in results),
        clv_kind=kind,
        clv_count=len(clv),
        median_clv=_median(clv),
        share_positive_clv=_share_positive(clv),
        wins=outcomes[CandidateOutcome.WIN],
        losses=outcomes[CandidateOutcome.LOSS],
        pushes=outcomes[CandidateOutcome.PUSH],
        outcome_unavailable=outcomes[CandidateOutcome.UNAVAILABLE]
        + outcomes[CandidateOutcome.CONFLICT],
    )


# ---------------------------------------------------------------------------
# Recorded wagers
# ---------------------------------------------------------------------------


def _ordered(bets: DataFrame) -> DataFrame:
    """Return the bets in a stable order that does not depend on internal ids."""
    if bets.empty:
        return bets.reset_index(drop=True)
    return bets.sort_values(["placed_at", "source_bet_id"], kind="stable").reset_index(drop=True)


def ledger_digest(bets: DataFrame) -> str:
    """Digest the scoped ledger rows without internal ids, so a settlement changes it."""
    rows = _ordered(bets).drop(columns=list(_INTERNAL_BET_COLUMNS), errors="ignore")
    return frame_content_digest(rows.astype(object))


def wager_closeouts(bets: DataFrame, quotes: DataFrame) -> dict[str, MarketCloseoutResult]:
    """Close every bet the quote ledger can close, keyed by ``bet_id``."""
    closable = bets.loc[
        bets["market_type"].isin(_CLOSABLE_BET_MARKETS)
        & bets["game_id"].notna()
        & bets["side"].notna()
    ]
    if closable.empty:
        return {}
    return {
        result.reference.reference_id: result for result in close_recorded_wagers(closable, quotes)
    }


def wager_summary(
    bets: DataFrame,
    closeouts: Mapping[str, MarketCloseoutResult] | None = None,
) -> WagerSummary:
    """Summarize recorded wagers through :mod:`gridiron_edge.betting.performance`."""
    closeouts = closeouts or {}
    ordered = _ordered(bets)
    counts = record(ordered)
    money = roi(ordered)
    settled = not counts.empty
    clv_available = sum(
        closeouts[bet_id].status is MarketCloseoutStatus.AVAILABLE
        for bet_id in ordered.get("bet_id", pd.Series(dtype=object))
        if bet_id in closeouts
    )
    roi_pct = float(money["roi_pct"].iloc[0]) if settled else math.nan
    return WagerSummary(
        bets=len(ordered),
        settled=int(counts["total"].iloc[0]) if settled else 0,
        unsettled=int(ordered["status"].eq("open").sum()) if len(ordered) else 0,
        wins=int(counts["wins"].iloc[0]) if settled else 0,
        losses=int(counts["losses"].iloc[0]) if settled else 0,
        pushes=int(counts["pushes"].iloc[0]) if settled else 0,
        staked=math.fsum(float(value) for value in ordered.get("stake", [])),
        settled_staked=float(money["total_staked"].iloc[0]) if settled else 0.0,
        pnl=float(money["total_pnl"].iloc[0]) if settled else 0.0,
        roi_pct=None if math.isnan(roi_pct) else roi_pct,
        bonus_funded=int(ordered["funding_type"].eq("bonus").sum()) if len(ordered) else 0,
        clv_available=clv_available,
        clv_unavailable=len(ordered) - clv_available,
        linked_recommendations=(
            int(ordered["recommended_bet_result_id"].notna().sum()) if len(ordered) else 0
        ),
    )


def recorded_wagers(bets: DataFrame, quotes: DataFrame) -> RecordedWagers:
    """Summarize the week's recorded wagers overall and per market type."""
    closeouts = wager_closeouts(bets, quotes)
    markets: list[WagerMarket] = []
    for market_type in sorted(bets["market_type"].dropna().astype(str).unique()):
        rows = bets.loc[bets["market_type"].eq(market_type)]
        clv, kind = _clv_values(
            closeouts[str(bet_id)] for bet_id in rows["bet_id"] if str(bet_id) in closeouts
        )
        markets.append(
            WagerMarket(
                market_type=market_type,
                summary=wager_summary(rows, closeouts),
                clv_kind=kind,
                median_clv=_median(clv),
                share_positive_clv=_share_positive(clv),
            )
        )
    return RecordedWagers(total=wager_summary(bets, closeouts), markets=tuple(markets))


def sum_wager_summaries(summaries: Iterable[WagerSummary]) -> WagerSummary:
    """Add weekly summaries into one; ROI is recomputed from the summed P&L and stake."""
    items = list(summaries)
    settled_staked = math.fsum(item.settled_staked for item in items)
    pnl = math.fsum(item.pnl for item in items)
    return WagerSummary(
        bets=sum(item.bets for item in items),
        settled=sum(item.settled for item in items),
        unsettled=sum(item.unsettled for item in items),
        wins=sum(item.wins for item in items),
        losses=sum(item.losses for item in items),
        pushes=sum(item.pushes for item in items),
        staked=math.fsum(item.staked for item in items),
        settled_staked=settled_staked,
        pnl=pnl,
        roi_pct=pnl / settled_staked * 100 if settled_staked > 0 else None,
        bonus_funded=sum(item.bonus_funded for item in items),
        clv_available=sum(item.clv_available for item in items),
        clv_unavailable=sum(item.clv_unavailable for item in items),
        linked_recommendations=sum(item.linked_recommendations for item in items),
    )


__all__ = [
    "CANDIDATE_MARKETS",
    "CandidateFamily",
    "CandidateRecommendationState",
    "EdgeEvidence",
    "EdgeEvidenceStatus",
    "EdgeReceiptEntry",
    "MarketInputs",
    "ProductionChain",
    "RecordedWagers",
    "StatusCount",
    "WagerMarket",
    "WagerSummary",
    "edge_evidence",
    "edge_receipt_window",
    "ledger_digest",
    "production_chain",
    "receipt_blockers",
    "recorded_wagers",
    "sum_wager_summaries",
    "wager_closeouts",
    "wager_summary",
]
