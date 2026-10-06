"""Tests for the weekly recap's edge-evidence and recorded-wager sections."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.recommended_bet_results import policy
from tests.fixtures.weekly_recap import (
    BUILT_AT,
    GAME_A,
    GAME_B,
    GAME_C,
    PRODUCT_RUN,
    SEASON,
    WEEK,
    RecapWeek,
    bet,
    edge_receipt,
    ledger,
    recap_week,
)

from gridiron_edge.evaluation.weekly_recap import (
    BuiltWeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
    weekly_recap_from_payload,
    weekly_recap_payload,
)
from gridiron_edge.evaluation.weekly_recap_markets import (
    CandidateRecommendationState,
    EdgeEvidenceStatus,
    MarketInputs,
    edge_evidence,
    ledger_digest,
    production_chain,
    receipt_blockers,
    recorded_wagers,
    sum_wager_summaries,
    wager_summary,
)
from gridiron_edge.evaluation.weekly_recap_season import unattributed_bets
from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS, empty_quote_frame
from gridiron_edge.market.candidate_issuance import (
    CANDIDATE_ISSUANCE_SCHEMA_VERSION,
    CandidateIssuance,
    CandidateIssuanceReason,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_id,
)
from gridiron_edge.market.edge_diagnostics import EdgeDiagnosticBlocker
from gridiron_edge.market.recommended_bet_result import (
    RecommendedBetResultState,
    evaluate_recommendation_issuance,
)
from gridiron_edge.market.recommended_bet_result_store import LoadedRecommendedBetResult

pytestmark = pytest.mark.unit

# Game A kicks off 2026-10-04 13:00 ET; both quotes are pregame.
KICKOFF_A = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
EARLY = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
LATE = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
LAST_KICKOFF = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)
SCORES = {GAME_A: (17.0, 24.0), GAME_B: (20.0, 20.0), GAME_C: (30.0, 27.0)}
PRODUCT_ID = "weekly_product"


def _implied(american: int) -> float:
    return -american / (-american + 100) if american < 0 else 100 / (american + 100)


# Moneyline CLV of a -150 home price that closed at -170: (close - bet) / bet probability.
HAND_CLV = (_implied(-170) - _implied(-150)) / _implied(-150)


def _quote(fetched_at: datetime, odds: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "fetched_at": fetched_at,
        "provider": "the_odds_api",
        "provider_event_id": "event-a",
        "sportsbook": "draftkings",
        "sportsbook_updated_at": fetched_at,
        "commence_time": KICKOFF_A,
        "is_live": False,
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_A,
        "game_date": "2026-10-04",
        "away_team": "Away",
        "home_team": "Home",
        "market": "moneyline",
        "side": "home",
        "odds": odds,
        "line": None,
    }
    return row | overrides


def _quotes() -> DataFrame:
    return DataFrame([_quote(EARLY, -150.0), _quote(LATE, -170.0)], columns=list(QUOTE_COLUMNS))


def _row(
    game_id: str,
    market: str,
    *,
    line: float | None = None,
    state: CandidateIssuanceState = CandidateIssuanceState.CANDIDATE,
    sportsbook: str = "draftkings",
) -> CandidateIssuanceRow:
    reason = {
        CandidateIssuanceState.CANDIDATE: CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
        CandidateIssuanceState.NOT_CANDIDATE: CandidateIssuanceReason.EXPECTED_VALUE_NOT_POSITIVE,
    }[state]
    return CandidateIssuanceRow(
        game_id,
        market,
        "home",
        "the_odds_api",
        "event-a" if game_id == GAME_A else f"event-{game_id}",
        sportsbook,
        line,
        -150,
        EARLY,
        EARLY,
        KICKOFF_A,
        False,
        f"forecast-{game_id}",
        PRODUCT_RUN,
        "live",
        EARLY,
        "win_prob",
        "random_forest",
        0.65,
        0.04,
        state,
        reason,
    )


def _issuance(*rows: CandidateIssuanceRow) -> CandidateIssuance:
    evaluated = datetime(2026, 10, 2, 12, 5, tzinfo=UTC)
    return CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        candidate_issuance_id(
            product_id=PRODUCT_ID,
            product_run_id=PRODUCT_RUN,
            season=SEASON,
            week=WEEK,
            evaluated_at=evaluated,
        ),
        PRODUCT_ID,
        PRODUCT_RUN,
        EARLY,
        SEASON,
        WEEK,
        evaluated,
        rows,
    )


def _chain_issuance() -> CandidateIssuance:
    """A won moneyline candidate at two books, a pushed spread candidate, one non-candidate."""
    return _issuance(
        _row(GAME_A, "moneyline"),
        _row(GAME_A, "moneyline", sportsbook="fanduel"),
        _row(GAME_B, "spread", line=0.0),
        _row(GAME_C, "total", line=50.0, state=CandidateIssuanceState.NOT_CANDIDATE),
    )


def _results(issuance: CandidateIssuance) -> tuple[LoadedRecommendedBetResult, ...]:
    evaluation = evaluate_recommendation_issuance(
        policy=policy(active=False),
        issuance=issuance,
        decision_at=datetime(2026, 10, 2, 12, 10, tzinfo=UTC),
        bankroll=None,
        portfolio=None,
    )
    return tuple(
        LoadedRecommendedBetResult(evaluation.evaluation_id, item) for item in evaluation.results
    )


def _inputs(**overrides: object) -> MarketInputs:
    values: dict[str, object] = {
        "issuances": (),
        "recommendations": (),
        "receipts": (),
        "bets": ledger().iloc[0:0],
        "quotes": empty_quote_frame(),
    }
    return MarketInputs(**(values | overrides))  # type: ignore[arg-type]


WITHHELD = "edge calculation blocked: model_edges_unvalidated"


# ---------------------------------------------------------------------------
# Edge evidence
# ---------------------------------------------------------------------------


def test_the_edge_stage_detail_names_its_blockers() -> None:
    from gridiron_edge.cli.weekly_predict import _edge_stage_detail
    from gridiron_edge.market.edge_diagnostics import EdgeDiagnostics, EdgeResultState
    from gridiron_edge.market.edge_report import EdgeResult

    diagnostics = EdgeDiagnostics(
        season=SEASON,
        week=WEEK,
        prediction_game_count=15,
        market_game_count=15,
        matched_game_count=15,
        complete_moneyline_count=15,
        complete_spread_count=15,
        complete_total_count=15,
        eligible_market_count=45,
        calculated_edge_count=0,
        positive_edge_count=0,
        filtered_edge_count=0,
        state=EdgeResultState.BLOCKED,
        blockers=(
            EdgeDiagnosticBlocker.MARKET_STALE,
            EdgeDiagnosticBlocker.MODEL_EDGES_UNVALIDATED,
        ),
    )
    detail = _edge_stage_detail(
        EdgeResult(rows=pd.DataFrame(), diagnostics=diagnostics), min_ev=0.0
    )

    assert receipt_blockers(detail) == ("market_stale", "model_edges_unvalidated")
    assert receipt_blockers("45 edges (top EV 27.1%)") == ()


def test_the_latest_receipt_before_the_last_kickoff_decides_the_status() -> None:
    receipts = (
        edge_receipt("2026-10-04T20:00:54+00:00", "done", "149 edges (top EV 27.1%)"),
        edge_receipt("2026-10-04T22:00:54+00:00", "blocked", WITHHELD),
        edge_receipt("2026-10-06T01:00:00+00:00", "done", "3 edges"),
        edge_receipt("2026-10-04T21:00:00+00:00", "unchanged", "not an edge receipt"),
    )
    receipts = (*receipts[:3], replace(receipts[3], attempted="watch"))

    evidence = edge_evidence(_inputs(receipts=receipts), last_kickoff=LAST_KICKOFF, scores=SCORES)

    assert evidence.status is EdgeEvidenceStatus.WITHHELD_UNVALIDATED
    assert evidence.chain is None
    assert evidence.receipts_through == LAST_KICKOFF
    assert [(item.outcome, item.detail) for item in evidence.receipts] == [
        ("done", "149 edges (top EV 27.1%)"),
        ("blocked", WITHHELD),
    ]
    assert evidence.receipts[0].product_id == PRODUCT_ID
    assert evidence.receipts[0].market_fetched_at == "2026-10-04T20:00:54+00:00"


@pytest.mark.parametrize(
    ("outcome", "detail"),
    [
        ("done", "45 edges (top EV 27.1%)"),
        ("blocked", "edge calculation blocked: market_stale (newest quotes fetched x)"),
        ("failed", "boom"),
    ],
)
def test_a_published_or_otherwise_blocked_week_is_not_recorded(outcome: str, detail: str) -> None:
    receipts = (
        edge_receipt("2026-10-04T20:00:00+00:00", "blocked", WITHHELD),
        edge_receipt("2026-10-04T22:00:00+00:00", outcome, detail),
    )

    evidence = edge_evidence(_inputs(receipts=receipts), last_kickoff=LAST_KICKOFF, scores=SCORES)

    assert evidence.status is EdgeEvidenceStatus.NOT_RECORDED
    assert len(evidence.receipts) == 2


def test_a_week_without_receipts_is_not_recorded() -> None:
    evidence = edge_evidence(_inputs(), last_kickoff=LAST_KICKOFF, scores=SCORES)

    assert evidence.status is EdgeEvidenceStatus.NOT_RECORDED
    assert evidence.receipts == ()


def test_an_issuance_reads_production_chain_whatever_the_receipts_say() -> None:
    issuance = _chain_issuance()
    inputs = _inputs(
        issuances=(issuance,),
        receipts=(edge_receipt("2026-10-04T22:00:00+00:00", "blocked", WITHHELD),),
        quotes=_quotes(),
    )

    evidence = edge_evidence(inputs, last_kickoff=LAST_KICKOFF, scores=SCORES)

    assert evidence.status is EdgeEvidenceStatus.PRODUCTION_CHAIN
    assert evidence.chain is not None
    assert evidence.chain.issuance_id == issuance.issuance_id
    assert len(evidence.receipts) == 1


def test_the_chain_counts_rows_states_closeout_clv_and_outcomes() -> None:
    issuance = _chain_issuance()
    results = _results(issuance)
    assert {item.result.result_state for item in results} == {RecommendedBetResultState.UNAVAILABLE}

    # The fanduel moneyline row has no result: it reads not_evaluated.
    chain = production_chain(issuance, results[1:], _quotes(), scores=SCORES)

    assert (chain.rows, chain.candidates, chain.not_candidates, chain.unavailable) == (4, 3, 1, 0)
    assert chain.issued_at == issuance.evaluated_at
    assert chain.evaluation_ids == (results[0].evaluation_id,)
    assert chain.policy_ids == (results[0].result.policy_id,)
    assert {(item.status, item.count) for item in chain.recommendations} == {
        (CandidateRecommendationState.UNAVAILABLE.value, 2),
        (CandidateRecommendationState.NOT_EVALUATED.value, 1),
    }
    moneyline, spread, total = chain.families
    assert (moneyline.market, moneyline.candidates, moneyline.sides) == ("moneyline", 2, 1)
    # draftkings closes at the later -170 quote; fanduel has no quote of its own.
    assert [(item.status, item.count) for item in moneyline.closeout] == [
        ("available", 1),
        ("closeout_missing", 1),
    ]
    assert moneyline.clv_kind == "moneyline_price"
    assert moneyline.clv_count == 1
    assert moneyline.median_clv == pytest.approx(HAND_CLV)
    assert moneyline.share_positive_clv == 1.0
    assert (moneyline.wins, moneyline.losses, moneyline.pushes) == (2, 0, 0)
    # Home +0 in a 20-20 tie pushes.
    assert (spread.candidates, spread.pushes, spread.clv_count, spread.median_clv) == (
        1,
        1,
        0,
        None,
    )
    assert total.candidates == 0 and total.closeout == ()


def test_more_than_one_issuance_cannot_stand_for_the_week() -> None:
    issuance = _chain_issuance()

    with pytest.raises(ValueError, match="At most one"):
        edge_evidence(
            _inputs(issuances=(issuance, issuance)), last_kickoff=LAST_KICKOFF, scores=SCORES
        )


def test_every_result_state_has_a_candidate_state() -> None:
    for state in RecommendedBetResultState:
        assert CandidateRecommendationState(state.value).value == state.value


# ---------------------------------------------------------------------------
# Recorded wagers
# ---------------------------------------------------------------------------


def _week_bets() -> DataFrame:
    bets = ledger()
    return bets.loc[bets["game_id"].notna()].reset_index(drop=True)


def test_wagers_read_record_money_unsettled_and_clv_unavailable() -> None:
    wagers = recorded_wagers(_week_bets(), empty_quote_frame())

    total = wagers.total
    assert (total.bets, total.settled, total.unsettled) == (4, 3, 1)
    assert (total.wins, total.losses, total.pushes) == (1, 2, 0)
    assert total.staked == pytest.approx(3.1)
    assert total.settled_staked == pytest.approx(2.1)
    assert total.pnl == pytest.approx(0.66 - 1.0 - 0.1)
    assert total.roi_pct == pytest.approx((0.66 - 1.1) / 2.1 * 100)
    assert (total.bonus_funded, total.linked_recommendations) == (1, 0)
    # Imports carry no quote reference, and a parlay has no single quote.
    assert (total.clv_available, total.clv_unavailable) == (0, 4)
    moneyline, parlay = wagers.markets
    assert (moneyline.market_type, moneyline.summary.bets, moneyline.summary.unsettled) == (
        "moneyline",
        3,
        1,
    )
    assert (parlay.market_type, parlay.summary.losses, parlay.clv_kind) == (
        "same_game_parlay",
        1,
        None,
    )


def test_a_referenced_bet_closes_against_the_quote_ledger() -> None:
    referenced = bet("DK9", GAME_A, status="won", paid=1.66)
    referenced |= {
        "reference_provider": "the_odds_api",
        "reference_provider_event_id": "event-a",
        "reference_sportsbook": "draftkings",
        "reference_market_fetched_at": pd.Timestamp(EARLY),
        "reference_sportsbook_updated_at": pd.Timestamp(EARLY),
        "reference_commence_time": pd.Timestamp(KICKOFF_A),
        "reference_american_odds": -150,
        "recommended_bet_result_id": "r" * 64,
    }
    bets = DataFrame([referenced], columns=ledger().columns)

    wagers = recorded_wagers(bets, _quotes())

    assert (wagers.total.clv_available, wagers.total.clv_unavailable) == (1, 0)
    assert wagers.total.linked_recommendations == 1
    (moneyline,) = wagers.markets
    assert moneyline.clv_kind == "moneyline_price"
    assert moneyline.median_clv == pytest.approx(HAND_CLV)


def test_no_bets_is_an_empty_summary() -> None:
    summary = wager_summary(ledger().iloc[0:0])

    assert (summary.bets, summary.settled, summary.pnl, summary.roi_pct) == (0, 0, 0.0, None)


def test_summaries_add_up_and_recompute_roi() -> None:
    week = recorded_wagers(_week_bets(), empty_quote_frame()).total

    total = sum_wager_summaries([week, week])

    assert (total.bets, total.wins, total.unsettled) == (8, 2, 2)
    assert total.pnl == pytest.approx(2 * week.pnl)
    assert total.roi_pct == pytest.approx(week.roi_pct)
    assert sum_wager_summaries([]).roi_pct is None


def test_the_ledger_digest_follows_settlement_not_internal_ids() -> None:
    bets = _week_bets()
    reimported = bets.assign(bet_id=[f"new-{index}" for index in range(len(bets))])
    settled = bets.copy()
    settled.loc[settled["status"] == "open", ["status", "pnl"]] = ["lost", -1.0]

    assert ledger_digest(reimported) == ledger_digest(bets)
    assert ledger_digest(settled) != ledger_digest(bets)


def test_unattributed_bets_are_the_season_window_without_a_game() -> None:
    bets = DataFrame(
        [
            *ledger().to_dict("records"),
            bet("OLD", None, market_type="parlay", side=None, placed_at="2026-02-28T12:00:00Z"),
            bet("NEXT", None, market_type="parlay", side=None, placed_at="2027-03-01T00:00:00Z"),
            bet("IN", None, market_type="parlay", side=None, placed_at="2027-02-28T23:59:00Z"),
        ],
        columns=ledger().columns,
    )

    selected = unattributed_bets(bets, season=SEASON)

    assert sorted(selected["source_bet_id"]) == ["DK5", "IN"]


# ---------------------------------------------------------------------------
# The recap
# ---------------------------------------------------------------------------


def _build(week: RecapWeek | None = None, **markets: object) -> BuiltWeeklyRecap:
    week = week or recap_week()
    return build_weekly_recap(
        season=SEASON,
        week=WEEK,
        schedule=week.schedule,
        product=week.product,
        forecast_events=week.events,
        games=week.games,
        player_game_logs=week.logs,
        sidecar_captures=week.captures,
        markets=replace(week.markets, **markets),  # type: ignore[arg-type]
        built_at=BUILT_AT,
    )


def test_the_recap_carries_both_sections_and_round_trips() -> None:
    built = _build()

    assert built.recap.edges.status is EdgeEvidenceStatus.WITHHELD_UNVALIDATED
    assert built.recap.wagers.total.bets == 4
    assert built.recap.ledger_digest == ledger_digest(recap_week().markets.bets)
    assert weekly_recap_from_payload(weekly_recap_payload(built.recap)) == built.recap


def test_a_chain_week_round_trips_through_json() -> None:
    built = _build(issuances=(_chain_issuance(),), quotes=_quotes())

    assert built.recap.edges.chain is not None
    assert weekly_recap_from_payload(weekly_recap_payload(built.recap)) == built.recap


def test_a_settlement_or_a_new_quote_gives_a_new_identity() -> None:
    week = recap_week()
    settled = week.markets.bets.copy()
    settled.loc[settled["status"] == "open", ["status", "pnl"]] = ["lost", -1.0]

    base = _build(week).recap.recap_id

    assert _build(week, bets=settled).recap.recap_id != base
    assert _build(week, quotes=_quotes()).recap.recap_id != base
    assert _build(week).recap.recap_id == base


def test_two_issuances_refuse_the_week() -> None:
    with pytest.raises(WeeklyRecapRefusalError, match="2 candidate issuances"):
        _build(issuances=(_chain_issuance(), _chain_issuance()))


def test_an_issuance_of_another_product_is_rejected() -> None:
    other = replace(_chain_issuance(), product_run_id="another-run")

    with pytest.raises(ValueError, match="does not cite the selected product"):
        _build(issuances=(other,))


def test_a_wager_on_an_unscheduled_game_refuses_the_week() -> None:
    bets = DataFrame([bet("DK8", "2026_04_ZZZ_YYY")], columns=ledger().columns)

    with pytest.raises(WeeklyRecapRefusalError, match="2026_04_ZZZ_YYY"):
        _build(bets=bets)
