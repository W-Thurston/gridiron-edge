# tests/unit/cli/test_production_chain_cli.py
"""Tests for production-chain CLI reporting."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from gridiron_edge.cli.main import app
from gridiron_edge.core.settings import Settings
from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS
from gridiron_edge.market.candidate_issuance import (
    CANDIDATE_ISSUANCE_SCHEMA_VERSION,
    CandidateIssuance,
    CandidateIssuanceReason,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_id,
)
from gridiron_edge.market.recommendation_policy import (
    RECOMMENDATION_POLICY_DERIVATION_METHOD,
    RECOMMENDATION_POLICY_SCHEMA_VERSION,
    EmpiricalQualificationThresholds,
    MarketFamilyRecommendationPolicy,
    PolicyDerivationReason,
    PolicyDerivationStatus,
    PolicyValueSource,
    PortfolioAllocationReason,
    PortfolioAllocationState,
    RecommendationDecisionState,
    RecommendationPolicy,
    RecommendationPolicyGovernance,
    StakeRoundingMode,
    governance_fingerprint,
    recommendation_policy_id,
)
from gridiron_edge.market.recommended_bet_result import (
    RecommendedBetEvaluation,
    RecommendedBetResultState,
)

runner = CliRunner()

_BANKROLL_WIRING_FETCHED = datetime(2026, 9, 1, 12, tzinfo=UTC)
_BANKROLL_WIRING_EVALUATED = datetime(2026, 9, 1, 12, 5, tzinfo=UTC)
_BANKROLL_WIRING_DECISION = datetime(2026, 9, 1, 12, 10, tzinfo=UTC)
_BANKROLL_WIRING_KICKOFF = datetime(2026, 9, 1, 20, tzinfo=UTC)


def _bankroll_wiring_settings(repo_root: Path) -> Settings:
    """Build a real, fully-populated Settings instance for one isolated
    test repo root -- mirrors the project's own established fixture
    pattern (tests/unit/api/test_deps.py::_make_settings)."""
    return Settings(
        repo_root=repo_root,
        owm_api_key=None,
        odds_api_key=None,
        data_raw=repo_root / "data" / "raw",
        data_cleaned=repo_root / "data" / "cleaned",
        data_modeling=repo_root / "data" / "modeling",
        data_output=repo_root / "data" / "output",
    )


def _bankroll_wiring_governance() -> RecommendationPolicyGovernance:
    return RecommendationPolicyGovernance(
        0.25, 5.0, 1.0, StakeRoundingMode.DOWN, 0.02, 0.05, 0.20, True, False, ("open",)
    )


def _bankroll_wiring_family(market: str, *, active: bool) -> MarketFamilyRecommendationPolicy:
    return MarketFamilyRecommendationPolicy(
        market,
        PolicyDerivationStatus.ACTIVE if active else PolicyDerivationStatus.INSUFFICIENT_EVIDENCE,
        PolicyDerivationReason.DERIVED
        if active
        else PolicyDerivationReason.NO_VALIDATED_THRESHOLD_SELECTION_METHOD,
        2,
        2,
        2,
        2,
        (("synthetic_test_evidence", "available"),),
        EmpiricalQualificationThresholds(0.01, 900.0, None, None) if active else None,
        PolicyValueSource.EMPIRICAL_MARKET_FAMILY_EVIDENCE,
    )


def _bankroll_wiring_policy() -> RecommendationPolicy:
    governance = _bankroll_wiring_governance()
    policy = RecommendationPolicy(
        RECOMMENDATION_POLICY_SCHEMA_VERSION,
        "0" * 64,
        datetime(2026, 8, 17, tzinfo=UTC),
        "a" * 64,
        governance_fingerprint(governance),
        RECOMMENDATION_POLICY_DERIVATION_METHOD,
        _bankroll_wiring_family("moneyline", active=True),
        _bankroll_wiring_family("spread", active=True),
        _bankroll_wiring_family("total", active=True),
        governance,
    )
    return replace(policy, policy_id=recommendation_policy_id(policy))


def _bankroll_wiring_row() -> CandidateIssuanceRow:
    return CandidateIssuanceRow(
        "2026_01_KC_LAC",
        "moneyline",
        "home",
        "the_odds_api",
        "event-1",
        "draftkings",
        None,
        100,
        _BANKROLL_WIRING_FETCHED,
        _BANKROLL_WIRING_FETCHED,
        _BANKROLL_WIRING_KICKOFF,
        False,
        "forecast-1",
        "forecast-run",
        "champion",
        _BANKROLL_WIRING_FETCHED,
        "win_prob",
        "random_forest",
        0.60,
        0.20,
        CandidateIssuanceState.CANDIDATE,
        CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
    )


def _bankroll_wiring_issuance() -> CandidateIssuance:
    issuance_id = candidate_issuance_id(
        product_id="product-1",
        product_run_id="run-1",
        season="2026-2027",
        week=1,
        evaluated_at=_BANKROLL_WIRING_EVALUATED,
    )
    return CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        "product-1",
        "run-1",
        _BANKROLL_WIRING_FETCHED,
        "2026-2027",
        1,
        _BANKROLL_WIRING_EVALUATED,
        (_bankroll_wiring_row(),),
    )


def _patch_bankroll_wiring_issuance_and_policy(monkeypatch) -> None:
    """Patch issuance/policy reads at their SOURCE modules (this file's
    established convention: production_chain.py imports these locally
    inside the command function)."""
    issuance = _bankroll_wiring_issuance()
    policy = _bankroll_wiring_policy()
    monkeypatch.setattr(
        "gridiron_edge.market.candidate_issuance_store.read_candidate_issuance",
        lambda *_a, **_k: issuance,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.recommendation_policy_store.read_recommendation_policy",
        lambda *_a, **_k: policy,
    )


def _invoke_bankroll_wiring_evaluate_recommendations():
    return runner.invoke(
        app,
        [
            "production-chain",
            "evaluate-recommendations",
            "--issuance-id",
            "a" * 64,
            "--policy-id",
            "b" * 64,
            "--decision-at",
            _BANKROLL_WIRING_DECISION.isoformat(),
            "--write",
        ],
    )


def _capture_written_bankroll_evaluation(monkeypatch, tmp_path: Path) -> dict:
    written: dict[str, object] = {}
    monkeypatch.setattr(
        "gridiron_edge.market.recommended_bet_result_store.write_recommended_bet_evaluation",
        lambda value, **_kwargs: written.update(evaluation=value) or (tmp_path / "eval.json"),
    )
    return written


def _record_bankroll_deposit_at(
    repo: Path,
    *,
    amount: float,
    timestamp: datetime,
) -> None:
    """Record a real bankroll deposit at an explicit test timestamp."""
    import uuid

    import gridiron_edge.betting.bankroll as bankroll_module

    transactions = bankroll_module._read_txn_log(repo)

    deposit_row = pd.DataFrame(
        [
            {
                "txn_id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "txn_type": "deposit",
                "amount": amount,
                "reference_id": None,
                "note": None,
            }
        ]
    )

    if transactions.empty:
        updated = deposit_row.copy()
    else:
        updated = pd.concat(
            [transactions, deposit_row],
            ignore_index=True,
        )

    bankroll_module._write_txn_log(updated, repo)


class TestEvaluateRecommendationsBankrollWiring:
    """CLI-level acceptance tests: bankroll evidence reaches the governed
    recommendation writer and produces the promised observable domain
    outcomes. Where practical, these assert on the actual persisted
    RecommendedBetEvaluation/Result objects, not stdout text."""

    def test_empty_bankroll_history_yields_eligible_recommendation_with_unevaluated_allocation(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """No bankroll transactions at all -> bankroll_snapshot_as_of returns
        None -> bankroll=None. Under the corrected Stage 1/Stage 2 model, this
        candidate is still a genuinely eligible recommendation (bankroll is a
        Stage 2 / allocation-evidence concern, not a Stage 1 / recommendation-
        eligibility concern) -- but allocation itself could not be evaluated,
        so the persisted result_state is UNAVAILABLE, not RECOMMENDED, and the
        allocation axis explicitly records why."""
        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            lambda: _bankroll_wiring_settings(tmp_path),
        )
        _patch_bankroll_wiring_issuance_and_policy(monkeypatch)
        written = _capture_written_bankroll_evaluation(monkeypatch, tmp_path)

        result = _invoke_bankroll_wiring_evaluate_recommendations()

        assert result.exit_code == 0
        evaluation: RecommendedBetEvaluation = written["evaluation"]
        assert len(evaluation.results) == 1
        wager_result = evaluation.results[0]
        assert wager_result.decision_state is RecommendationDecisionState.RECOMMENDATION_ELIGIBLE
        assert wager_result.recommendation_eligible
        assert wager_result.result_state is RecommendedBetResultState.UNAVAILABLE
        assert wager_result.allocation.state is PortfolioAllocationState.NOT_EVALUATED
        assert (
            wager_result.allocation.reason
            is PortfolioAllocationReason.ALLOCATION_EVIDENCE_UNAVAILABLE
        )
        assert wager_result.bankroll_basis is None

    def test_pre_decision_deposit_supplies_exact_bankroll_basis(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A real, pre-decision deposit produces the exact expected
        BankrollBasis, derived by the real bankroll_snapshot_as_of
        function running against a real, isolated transaction log."""
        from gridiron_edge.betting.bankroll import bankroll_snapshot_as_of

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            lambda: _bankroll_wiring_settings(tmp_path),
        )
        _patch_bankroll_wiring_issuance_and_policy(monkeypatch)
        written = _capture_written_bankroll_evaluation(monkeypatch, tmp_path)

        _record_bankroll_deposit_at(
            tmp_path,
            amount=1000.0,
            timestamp=_BANKROLL_WIRING_FETCHED,
        )
        expected_snapshot = bankroll_snapshot_as_of(_BANKROLL_WIRING_DECISION, repo=tmp_path)
        assert expected_snapshot is not None  # sanity check on the test's own arrangement

        result = _invoke_bankroll_wiring_evaluate_recommendations()

        assert result.exit_code == 0
        evaluation: RecommendedBetEvaluation = written["evaluation"]
        wager_result = evaluation.results[0]
        assert wager_result.bankroll_basis is not None
        assert wager_result.bankroll_basis.amount == expected_snapshot.amount
        assert wager_result.bankroll_basis.observed_at == expected_snapshot.observed_at
        assert wager_result.bankroll_basis.source_kind == expected_snapshot.source_kind
        assert wager_result.bankroll_basis.source_id == expected_snapshot.source_id

    def test_supplied_bankroll_reaches_recommendation_eligible(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """With a real, sufficient, pre-decision deposit and an ACTIVE
        policy whose other checks pass, the decision genuinely reaches
        RECOMMENDATION_ELIGIBLE -- not merely 'bankroll was non-null'."""
        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            lambda: _bankroll_wiring_settings(tmp_path),
        )
        _patch_bankroll_wiring_issuance_and_policy(monkeypatch)
        written = _capture_written_bankroll_evaluation(monkeypatch, tmp_path)

        _record_bankroll_deposit_at(
            tmp_path,
            amount=1000.0,
            timestamp=_BANKROLL_WIRING_FETCHED,
        )
        result = _invoke_bankroll_wiring_evaluate_recommendations()

        assert result.exit_code == 0
        evaluation: RecommendedBetEvaluation = written["evaluation"]
        wager_result = evaluation.results[0]
        assert wager_result.decision_state is RecommendationDecisionState.RECOMMENDATION_ELIGIBLE
        assert wager_result.result_state is RecommendedBetResultState.RECOMMENDED

    def test_post_cutoff_transaction_does_not_affect_evaluation(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A transaction recorded strictly after the decision cutoff does
        not change the bankroll evidence used, and therefore does not
        change the resulting decision -- proven against the real
        bankroll ledger, not a mock."""
        import uuid as uuid_module

        import pandas as pd

        import gridiron_edge.betting.bankroll as bankroll_module

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            lambda: _bankroll_wiring_settings(tmp_path),
        )
        _patch_bankroll_wiring_issuance_and_policy(monkeypatch)

        _record_bankroll_deposit_at(
            tmp_path,
            amount=1000.0,
            timestamp=_BANKROLL_WIRING_FETCHED,
        )
        written_before = _capture_written_bankroll_evaluation(monkeypatch, tmp_path)
        result_before = _invoke_bankroll_wiring_evaluate_recommendations()
        assert result_before.exit_code == 0
        evaluation_before: RecommendedBetEvaluation = written_before["evaluation"]

        post_cutoff = _BANKROLL_WIRING_DECISION + timedelta(hours=1)
        df = bankroll_module._read_txn_log(tmp_path)
        later_row = pd.DataFrame(
            [
                {
                    "txn_id": str(uuid_module.uuid4()),
                    "timestamp": post_cutoff,
                    "txn_type": "deposit",
                    "amount": 5000.0,
                    "reference_id": None,
                    "note": None,
                }
            ]
        )
        bankroll_module._write_txn_log(pd.concat([df, later_row], ignore_index=True), tmp_path)

        written_after = _capture_written_bankroll_evaluation(monkeypatch, tmp_path)
        result_after = _invoke_bankroll_wiring_evaluate_recommendations()
        assert result_after.exit_code == 0
        evaluation_after: RecommendedBetEvaluation = written_after["evaluation"]

        before_result = evaluation_before.results[0]
        after_result = evaluation_after.results[0]

        assert before_result.bankroll_basis is not None
        assert before_result.bankroll_basis.amount == 1000.0
        assert after_result.bankroll_basis == before_result.bankroll_basis
        assert after_result.decision_state == before_result.decision_state
        assert after_result.result_state == before_result.result_state

    def test_evaluate_recommendations_uses_named_policy_schema_constant(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Regression guard: recommendation_policy_path is called with
        the real, current value of RECOMMENDATION_POLICY_SCHEMA_VERSION,
        not a hardcoded literal that happens to equal it."""
        captured_schema: dict[str, object] = {}

        def capturing_path(schema_version, policy_id, *, repo=None):
            captured_schema["schema_version"] = schema_version
            return Path("unused.json")

        monkeypatch.setattr(
            "gridiron_edge.core.settings.get_settings",
            lambda: _bankroll_wiring_settings(tmp_path),
        )
        issuance = _bankroll_wiring_issuance()
        monkeypatch.setattr(
            "gridiron_edge.market.candidate_issuance_store.read_candidate_issuance",
            lambda *_a, **_k: issuance,
        )
        monkeypatch.setattr(
            "gridiron_edge.market.recommendation_policy_store.recommendation_policy_path",
            capturing_path,
        )
        monkeypatch.setattr(
            "gridiron_edge.market.recommendation_policy_store.read_recommendation_policy",
            lambda *_a, **_k: _bankroll_wiring_policy(),
        )
        monkeypatch.setattr(
            "gridiron_edge.betting.bankroll.bankroll_snapshot_as_of",
            lambda *_a, **_k: None,
        )

        _invoke_bankroll_wiring_evaluate_recommendations()
        assert captured_schema["schema_version"] == RECOMMENDATION_POLICY_SCHEMA_VERSION


def _canonical_quote_row(
    fetched_at: datetime,
    *,
    sportsbook: str,
    market: str = "spread",
    side: str = "home",
    line: float | None = 1.5,
) -> dict:
    """Build one canonical quote-observation row for CLI fixtures."""
    kickoff = datetime(2026, 9, 10, tzinfo=UTC)
    return {
        "fetched_at": fetched_at,
        "provider": "provider",
        "provider_event_id": "event",
        "sportsbook": sportsbook,
        "sportsbook_updated_at": fetched_at,
        "commence_time": kickoff,
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_A_B",
        "game_date": datetime(2026, 9, 10, tzinfo=UTC),
        "away_team": "A",
        "home_team": "B",
        "market": market,
        "side": side,
        "odds": -110.0,
        "line": line,
    }


def _canonical_quotes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_help_registers_production_chain_group() -> None:
    result = runner.invoke(app, ["production-chain", "--help"])
    assert result.exit_code == 0
    assert "assess" in result.stdout
    assert "verify" in result.stdout


def test_assess_renders_independent_states(monkeypatch) -> None:
    from tests.unit.market.test_production_chain_preflight_store import _preflight

    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.assess_production_chain_preflight",
        lambda **_kwargs: _preflight(),
    )
    result = runner.invoke(
        app,
        [
            "production-chain",
            "assess",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--assessed-at",
            datetime(2026, 8, 17, 18, 57, tzinfo=UTC).isoformat(),
        ],
    )
    assert result.exit_code == 0
    assert "MONEYLINE" in result.stdout
    assert "SPREAD" in result.stdout
    assert "TOTAL" in result.stdout
    assert "UNAVAILABLE" in result.stdout
    assert "PROOF INCOMPLETE" in result.stdout


def test_require_ready_exits_nonzero(monkeypatch) -> None:
    from tests.unit.market.test_production_chain_preflight_store import _preflight

    monkeypatch.setattr(
        "gridiron_edge.market.production_chain_preflight.assess_production_chain_preflight",
        lambda **_kwargs: _preflight(),
    )
    result = runner.invoke(
        app,
        ["production-chain", "assess", "--season", "2026-2027", "--week", "1", "--require-ready"],
    )
    assert result.exit_code == 1


def test_issue_candidates_requires_explicit_utc_timestamp() -> None:
    result = runner.invoke(
        app,
        [
            "production-chain",
            "issue-candidates",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--evaluated-at",
            "2026-08-18T14:45:00",
        ],
    )
    assert result.exit_code == 2
    assert "timezone-aware UTC" in result.stderr


def test_issue_candidates_uses_selected_product_events_and_history(monkeypatch) -> None:
    from datetime import datetime

    import pandas as pd

    from gridiron_edge.market.candidate_issuance import (
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        CandidateIssuance,
        CandidateIssuanceReason,
        CandidateIssuanceRow,
        CandidateIssuanceState,
        candidate_issuance_id,
    )

    evaluated = datetime(2026, 8, 18, 14, 45, tzinfo=UTC)
    issuance_id = candidate_issuance_id(
        product_id="product-1",
        product_run_id="run-1",
        season="2026-2027",
        week=1,
        evaluated_at=evaluated,
    )
    rows = tuple(
        CandidateIssuanceRow(
            game_id="2026_01_A_B",
            market=market,
            side=side,
            provider="provider",
            provider_event_id="event",
            sportsbook=f"book-{index}",
            line=None if market == "moneyline" else 1.5,
            american_price=-110,
            fetched_at=evaluated,
            sportsbook_updated_at=evaluated,
            kickoff=datetime(2026, 9, 10, tzinfo=UTC),
            is_live=False,
            forecast_event_id=f"forecast-{index}",
            forecast_run_id="run-1",
            forecast_role="live",
            forecast_generated_at=evaluated,
            model_name="model",
            model_type="type",
            model_probability=0.6,
            expected_value=0.1,
            state=state,
            reason=reason,
        )
        for index, (market, side, state, reason) in enumerate(
            (
                (
                    "moneyline",
                    "home",
                    CandidateIssuanceState.CANDIDATE,
                    CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
                ),
                (
                    "spread",
                    "away",
                    CandidateIssuanceState.NOT_CANDIDATE,
                    CandidateIssuanceReason.EXPECTED_VALUE_NOT_POSITIVE,
                ),
                (
                    "total",
                    "over",
                    CandidateIssuanceState.UNAVAILABLE,
                    CandidateIssuanceReason.MODEL_UNAVAILABLE,
                ),
            )
        )
    )
    issuance = CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        "product-1",
        "run-1",
        evaluated,
        "2026-2027",
        1,
        evaluated,
        rows,
    )
    product = pd.DataFrame({"product_run_id": ["run-1"]})
    events = pd.DataFrame({"event_id": ["forecast-1"]})
    before = datetime(2026, 8, 18, 11, 0, tzinfo=UTC)
    after = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)
    quotes = _canonical_quotes(
        [
            _canonical_quote_row(before, sportsbook="dk"),
            _canonical_quote_row(before, sportsbook="fd"),
            _canonical_quote_row(evaluated, sportsbook="dk"),
            _canonical_quote_row(after, sportsbook="dk"),
        ]
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "gridiron_edge.datasets.loaders.load_current_weekly_product",
        lambda repo, **kwargs: observed.update(product_scope=(repo, kwargs)) or product,
    )
    monkeypatch.setattr(
        "gridiron_edge.evaluation.forecast_store.load_forecast_events",
        lambda **kwargs: observed.update(event_scope=kwargs) or events,
    )
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.store.load_odds_ledger",
        lambda **kwargs: observed.update(quote_scope=kwargs) or quotes,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.candidate_issuance.issue_pregame_candidates",
        lambda **kwargs: observed.update(issue_args=kwargs) or issuance,
    )

    result = runner.invoke(
        app,
        [
            "production-chain",
            "issue-candidates",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--evaluated-at",
            evaluated.isoformat(),
        ],
    )
    assert result.exit_code == 0
    assert observed["event_scope"]["run_id"] == "run-1"
    assert observed["quote_scope"]["season"] == "2026-2027"
    assert observed["quote_scope"]["week"] == 1
    assert observed["issue_args"]["product"] is product
    assert observed["issue_args"]["forecast_events"] is events
    assert observed["issue_args"]["evaluated_at"] == evaluated
    visible = observed["issue_args"]["quotes"]
    assert list(visible["fetched_at"]) == [before, before, evaluated]
    assert list(visible.columns) == list(QUOTE_COLUMNS)
    assert "MONEYLINE" in result.stdout
    assert "candidate         1" in result.stdout
    assert "SPREAD" in result.stdout
    assert "not candidate     1" in result.stdout
    assert "TOTAL" in result.stdout
    assert "unavailable       1" in result.stdout
    assert issuance_id in result.stdout


def test_issue_candidates_accepts_no_quotes_visible_by_evaluation_time(monkeypatch) -> None:
    from gridiron_edge.market.candidate_issuance import (
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        CandidateIssuance,
        candidate_issuance_id,
    )

    evaluated = datetime(2026, 8, 18, 14, 45, tzinfo=UTC)
    after = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)
    issuance_id = candidate_issuance_id(
        product_id="product-1",
        product_run_id="run-1",
        season="2026-2027",
        week=1,
        evaluated_at=evaluated,
    )
    issuance = CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        "product-1",
        "run-1",
        evaluated,
        "2026-2027",
        1,
        evaluated,
        (),  # zero rows
    )

    product = pd.DataFrame({"product_run_id": ["run-1"]})
    events = pd.DataFrame({"event_id": ["forecast-1"]})
    quotes = _canonical_quotes([_canonical_quote_row(after, sportsbook="dk")])
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "gridiron_edge.datasets.loaders.load_current_weekly_product",
        lambda repo, **kwargs: product,
    )
    monkeypatch.setattr(
        "gridiron_edge.evaluation.forecast_store.load_forecast_events",
        lambda **kwargs: events,
    )
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.store.load_odds_ledger",
        lambda **kwargs: quotes,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.candidate_issuance.issue_pregame_candidates",
        lambda **kwargs: observed.update(issue_args=kwargs) or issuance,
    )

    result = runner.invoke(
        app,
        [
            "production-chain",
            "issue-candidates",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--evaluated-at",
            evaluated.isoformat(),
        ],
    )

    assert result.exit_code == 0
    visible = observed["issue_args"]["quotes"]
    assert visible.empty
    assert list(visible.columns) == list(QUOTE_COLUMNS)
    assert observed["issue_args"]["evaluated_at"] == evaluated
    assert issuance_id in result.stdout


def test_issue_candidates_write_persists_exact_issuance(monkeypatch, tmp_path) -> None:
    import pandas as pd
    from tests.unit.market.test_candidate_issuance_store import _issuance

    from gridiron_edge.ingest.odds.store import empty_quote_frame

    issuance = _issuance()
    stored = tmp_path / "issuance.json"
    monkeypatch.setattr(
        "gridiron_edge.datasets.loaders.load_current_weekly_product",
        lambda *_args, **_kwargs: pd.DataFrame({"product_run_id": ["run-1"]}),
    )
    monkeypatch.setattr(
        "gridiron_edge.evaluation.forecast_store.load_forecast_events",
        lambda **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.store.load_odds_ledger",
        lambda **_kwargs: empty_quote_frame(),
    )
    monkeypatch.setattr(
        "gridiron_edge.market.candidate_issuance.issue_pregame_candidates",
        lambda **_kwargs: issuance,
    )
    monkeypatch.setattr(
        "gridiron_edge.market.candidate_issuance_store.write_candidate_issuance",
        lambda value, **_kwargs: stored if value == issuance else None,
    )

    result = runner.invoke(
        app,
        [
            "production-chain",
            "issue-candidates",
            "--season",
            "2026-2027",
            "--week",
            "1",
            "--evaluated-at",
            issuance.evaluated_at.isoformat(),
            "--write",
        ],
    )
    assert result.exit_code == 0
    assert f"stored              {stored}" in result.stdout


def test_help_registers_governance_commands() -> None:
    result = runner.invoke(app, ["production-chain", "--help"])
    assert result.exit_code == 0
    assert "create-governance" in result.stdout
    assert "verify-governance" in result.stdout


def test_help_registers_policy_and_evaluation_commands() -> None:
    result = runner.invoke(app, ["production-chain", "--help"])
    assert result.exit_code == 0
    assert "derive-policy" in result.stdout
    assert "evaluate-recommendations" in result.stdout


def test_evaluate_recommendations_requires_exact_utc_decision_time() -> None:
    result = runner.invoke(
        app,
        [
            "production-chain",
            "evaluate-recommendations",
            "--issuance-id",
            "a" * 64,
            "--policy-id",
            "b" * 64,
            "--decision-at",
            "2026-08-18T16:00:00",
        ],
    )
    assert result.exit_code == 2
