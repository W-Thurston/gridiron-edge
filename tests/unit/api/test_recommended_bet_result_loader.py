"""Tests for persisted weekly recommendation loading and selection."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from tests.fixtures.recommended_bet_results import (
    DECISION,
    EVALUATED,
    bankroll,
    evaluation,
    issuance,
    policy,
    row,
)

from gridiron_edge.api.loaders import load_recommended_bet_results_for_week
from gridiron_edge.core.settings import Settings
from gridiron_edge.market.recommendation_policy import portfolio_exposure_snapshot
from gridiron_edge.market.recommended_bet_result import (
    evaluate_recommendation_issuance,
)
from gridiron_edge.market.recommended_bet_result_store import (
    recommended_bet_offer_key,
    write_recommended_bet_evaluation,
)


def _settings(repo: Path) -> Settings:
    return Settings(
        owm_api_key="",
        odds_api_key="",
        repo_root=repo,
        data_raw=repo / "data" / "raw",
        data_cleaned=repo / "data" / "cleaned",
        data_modeling=repo / "data" / "modeling",
        data_output=repo / "data" / "output",
    )


def test_missing_store_returns_empty_tuple(tmp_path: Path) -> None:
    assert load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1) == ()


def test_loader_reads_strict_manifests_and_filters_scope(tmp_path: Path) -> None:
    value = evaluation()
    write_recommended_bet_evaluation(value, repo=tmp_path)
    loaded = load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1)
    assert len(loaded) == 1
    assert loaded[0].evaluation_id == value.evaluation_id
    assert loaded[0].result == value.results[0]
    assert recommended_bet_offer_key(loaded[0].result) == recommended_bet_offer_key(
        value.results[0]
    )
    assert load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=2) == ()


def test_loader_selects_latest_explicit_evaluation_time(tmp_path: Path) -> None:
    first = evaluation()
    later = evaluate_recommendation_issuance(
        policy=policy(),
        issuance=issuance(row()),
        decision_at=DECISION + timedelta(minutes=1),
        bankroll=bankroll(),
        portfolio=portfolio_exposure_snapshot(observed_at=EVALUATED, rows=()),
    )
    write_recommended_bet_evaluation(first, repo=tmp_path)
    write_recommended_bet_evaluation(later, repo=tmp_path)
    loaded = load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1)
    assert len(loaded) == 1
    assert loaded[0].evaluation_id == later.evaluation_id
    assert loaded[0].result.result_id == later.results[0].result_id


def test_equal_time_different_results_are_rejected(tmp_path: Path) -> None:
    active = evaluation(active=True)
    inactive = evaluation(active=False)
    assert active.evaluated_at == inactive.evaluated_at
    assert active.results[0].result_id != inactive.results[0].result_id
    write_recommended_bet_evaluation(active, repo=tmp_path)
    write_recommended_bet_evaluation(inactive, repo=tmp_path)
    with pytest.raises(ValueError, match="Conflicting persisted recommendation"):
        load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1)


def test_loader_order_is_deterministic_by_exact_offer(tmp_path: Path) -> None:
    value = evaluate_recommendation_issuance(
        policy=policy(),
        issuance=issuance(
            row(game_id="2026_01_KC_LAC", sportsbook="draftkings"),
            row(game_id="2026_01_BUF_NYJ", sportsbook="fanduel"),
        ),
        decision_at=DECISION,
        bankroll=bankroll(),
        portfolio=portfolio_exposure_snapshot(observed_at=EVALUATED, rows=()),
    )
    write_recommended_bet_evaluation(value, repo=tmp_path)
    first = load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1)
    second = load_recommended_bet_results_for_week(_settings(tmp_path), season="2026", week=1)
    assert first == second
    assert [item.result.game_id for item in first] == sorted(item.result.game_id for item in first)


def test_loader_has_no_policy_evaluation_or_mutable_dependency() -> None:
    source = Path("src/gridiron_edge/api/loaders.py").read_text(encoding="utf-8")
    function_source = source[source.index("def load_recommended_bet_results_for_week(") :]
    for forbidden in (
        "evaluate_recommendation_candidate",
        "evaluate_recommendation_issuance",
        "gridiron_edge.market.kelly",
        "gridiron_edge.betting.bankroll",
        "gridiron_edge.betting.ledger",
    ):
        assert forbidden not in function_source
