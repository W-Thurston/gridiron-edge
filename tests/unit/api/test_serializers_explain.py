# tests/unit/api/test_serializers_explain.py

"""Tests for /games/{game_id}/explain serializer."""

from __future__ import annotations

from gridiron_edge.api.meta import BlockedStatus
from gridiron_edge.api.serializers.explain import serialize_game_explain
from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    sigmoid,
)

FEATURE_NAMES = ("ELO_DIFF", "OFF_PASS_EPA_DIFF")


def _row(
    *,
    win_status: str = "available",
    win_model_type: str | None = "logistic",
    home_win_prob: float | None = 0.62,
    win_event_id: str = "event-1",
) -> dict:
    return {
        "game_id": "2026_02_CAR_ATL",
        "win_status": win_status,
        "win_model_type": win_model_type,
        "home_win_prob": home_win_prob,
        "win_event_id": win_event_id,
    }


def _explanation_event(
    *,
    intercept: float = 0.1,
    transformed_values: tuple[float, ...] = (0.5, -0.25),
    coefficients: tuple[float, ...] = (0.8, -0.4),
) -> LogisticExplanationEvent:
    contributions = tuple(
        LogisticFeatureContribution(
            feature_name=name,
            transformed_value=value,
            coefficient=coefficient,
            contribution=coefficient * value,
        )
        for name, value, coefficient in zip(
            FEATURE_NAMES, transformed_values, coefficients, strict=True
        )
    )
    reconstructed_log_odds = intercept + sum(item.contribution for item in contributions)
    reconstructed_probability = sigmoid(reconstructed_log_odds)
    return LogisticExplanationEvent(
        event_id="event-1",
        game_id="2026_02_CAR_ATL",
        intercept=intercept,
        contributions=contributions,
        reconstructed_log_odds=reconstructed_log_odds,
        reconstructed_probability=reconstructed_probability,
        raw_estimator_output=reconstructed_probability,
        tolerance=1e-9,
    )


def _blocked_slugs(explain) -> dict[str, str]:
    assert explain.response_meta is not None
    return {
        field: status.blocker
        for field, status in explain.response_meta.field_status.items()
        if isinstance(status, BlockedStatus)
    }


class TestScenarioEngineFieldsAlwaysBlocked:
    def test_band_distribution_market_implied(self) -> None:
        explain = serialize_game_explain(_row(), _explanation_event())
        slugs = _blocked_slugs(explain)
        assert slugs["band"] == "scenario_engine"
        assert slugs["distribution"] == "scenario_engine"
        assert slugs["market_implied"] == "scenario_engine"


class TestWinUnavailable:
    def test_headline_and_factors_blocked(self) -> None:
        explain = serialize_game_explain(
            _row(win_status="forecast_missing", home_win_prob=None),
            None,
        )
        slugs = _blocked_slugs(explain)
        assert slugs["headline_win_prob"] == "no_win_forecast"
        assert slugs["factors"] == "no_win_forecast"
        assert explain.headline_win_prob is None
        assert explain.factors is None


class TestNonLogisticChampion:
    def test_factors_blocked_headline_populated(self) -> None:
        explain = serialize_game_explain(
            _row(win_model_type="random_forest"),
            None,
        )
        slugs = _blocked_slugs(explain)
        assert slugs["factors"] == "feature_attribution"
        assert "headline_win_prob" not in slugs
        assert explain.headline_win_prob == 0.62
        assert explain.factors is None


class TestLogisticNoEvidenceYet:
    def test_factors_blocked_headline_populated(self) -> None:
        explain = serialize_game_explain(_row(), None)
        slugs = _blocked_slugs(explain)
        assert slugs["factors"] == "no_explanation_evidence"
        assert explain.headline_win_prob == 0.62
        assert explain.factors is None


class TestLogisticAmbiguousEvidence:
    def test_factors_blocked_headline_populated(self) -> None:
        explain = serialize_game_explain(_row(), None, ambiguous_evidence=True)
        slugs = _blocked_slugs(explain)
        assert slugs["factors"] == "ambiguous_explanation_evidence"
        assert explain.headline_win_prob == 0.62
        assert explain.factors is None


class TestLogisticWithEvidence:
    def test_factors_populated_from_event(self) -> None:
        event = _explanation_event()
        explain = serialize_game_explain(_row(), event)
        slugs = _blocked_slugs(explain)
        assert "factors" not in slugs
        assert explain.headline_win_prob == 0.62
        assert explain.factors is not None
        assert len(explain.factors) == len(FEATURE_NAMES) + 1

        baseline = explain.factors[0]
        assert baseline.key == "intercept"
        assert baseline.is_baseline is True
        assert baseline.log_odds_contribution == event.intercept
        assert baseline.coefficient is None
        assert baseline.transformed_value is None

        for factor, contribution in zip(explain.factors[1:], event.contributions, strict=True):
            assert factor.key == contribution.feature_name
            assert factor.is_baseline is False
            assert factor.log_odds_contribution == contribution.contribution
            assert factor.coefficient == contribution.coefficient
            assert factor.transformed_value == contribution.transformed_value

        reconstructed = baseline.log_odds_contribution + sum(
            factor.log_odds_contribution for factor in explain.factors[1:]
        )
        assert reconstructed == event.reconstructed_log_odds
