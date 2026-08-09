"""Unit tests for recommendation-qualification diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from gridiron_edge.market.qualification import (
    ExactOfferCandidate,
    ForecastProvenance,
    QualificationCheck,
    QualificationCheckName,
    QualificationCheckStatus,
    QualificationResult,
    QualificationState,
    SelectedForecastReference,
    evaluate_qualification,
)

NOW = datetime(2026, 9, 9, 12, tzinfo=UTC)


def candidate() -> ExactOfferCandidate:
    """Build one valid positive-EV exact offer."""
    return ExactOfferCandidate(
        game_id="2026_01_NE_SEA",
        season="2026-2027",
        week=1,
        market="moneyline",
        side="away",
        sportsbook="draftkings",
        american_odds=120,
        line=None,
        model_status="available",
        expected_value=0.043,
        product_id="product",
        product_run_id="product-run",
        market_fetched_at=NOW - timedelta(minutes=10),
        suggested_stake=25.0,
    )


def reference() -> SelectedForecastReference:
    """Build the selected Win forecast identity expected by the product."""
    return SelectedForecastReference(
        event_id="event",
        run_id="run",
        model_name="win_prob",
        model_type="logistic",
    )


def forecast() -> ForecastProvenance:
    """Build matching immutable live forecast provenance."""
    return ForecastProvenance(
        event_id="event",
        run_id="run",
        role="live",
        season="2026-2027",
        week=1,
        game_id="2026_01_NE_SEA",
        model_name="win_prob",
        model_type="logistic",
    )


def complete(
    value: ExactOfferCandidate | None = None,
) -> QualificationResult:
    """Evaluate a candidate with all currently knowable inputs."""
    return evaluate_qualification(
        value or candidate(),
        expected_forecast=reference(),
        forecast=forecast(),
        as_of=NOW,
        max_quote_age=timedelta(hours=1),
        sizing_inputs_available=True,
    )


def get(
    result: QualificationResult,
    name: QualificationCheckName,
) -> QualificationCheck:
    """Return one named check from a qualification result."""
    return next(check for check in result.checks if check.name is name)


def test_positive_candidate_remains_qualification_unavailable() -> None:
    """Future unavailable policies prevent an invented success state."""
    result = complete()

    assert result.state is QualificationState.QUALIFICATION_UNAVAILABLE
    assert (
        get(
            result,
            QualificationCheckName.POSITIVE_EXPECTED_VALUE,
        ).status
        is QualificationCheckStatus.PASSED
    )
    assert (
        get(
            result,
            QualificationCheckName.EMPIRICAL_EDGE_POLICY,
        ).status
        is QualificationCheckStatus.UNAVAILABLE
    )


@pytest.mark.parametrize("expected_value", [-0.01, 0.0])
def test_nonpositive_ev_is_not_candidate(expected_value: float) -> None:
    """Negative and break-even offers are not candidates."""
    result = complete(replace(candidate(), expected_value=expected_value))

    assert result.state is QualificationState.NOT_CANDIDATE


def test_missing_expected_value_is_not_candidate() -> None:
    """An offer without evaluable EV cannot become a candidate."""
    result = complete(replace(candidate(), expected_value=None))

    assert result.state is QualificationState.NOT_CANDIDATE
    assert (
        get(
            result,
            QualificationCheckName.POSITIVE_EXPECTED_VALUE,
        ).status
        is QualificationCheckStatus.UNAVAILABLE
    )


@pytest.mark.parametrize(
    "model_status",
    ["model_unavailable", "uncertainty_unavailable"],
)
def test_unavailable_model_is_not_candidate(model_status: str) -> None:
    """Unavailable model components prevent candidate eligibility."""
    result = complete(replace(candidate(), model_status=model_status))

    assert result.state is QualificationState.NOT_CANDIDATE


@pytest.mark.parametrize(
    "actual",
    [
        replace(forecast(), event_id="other"),
        replace(forecast(), run_id="other"),
        replace(forecast(), model_name="other"),
        replace(forecast(), model_type="other"),
        replace(forecast(), game_id="other"),
        replace(forecast(), season="other"),
        replace(forecast(), week=2),
    ],
    ids=[
        "event-id",
        "run-id",
        "model-name",
        "model-type",
        "game-id",
        "season",
        "week",
    ],
)
def test_forecast_identity_mismatch_is_not_qualified(
    actual: ForecastProvenance,
) -> None:
    """Every selected-event identity field must match exactly."""
    result = evaluate_qualification(
        candidate(),
        expected_forecast=reference(),
        forecast=actual,
    )

    assert result.state is QualificationState.NOT_QUALIFIED
    assert (
        get(
            result,
            QualificationCheckName.SELECTED_FORECAST_PROVENANCE,
        ).status
        is QualificationCheckStatus.FAILED
    )


def test_absent_forecast_is_unavailable_not_fabricated() -> None:
    """Missing immutable event provenance remains explicit."""
    result = evaluate_qualification(candidate())

    assert result.state is QualificationState.QUALIFICATION_UNAVAILABLE
    assert (
        get(
            result,
            QualificationCheckName.LIVE_FORECAST_ROLE,
        ).status
        is QualificationCheckStatus.UNAVAILABLE
    )


def test_backfilled_event_is_not_qualified() -> None:
    """A backfilled event cannot satisfy live forecast provenance."""
    result = evaluate_qualification(
        candidate(),
        expected_forecast=reference(),
        forecast=replace(forecast(), role="backfilled"),
    )

    assert result.state is QualificationState.NOT_QUALIFIED


@pytest.mark.parametrize(
    ("market", "side", "line", "model_name", "model_type"),
    [
        ("moneyline", "away", None, "win_prob", "logistic"),
        ("spread", "away", 3.5, "win_prob", "logistic"),
        ("total", "over", 47.5, "total", "random_forest"),
    ],
)
def test_market_family_forecast_provenance(
    market: str,
    side: str,
    line: float | None,
    model_name: str,
    model_type: str,
) -> None:
    """Moneyline and Spread use Win; Total uses Total provenance."""
    value = replace(
        candidate(),
        market=market,
        side=side,
        line=line,
    )
    expected = replace(
        reference(),
        model_name=model_name,
        model_type=model_type,
    )
    actual = replace(
        forecast(),
        model_name=model_name,
        model_type=model_type,
    )
    result = evaluate_qualification(
        value,
        expected_forecast=expected,
        forecast=actual,
    )

    assert (
        get(
            result,
            QualificationCheckName.SELECTED_FORECAST_PROVENANCE,
        ).status
        is QualificationCheckStatus.PASSED
    )


def test_freshness_at_cutoff_passes_and_stale_fails() -> None:
    """The timestamp cutoff is inclusive and older quotes fail."""
    fresh = evaluate_qualification(
        replace(
            candidate(),
            market_fetched_at=NOW - timedelta(hours=1),
        ),
        as_of=NOW,
        max_quote_age=timedelta(hours=1),
    )
    stale = evaluate_qualification(
        replace(
            candidate(),
            market_fetched_at=NOW - timedelta(hours=2),
        ),
        as_of=NOW,
        max_quote_age=timedelta(hours=1),
    )

    assert (
        get(fresh, QualificationCheckName.QUOTE_FRESHNESS).status is QualificationCheckStatus.PASSED
    )
    assert stale.state is QualificationState.NOT_QUALIFIED


def test_timestamp_without_policy_keeps_freshness_unavailable() -> None:
    """Timestamp evidence does not invent a maximum-age policy."""
    result = evaluate_qualification(candidate())

    assert (
        get(
            result,
            QualificationCheckName.QUOTE_TIMESTAMP_AVAILABLE,
        ).status
        is QualificationCheckStatus.PASSED
    )
    assert (
        get(result, QualificationCheckName.QUOTE_FRESHNESS).status
        is QualificationCheckStatus.UNAVAILABLE
    )


def test_missing_timestamp_is_explicitly_unavailable() -> None:
    """Missing fetch time remains distinct from a fresh quote."""
    result = evaluate_qualification(replace(candidate(), market_fetched_at=None))

    assert (
        get(
            result,
            QualificationCheckName.QUOTE_TIMESTAMP_AVAILABLE,
        ).status
        is QualificationCheckStatus.UNAVAILABLE
    )


def test_freshness_policy_validation() -> None:
    """Freshness inputs must be supplied together and use UTC."""
    with pytest.raises(ValueError, match="provided together"):
        evaluate_qualification(candidate(), as_of=NOW)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        evaluate_qualification(
            candidate(),
            as_of=datetime(2026, 9, 9),
            max_quote_age=timedelta(hours=1),
        )
    with pytest.raises(ValueError, match="must not be negative"):
        evaluate_qualification(
            candidate(),
            as_of=NOW,
            max_quote_age=timedelta(seconds=-1),
        )


def test_sizing_is_informational_and_may_be_unavailable() -> None:
    """Missing stake does not turn a positive-EV offer into a recommendation."""
    result = evaluate_qualification(replace(candidate(), suggested_stake=None))

    assert result.state is QualificationState.QUALIFICATION_UNAVAILABLE
    assert (
        get(
            result,
            QualificationCheckName.SUGGESTED_STAKE_AVAILABLE,
        ).status
        is QualificationCheckStatus.UNAVAILABLE
    )


def test_result_serialization_is_deterministic() -> None:
    """Serialization preserves state and canonical check order."""
    result = complete()
    payload = result.to_dict()

    assert payload["state"] == "qualification_unavailable"
    assert [item["name"] for item in payload["checks"]] == [
        name.value for name in QualificationCheckName
    ]


def test_result_contract_is_frozen() -> None:
    """Qualification results are immutable after construction."""
    result = complete()

    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        result.state = QualificationState.NOT_CANDIDATE
