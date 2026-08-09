"""Recommendation-qualification diagnostics for exact sportsbook offers.

The module records what is known, failed, or unavailable for one evaluated
sportsbook offer. It intentionally does not emit a qualified or recommended
state while empirical reliability, edge, freshness, and exposure policies are
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from typing import Any, Final


class QualificationCheckStatus(StrEnum):
    """Resolution status of one recommendation-qualification check."""

    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class QualificationState(StrEnum):
    """Aggregate diagnostic state of one exact sportsbook offer."""

    NOT_CANDIDATE = "not_candidate"
    NOT_QUALIFIED = "not_qualified"
    QUALIFICATION_UNAVAILABLE = "qualification_unavailable"


class QualificationCheckName(StrEnum):
    """Canonical checks evaluated for every exact sportsbook offer."""

    POSITIVE_EXPECTED_VALUE = "positive_expected_value"
    MODEL_COMPONENT_AVAILABLE = "model_component_available"
    SELECTED_PRODUCT_PROVENANCE = "selected_product_provenance"
    QUOTE_IDENTITY_AVAILABLE = "quote_identity_available"
    SELECTED_FORECAST_PROVENANCE = "selected_forecast_provenance"
    LIVE_FORECAST_ROLE = "live_forecast_role"
    QUOTE_TIMESTAMP_AVAILABLE = "quote_timestamp_available"
    QUOTE_FRESHNESS = "quote_freshness"
    SIZING_INPUTS_AVAILABLE = "sizing_inputs_available"
    SUGGESTED_STAKE_AVAILABLE = "suggested_stake_available"
    EMPIRICAL_EDGE_POLICY = "empirical_edge_policy"
    MODEL_RELIABILITY_POLICY = "model_reliability_policy"
    DUPLICATE_EXPOSURE_POLICY = "duplicate_exposure_policy"
    CONFLICTING_EXPOSURE_POLICY = "conflicting_exposure_policy"
    PORTFOLIO_CONCENTRATION_POLICY = "portfolio_concentration_policy"
    CORRELATION_POLICY = "correlation_policy"


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    """One explicit recommendation-qualification finding."""

    name: QualificationCheckName
    status: QualificationCheckStatus
    reason: str
    detail: str

    def __post_init__(self) -> None:
        """Validate nonempty reason and explanatory detail."""
        if not self.reason.strip() or not self.detail.strip():
            raise ValueError("Qualification reason and detail must not be empty.")

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "name": self.name.value,
            "status": self.status.value,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ExactOfferCandidate:
    """Exact evaluated sportsbook offer submitted for qualification."""

    game_id: str | None
    season: str | None
    week: int | None
    market: str | None
    side: str | None
    sportsbook: str | None
    american_odds: int | None
    line: float | None
    model_status: str
    expected_value: float | None
    product_id: str | None
    product_run_id: str | None
    provider: str | None = None
    provider_event_id: str | None = None
    market_fetched_at: datetime | None = None
    sportsbook_updated_at: datetime | None = None
    model_probability: float | None = None
    kelly_fraction: float | None = None
    suggested_stake: float | None = None


@dataclass(frozen=True, slots=True)
class SelectedForecastReference:
    """Forecast identity selected by the owning weekly product."""

    event_id: str
    run_id: str
    model_name: str
    model_type: str


@dataclass(frozen=True, slots=True)
class ForecastProvenance:
    """Immutable forecast-event provenance supplied for verification."""

    event_id: str
    run_id: str
    role: str
    season: str
    week: int
    game_id: str
    model_name: str
    model_type: str


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Aggregate state and ordered checks for one exact offer."""

    state: QualificationState
    checks: tuple[QualificationCheck, ...]

    def __post_init__(self) -> None:
        """Require every canonical check exactly once and in order."""
        actual = tuple(check.name for check in self.checks)
        expected = tuple(QualificationCheckName)
        if actual != expected:
            raise ValueError("Qualification checks must appear exactly once in canonical order.")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "state": self.state.value,
            "checks": [check.to_dict() for check in self.checks],
        }


_FUTURE_POLICY_DETAILS: Final[tuple[tuple[QualificationCheckName, str, str], ...]] = (
    (
        QualificationCheckName.EMPIRICAL_EDGE_POLICY,
        "empirical_edge_policy_unavailable",
        "No empirically validated minimum-edge policy is available.",
    ),
    (
        QualificationCheckName.MODEL_RELIABILITY_POLICY,
        "model_reliability_policy_unavailable",
        "No market-specific model-reliability policy is available.",
    ),
    (
        QualificationCheckName.DUPLICATE_EXPOSURE_POLICY,
        "duplicate_exposure_policy_unavailable",
        "Duplicate economic exposure is not evaluated.",
    ),
    (
        QualificationCheckName.CONFLICTING_EXPOSURE_POLICY,
        "conflicting_exposure_policy_unavailable",
        "Conflicting exposure is not evaluated.",
    ),
    (
        QualificationCheckName.PORTFOLIO_CONCENTRATION_POLICY,
        "portfolio_concentration_policy_unavailable",
        "Portfolio concentration is not evaluated.",
    ),
    (
        QualificationCheckName.CORRELATION_POLICY,
        "correlation_policy_unavailable",
        "Wager correlation is not evaluated.",
    ),
)


def _check(
    name: QualificationCheckName,
    status: QualificationCheckStatus,
    reason: str,
    detail: str,
) -> QualificationCheck:
    """Build one qualification check."""
    return QualificationCheck(name, status, reason, detail)


def _resolved_check(
    *,
    name: QualificationCheckName,
    passed: bool,
    passed_reason: str,
    failed_reason: str,
    passed_detail: str,
    failed_detail: str,
) -> QualificationCheck:
    """Build a deterministic passed or failed qualification check."""
    if passed:
        return _check(
            name,
            QualificationCheckStatus.PASSED,
            passed_reason,
            passed_detail,
        )
    return _check(
        name,
        QualificationCheckStatus.FAILED,
        failed_reason,
        failed_detail,
    )


def _expected_value_check(
    candidate: ExactOfferCandidate,
) -> QualificationCheck:
    """Evaluate whether the exact offer has strictly positive EV."""
    value = candidate.expected_value
    if value is None or not isfinite(value):
        return _check(
            QualificationCheckName.POSITIVE_EXPECTED_VALUE,
            QualificationCheckStatus.UNAVAILABLE,
            "expected_value_unavailable",
            "Expected value is unavailable for this exact offer.",
        )
    return _resolved_check(
        name=QualificationCheckName.POSITIVE_EXPECTED_VALUE,
        passed=value > 0.0,
        passed_reason="positive_expected_value",
        failed_reason="expected_value_not_positive",
        passed_detail=("This exact offer has strictly positive expected value."),
        failed_detail=("This exact offer does not have strictly positive expected value."),
    )


def _model_component_check(
    candidate: ExactOfferCandidate,
) -> QualificationCheck:
    """Evaluate whether the required model component is available."""
    available = candidate.model_status == "available"
    return _resolved_check(
        name=QualificationCheckName.MODEL_COMPONENT_AVAILABLE,
        passed=available,
        passed_reason="model_component_available",
        failed_reason="model_component_unavailable",
        passed_detail="The required model component is available.",
        failed_detail=("The required model component or uncertainty is unavailable."),
    )


def _selected_product_check(
    candidate: ExactOfferCandidate,
) -> QualificationCheck:
    """Evaluate selected weekly-product provenance."""
    identified = _nonempty(candidate.product_id) and _nonempty(candidate.product_run_id)
    return _resolved_check(
        name=QualificationCheckName.SELECTED_PRODUCT_PROVENANCE,
        passed=identified,
        passed_reason="selected_product_identified",
        failed_reason="selected_product_provenance_missing",
        passed_detail="The selected weekly product and run are identified.",
        failed_detail="Selected weekly-product provenance is incomplete.",
    )


def _quote_identity_check(
    candidate: ExactOfferCandidate,
) -> QualificationCheck:
    """Evaluate the exact sportsbook-offer identity."""
    market = candidate.market
    side = candidate.side
    market_side_valid = (
        (market == "moneyline" and side in {"home", "away"})
        or (market == "spread" and side in {"home", "away"})
        or (market == "total" and side in {"over", "under"})
    )
    line_valid = (market == "moneyline" and candidate.line is None) or (
        market in {"spread", "total"} and candidate.line is not None and isfinite(candidate.line)
    )
    identity_available = all(
        (
            _nonempty(candidate.game_id),
            _nonempty(candidate.season),
            candidate.week is not None and candidate.week >= 1,
            market_side_valid,
            _nonempty(candidate.sportsbook),
            candidate.american_odds not in (None, 0),
            line_valid,
        )
    )
    return _resolved_check(
        name=QualificationCheckName.QUOTE_IDENTITY_AVAILABLE,
        passed=identity_available,
        passed_reason="quote_identity_available",
        failed_reason="quote_identity_incomplete",
        passed_detail="The exact offer identity is available.",
        failed_detail="The exact offer identity is incomplete or invalid.",
    )


def _forecast_checks(
    candidate: ExactOfferCandidate,
    *,
    expected: SelectedForecastReference | None,
    actual: ForecastProvenance | None,
) -> tuple[QualificationCheck, QualificationCheck]:
    """Evaluate exact forecast identity and live-role provenance."""
    if expected is None or actual is None:
        provenance = _check(
            QualificationCheckName.SELECTED_FORECAST_PROVENANCE,
            QualificationCheckStatus.UNAVAILABLE,
            "selected_forecast_provenance_unavailable",
            "Exact selected forecast-event provenance was not supplied.",
        )
        role = _check(
            QualificationCheckName.LIVE_FORECAST_ROLE,
            QualificationCheckStatus.UNAVAILABLE,
            "forecast_role_unavailable",
            "The forecast event role cannot be verified.",
        )
        return provenance, role

    expected_identity = (
        expected.event_id,
        expected.run_id,
        expected.model_name,
        expected.model_type,
        candidate.game_id,
        candidate.season,
        candidate.week,
    )
    actual_identity = (
        actual.event_id,
        actual.run_id,
        actual.model_name,
        actual.model_type,
        actual.game_id,
        actual.season,
        actual.week,
    )
    provenance = _resolved_check(
        name=QualificationCheckName.SELECTED_FORECAST_PROVENANCE,
        passed=expected_identity == actual_identity,
        passed_reason="selected_forecast_provenance_matched",
        failed_reason="selected_forecast_provenance_mismatch",
        passed_detail=("The forecast event matches the selected product and offer scope."),
        failed_detail=("The forecast event does not match the selected product and offer scope."),
    )
    role = _resolved_check(
        name=QualificationCheckName.LIVE_FORECAST_ROLE,
        passed=actual.role == "live",
        passed_reason="live_forecast_role",
        failed_reason="forecast_role_not_live",
        passed_detail="The selected forecast event has the live role.",
        failed_detail=("The selected forecast event does not have the live role."),
    )
    return provenance, role


def _quote_time_checks(
    candidate: ExactOfferCandidate,
    *,
    as_of: datetime | None,
    max_quote_age: timedelta | None,
) -> tuple[QualificationCheck, QualificationCheck]:
    """Evaluate quote timestamp evidence and optional freshness policy."""
    timestamp = candidate.market_fetched_at
    if timestamp is None or not _is_utc(timestamp):
        available = _check(
            QualificationCheckName.QUOTE_TIMESTAMP_AVAILABLE,
            QualificationCheckStatus.UNAVAILABLE,
            "quote_timestamp_unavailable",
            "The exact offer has no valid UTC market fetch timestamp.",
        )
        freshness = _check(
            QualificationCheckName.QUOTE_FRESHNESS,
            QualificationCheckStatus.UNAVAILABLE,
            "quote_freshness_unavailable",
            "Quote freshness cannot be evaluated.",
        )
        return available, freshness

    available = _check(
        QualificationCheckName.QUOTE_TIMESTAMP_AVAILABLE,
        QualificationCheckStatus.PASSED,
        "quote_timestamp_available",
        "The exact offer has a valid UTC market fetch timestamp.",
    )
    if as_of is None or max_quote_age is None:
        freshness = _check(
            QualificationCheckName.QUOTE_FRESHNESS,
            QualificationCheckStatus.UNAVAILABLE,
            "quote_freshness_policy_unavailable",
            "No maximum quote-age policy was supplied.",
        )
        return available, freshness

    fresh = timestamp >= as_of - max_quote_age
    freshness = _resolved_check(
        name=QualificationCheckName.QUOTE_FRESHNESS,
        passed=fresh,
        passed_reason="quote_fresh",
        failed_reason="quote_stale",
        passed_detail=("The exact offer is within the supplied quote-age policy."),
        failed_detail=("The exact offer is older than the supplied quote-age policy."),
    )
    return available, freshness


def _sizing_checks(
    candidate: ExactOfferCandidate,
    *,
    sizing_inputs_available: bool,
) -> tuple[QualificationCheck, QualificationCheck]:
    """Evaluate sizing-input and suggested-stake availability."""
    sizing = _check(
        QualificationCheckName.SIZING_INPUTS_AVAILABLE,
        (
            QualificationCheckStatus.PASSED
            if sizing_inputs_available
            else QualificationCheckStatus.UNAVAILABLE
        ),
        ("sizing_inputs_available" if sizing_inputs_available else "sizing_inputs_unavailable"),
        (
            "Bankroll and fractional-Kelly inputs are available."
            if sizing_inputs_available
            else "Bankroll or fractional-Kelly inputs are unavailable."
        ),
    )
    stake_value = candidate.suggested_stake
    stake_available = stake_value is not None and isfinite(stake_value) and stake_value >= 0.0
    stake = _check(
        QualificationCheckName.SUGGESTED_STAKE_AVAILABLE,
        (
            QualificationCheckStatus.PASSED
            if stake_available
            else QualificationCheckStatus.UNAVAILABLE
        ),
        ("suggested_stake_available" if stake_available else "suggested_stake_unavailable"),
        (
            "A nonnegative fractional-Kelly stake is available."
            if stake_available
            else "A fractional-Kelly stake is unavailable."
        ),
    )
    return sizing, stake


def _future_policy_checks() -> tuple[QualificationCheck, ...]:
    """Return explicit unavailable checks for policies not yet owned."""
    return tuple(
        _check(
            name,
            QualificationCheckStatus.UNAVAILABLE,
            reason,
            detail,
        )
        for name, reason, detail in _FUTURE_POLICY_DETAILS
    )


def _qualification_state(
    checks: tuple[QualificationCheck, ...],
) -> QualificationState:
    """Aggregate ordered checks without inventing a success state."""
    by_name = {check.name: check for check in checks}
    candidate_checks = (
        by_name[QualificationCheckName.POSITIVE_EXPECTED_VALUE],
        by_name[QualificationCheckName.MODEL_COMPONENT_AVAILABLE],
    )
    if any(check.status is not QualificationCheckStatus.PASSED for check in candidate_checks):
        return QualificationState.NOT_CANDIDATE
    if any(check.status is QualificationCheckStatus.FAILED for check in checks):
        return QualificationState.NOT_QUALIFIED
    return QualificationState.QUALIFICATION_UNAVAILABLE


def _validate_freshness_policy(
    *,
    as_of: datetime | None,
    max_quote_age: timedelta | None,
) -> None:
    """Validate an optional deterministic quote-freshness policy."""
    if (as_of is None) != (max_quote_age is None):
        raise ValueError("as_of and max_quote_age must be provided together.")
    if as_of is None or max_quote_age is None:
        return
    if not _is_utc(as_of):
        raise ValueError("as_of must be timezone-aware UTC.")
    if max_quote_age < timedelta(0):
        raise ValueError("max_quote_age must not be negative.")


def _is_utc(value: datetime) -> bool:
    """Return whether a datetime is timezone-aware UTC."""
    offset = value.utcoffset()
    return value.tzinfo is not None and offset == timedelta(0)


def _nonempty(value: str | None) -> bool:
    """Return whether a nullable text value contains non-whitespace text."""
    return value is not None and bool(value.strip())


def evaluate_qualification(
    candidate: ExactOfferCandidate,
    *,
    expected_forecast: SelectedForecastReference | None = None,
    forecast: ForecastProvenance | None = None,
    as_of: datetime | None = None,
    max_quote_age: timedelta | None = None,
    sizing_inputs_available: bool = False,
) -> QualificationResult:
    """Evaluate known and unavailable qualification checks for one offer."""
    _validate_freshness_policy(
        as_of=as_of,
        max_quote_age=max_quote_age,
    )
    forecast_check, role_check = _forecast_checks(
        candidate,
        expected=expected_forecast,
        actual=forecast,
    )
    timestamp_check, freshness_check = _quote_time_checks(
        candidate,
        as_of=as_of,
        max_quote_age=max_quote_age,
    )
    sizing_check, stake_check = _sizing_checks(
        candidate,
        sizing_inputs_available=sizing_inputs_available,
    )
    checks = (
        _expected_value_check(candidate),
        _model_component_check(candidate),
        _selected_product_check(candidate),
        _quote_identity_check(candidate),
        forecast_check,
        role_check,
        timestamp_check,
        freshness_check,
        sizing_check,
        stake_check,
        *_future_policy_checks(),
    )
    return QualificationResult(
        state=_qualification_state(checks),
        checks=checks,
    )
