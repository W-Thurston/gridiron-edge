# src/gridiron_edge/evaluation/forecast_evidence_disposition.py
"""Known-defect disposition for immutable forecast evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import re
from typing import Final

from pandas import DataFrame

from gridiron_edge.evaluation.forecast_store import validate_forecast_events

FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION: Final[int] = 1
_SEASON_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{4})$")
_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_EVENT_COUNT_PER_RUN: Final[int] = 16
_EXPECTED_PRODUCT_ROW_COUNT: Final[int] = 16


class ForecastEvidenceDispositionStatus(StrEnum):
    """Governance classification applied to immutable forecast evidence."""

    KNOWN_DEFECT = "known_defect"


class ForecastEvidenceDefectReason(StrEnum):
    """Confirmed cause of one evidence disposition."""

    INCOMPLETE_ELO_SOURCE_HISTORY = "incomplete_elo_source_history"


class ForecastEvidencePublicationEffect(StrEnum):
    """Operational publication effect of one disposition."""

    NOT_PREDICTION_READY = "not_prediction_ready"


class ForecastEvidenceReplacementPolicy(StrEnum):
    """Replacement and reselection policy for affected evidence."""

    PRESERVE_ORIGINAL_NO_AUTOMATIC_RESELECTION = "preserve_original_no_automatic_reselection"


class AffectedPredictionComponent(StrEnum):
    """Prediction component known to derive from defective evidence."""

    DERIVED_SPREAD = "derived_spread"
    WIN_PROBABILITY = "win_probability"


class ForecastEvidenceNotOperationalError(ValueError):
    """Raised when known-defective evidence is requested for operational use."""


@dataclass(frozen=True, slots=True)
class ForecastEvidenceDisposition:
    """Immutable known-defect classification for exact forecast evidence."""

    schema_version: int
    disposition_id: str
    recorded_at: datetime
    season: str
    week: int
    status: ForecastEvidenceDispositionStatus
    reason: ForecastEvidenceDefectReason
    publication_effect: ForecastEvidencePublicationEffect
    replacement_policy: ForecastEvidenceReplacementPolicy
    affected_components: tuple[AffectedPredictionComponent, ...]
    affected_run_ids: tuple[str, ...]
    affected_event_ids: tuple[str, ...]
    affected_product_ids: tuple[str, ...]
    selected_affected_product_id: str
    decision_references: tuple[str, ...]
    evidence_summary: str


def forecast_evidence_disposition_id(
    *,
    recorded_at: datetime,
    season: str,
    week: int,
    status: ForecastEvidenceDispositionStatus,
    reason: ForecastEvidenceDefectReason,
    publication_effect: ForecastEvidencePublicationEffect,
    replacement_policy: ForecastEvidenceReplacementPolicy,
    affected_components: tuple[AffectedPredictionComponent, ...],
    affected_run_ids: tuple[str, ...],
    affected_event_ids: tuple[str, ...],
    affected_product_ids: tuple[str, ...],
    selected_affected_product_id: str,
    decision_references: tuple[str, ...],
    evidence_summary: str,
) -> str:
    """Return the SHA-256 identity of the complete canonical payload."""
    payload = _identity_payload(
        recorded_at=recorded_at,
        season=season,
        week=week,
        status=status,
        reason=reason,
        publication_effect=publication_effect,
        replacement_policy=replacement_policy,
        affected_components=affected_components,
        affected_run_ids=affected_run_ids,
        affected_event_ids=affected_event_ids,
        affected_product_ids=affected_product_ids,
        selected_affected_product_id=selected_affected_product_id,
        decision_references=decision_references,
        evidence_summary=evidence_summary,
    )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def create_forecast_evidence_disposition(
    *,
    recorded_at: datetime,
    season: str,
    week: int,
    affected_run_ids: tuple[str, ...],
    affected_event_ids: tuple[str, ...],
    affected_product_ids: tuple[str, ...],
    selected_affected_product_id: str,
    decision_references: tuple[str, ...],
    evidence_summary: str,
) -> ForecastEvidenceDisposition:
    """Create the supported schema-1 known-defect disposition."""
    components = (
        AffectedPredictionComponent.DERIVED_SPREAD,
        AffectedPredictionComponent.WIN_PROBABILITY,
    )
    status = ForecastEvidenceDispositionStatus.KNOWN_DEFECT
    reason = ForecastEvidenceDefectReason.INCOMPLETE_ELO_SOURCE_HISTORY
    publication_effect = ForecastEvidencePublicationEffect.NOT_PREDICTION_READY
    replacement_policy = (
        ForecastEvidenceReplacementPolicy.PRESERVE_ORIGINAL_NO_AUTOMATIC_RESELECTION
    )
    disposition_id = forecast_evidence_disposition_id(
        recorded_at=recorded_at,
        season=season,
        week=week,
        status=status,
        reason=reason,
        publication_effect=publication_effect,
        replacement_policy=replacement_policy,
        affected_components=components,
        affected_run_ids=affected_run_ids,
        affected_event_ids=affected_event_ids,
        affected_product_ids=affected_product_ids,
        selected_affected_product_id=selected_affected_product_id,
        decision_references=decision_references,
        evidence_summary=evidence_summary,
    )
    disposition = ForecastEvidenceDisposition(
        schema_version=FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION,
        disposition_id=disposition_id,
        recorded_at=recorded_at,
        season=season,
        week=week,
        status=status,
        reason=reason,
        publication_effect=publication_effect,
        replacement_policy=replacement_policy,
        affected_components=components,
        affected_run_ids=affected_run_ids,
        affected_event_ids=affected_event_ids,
        affected_product_ids=affected_product_ids,
        selected_affected_product_id=selected_affected_product_id,
        decision_references=decision_references,
        evidence_summary=evidence_summary,
    )
    validate_forecast_evidence_disposition(disposition)
    return disposition


def validate_forecast_evidence_disposition(
    disposition: ForecastEvidenceDisposition,
) -> None:
    """Validate one complete disposition and its embedded identity."""
    if disposition.schema_version != FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported forecast-evidence disposition schema_version: "
            f"{disposition.schema_version}."
        )
    _digest(disposition.disposition_id, "disposition_id")
    _utc(disposition.recorded_at, "recorded_at")
    _season(disposition.season)
    _positive_week(disposition.week)
    _enum(disposition.status, ForecastEvidenceDispositionStatus, "status")
    _enum(disposition.reason, ForecastEvidenceDefectReason, "reason")
    _enum(
        disposition.publication_effect,
        ForecastEvidencePublicationEffect,
        "publication_effect",
    )
    _enum(
        disposition.replacement_policy,
        ForecastEvidenceReplacementPolicy,
        "replacement_policy",
    )

    expected_components = (
        AffectedPredictionComponent.DERIVED_SPREAD,
        AffectedPredictionComponent.WIN_PROBABILITY,
    )
    if disposition.affected_components != expected_components:
        raise ValueError(
            "affected_components must contain exactly derived_spread and "
            "win_probability in sorted order."
        )

    _sorted_unique_text(disposition.affected_run_ids, "affected_run_ids")
    _sorted_unique_text(disposition.affected_event_ids, "affected_event_ids")
    _sorted_unique_text(disposition.affected_product_ids, "affected_product_ids")
    _sorted_unique_text(disposition.decision_references, "decision_references")
    selected = _text(
        disposition.selected_affected_product_id,
        "selected_affected_product_id",
    )
    if selected not in disposition.affected_product_ids:
        raise ValueError("selected_affected_product_id must belong to affected_product_ids.")
    _text(disposition.evidence_summary, "evidence_summary")

    expected_id = forecast_evidence_disposition_id(
        recorded_at=disposition.recorded_at,
        season=disposition.season,
        week=disposition.week,
        status=disposition.status,
        reason=disposition.reason,
        publication_effect=disposition.publication_effect,
        replacement_policy=disposition.replacement_policy,
        affected_components=disposition.affected_components,
        affected_run_ids=disposition.affected_run_ids,
        affected_event_ids=disposition.affected_event_ids,
        affected_product_ids=disposition.affected_product_ids,
        selected_affected_product_id=disposition.selected_affected_product_id,
        decision_references=disposition.decision_references,
        evidence_summary=disposition.evidence_summary,
    )
    if disposition.disposition_id != expected_id:
        raise ValueError("disposition_id does not match canonical disposition content.")


def authenticate_forecast_evidence_disposition(
    disposition: ForecastEvidenceDisposition,
    *,
    forecast_events: DataFrame,
    weekly_products: Mapping[str, DataFrame],
    selected_product_id: str,
) -> None:
    """Authenticate every disposition reference against immutable evidence."""
    validate_forecast_evidence_disposition(disposition)
    events = validate_forecast_events(forecast_events)
    _authenticate_events(disposition, events)
    _authenticate_products(disposition, events, weekly_products)
    if _text(selected_product_id, "selected_product_id") != (
        disposition.selected_affected_product_id
    ):
        raise ValueError("Selected weekly product does not match selected_affected_product_id.")


def disposition_applies_to_product(
    disposition: ForecastEvidenceDisposition,
    product: DataFrame,
) -> bool:
    """Return whether one validated disposition references the product."""
    validate_forecast_evidence_disposition(disposition)
    product_id = _single_product_id(product)
    return product_id in disposition.affected_product_ids


def require_operational_weekly_product(
    product: DataFrame,
    dispositions: Sequence[ForecastEvidenceDisposition],
) -> None:
    """Reject operational use of one known-defective weekly product."""
    product_id = _single_product_id(product)
    applicable = tuple(
        disposition
        for disposition in dispositions
        if disposition_applies_to_product(disposition, product)
    )
    if not applicable:
        return
    if len(applicable) > 1:
        raise ValueError(
            f"Multiple forecast-evidence dispositions apply to product {product_id!r}."
        )
    disposition = applicable[0]
    if (
        disposition.status is ForecastEvidenceDispositionStatus.KNOWN_DEFECT
        and disposition.publication_effect is ForecastEvidencePublicationEffect.NOT_PREDICTION_READY
    ):
        raise ForecastEvidenceNotOperationalError(
            "Weekly product is not operational: "
            f"product_id={product_id!r}, "
            f"disposition_id={disposition.disposition_id!r}, "
            f"status={disposition.status.value!r}, "
            f"reason={disposition.reason.value!r}."
        )


def _identity_payload(
    *,
    recorded_at: datetime,
    season: str,
    week: int,
    status: ForecastEvidenceDispositionStatus,
    reason: ForecastEvidenceDefectReason,
    publication_effect: ForecastEvidencePublicationEffect,
    replacement_policy: ForecastEvidenceReplacementPolicy,
    affected_components: tuple[AffectedPredictionComponent, ...],
    affected_run_ids: tuple[str, ...],
    affected_event_ids: tuple[str, ...],
    affected_product_ids: tuple[str, ...],
    selected_affected_product_id: str,
    decision_references: tuple[str, ...],
    evidence_summary: str,
) -> dict[str, object]:
    _utc(recorded_at, "recorded_at")
    _season(season)
    _positive_week(week)
    _enum(status, ForecastEvidenceDispositionStatus, "status")
    _enum(reason, ForecastEvidenceDefectReason, "reason")
    _enum(publication_effect, ForecastEvidencePublicationEffect, "publication_effect")
    _enum(replacement_policy, ForecastEvidenceReplacementPolicy, "replacement_policy")
    return {
        "schema_version": FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION,
        "recorded_at": recorded_at.isoformat(),
        "season": season,
        "week": week,
        "status": status.value,
        "reason": reason.value,
        "publication_effect": publication_effect.value,
        "replacement_policy": replacement_policy.value,
        "affected_components": [value.value for value in affected_components],
        "affected_run_ids": list(affected_run_ids),
        "affected_event_ids": list(affected_event_ids),
        "affected_product_ids": list(affected_product_ids),
        "selected_affected_product_id": selected_affected_product_id,
        "decision_references": list(decision_references),
        "evidence_summary": evidence_summary,
    }


def _authenticate_events(
    disposition: ForecastEvidenceDisposition,
    events: DataFrame,
) -> None:
    if events["event_id"].duplicated().any():
        raise ValueError("Forecast evidence contains duplicate event IDs.")
    affected = events.loc[
        events["event_id"].astype(str).isin(disposition.affected_event_ids), :
    ].copy()
    actual_ids = tuple(sorted(affected["event_id"].astype(str).tolist()))
    if actual_ids != disposition.affected_event_ids:
        missing = sorted(set(disposition.affected_event_ids) - set(actual_ids))
        raise ValueError("Affected forecast events are missing: " + ", ".join(missing))

    in_scope = events.loc[
        (events["season"].astype(str) == disposition.season)
        & (events["week"].astype(int) == disposition.week)
        & (events["role"].astype(str) == "live")
        & (events["model_name"].astype(str) == "win_prob")
        & (events["model_type"].astype(str) == "logistic"),
        :,
    ]
    scoped_ids = tuple(sorted(in_scope["event_id"].astype(str).tolist()))
    if scoped_ids != disposition.affected_event_ids:
        raise ValueError("Disposition must include every live logistic Win event in its scope.")

    if not affected["season"].astype(str).eq(disposition.season).all():
        raise ValueError("Affected forecast event season does not match disposition.")
    if not affected["week"].astype(int).eq(disposition.week).all():
        raise ValueError("Affected forecast event week does not match disposition.")
    if not affected["role"].astype(str).eq("live").all():
        raise ValueError("Affected forecast events must have live role.")
    if not affected["model_name"].astype(str).eq("win_prob").all():
        raise ValueError("Affected forecast events must use model_name win_prob.")
    if not affected["model_type"].astype(str).eq("logistic").all():
        raise ValueError("Affected forecast events must use model_type logistic.")

    actual_runs = tuple(sorted(affected["run_id"].astype(str).unique().tolist()))
    if actual_runs != disposition.affected_run_ids:
        raise ValueError("Affected event run IDs do not match affected_run_ids.")

    for run_id in disposition.affected_run_ids:
        run = affected.loc[affected["run_id"].astype(str) == run_id, :]
        if len(run) != _EXPECTED_EVENT_COUNT_PER_RUN:
            raise ValueError(
                f"Affected forecast run {run_id!r} must contain exactly "
                f"{_EXPECTED_EVENT_COUNT_PER_RUN} events."
            )
        if run["game_id"].astype(str).nunique() != _EXPECTED_EVENT_COUNT_PER_RUN:
            raise ValueError(
                f"Affected forecast run {run_id!r} must contain exactly "
                f"{_EXPECTED_EVENT_COUNT_PER_RUN} unique game IDs."
            )


def _authenticate_products(
    disposition: ForecastEvidenceDisposition,
    events: DataFrame,
    weekly_products: Mapping[str, DataFrame],
) -> None:
    supplied_ids = tuple(sorted(weekly_products))
    if supplied_ids != disposition.affected_product_ids:
        missing = sorted(set(disposition.affected_product_ids) - set(supplied_ids))
        unexpected = sorted(set(supplied_ids) - set(disposition.affected_product_ids))
        raise ValueError(
            "Supplied affected weekly products do not match disposition; "
            f"missing={missing}, unexpected={unexpected}."
        )

    event_ids_by_run = {
        run_id: tuple(
            sorted(
                events.loc[events["run_id"].astype(str) == run_id, "event_id"].astype(str).tolist()
            )
        )
        for run_id in disposition.affected_run_ids
    }

    for product_id in disposition.affected_product_ids:
        product = weekly_products[product_id]
        if len(product) != _EXPECTED_PRODUCT_ROW_COUNT:
            raise ValueError(
                f"Affected weekly product {product_id!r} must contain exactly "
                f"{_EXPECTED_PRODUCT_ROW_COUNT} rows."
            )
        actual_product_id = _single_text_column(product, "product_id")
        if actual_product_id != product_id:
            raise ValueError("Weekly product mapping key and embedded product_id disagree.")
        if not product["season"].astype(str).eq(disposition.season).all():
            raise ValueError("Affected weekly product season does not match disposition.")
        if not product["week"].astype(int).eq(disposition.week).all():
            raise ValueError("Affected weekly product week does not match disposition.")

        product_run_id = _single_text_column(product, "product_run_id")
        if product_run_id not in disposition.affected_run_ids:
            raise ValueError("Affected weekly product run is not an affected run.")
        win_run_ids = tuple(sorted(product["win_run_id"].dropna().astype(str).unique()))
        if win_run_ids != (product_run_id,):
            raise ValueError("Weekly product Win run IDs must equal product_run_id.")
        win_event_ids = tuple(sorted(product["win_event_id"].dropna().astype(str).tolist()))
        if win_event_ids != event_ids_by_run[product_run_id]:
            raise ValueError("Weekly product Win event IDs do not match affected run events.")

        available_spreads = product.loc[product["spread_status"].astype(str) == "available", :]
        if available_spreads["spread_source_event_id"].isna().any():
            raise ValueError("Available derived spreads require source event IDs.")
        if not (
            available_spreads["spread_source_event_id"].astype(str)
            == available_spreads["win_event_id"].astype(str)
        ).all():
            raise ValueError("Derived spread source event must equal the row Win event.")

        total_columns = (
            "total_event_id",
            "total_run_id",
            "total_model_name",
            "total_model_type",
            "total_role",
        )
        _require_columns(product, total_columns, label="Affected weekly product")
        if product.loc[:, list(total_columns)].notna().any(axis=None):
            raise ValueError("Affected weekly products must not claim Total evidence.")


def _single_product_id(product: DataFrame) -> str:
    if product.empty:
        raise ValueError("Weekly product must not be empty.")
    return _single_text_column(product, "product_id")


def _single_text_column(frame: DataFrame, column: str) -> str:
    _require_columns(frame, (column,), label="Weekly product")
    values = tuple(sorted(frame[column].dropna().astype(str).str.strip().unique()))
    if len(values) != 1 or not values[0]:
        raise ValueError(f"Weekly product must contain one nonempty {column} value.")
    return values[0]


def _require_columns(frame: DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _sorted_unique_text(values: tuple[str, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty.")
    normalized = tuple(_text(value, label) for value in values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{label} must contain sorted unique nonempty values.")


def _season(value: str) -> str:
    text = _text(value, "season")
    match = _SEASON_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("season must use canonical YYYY-YYYY format.")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end != start + 1:
        raise ValueError("season ending year must be one greater than starting year.")
    return text


def _positive_week(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("week must be an integer.")
    if value < 1:
        raise ValueError("week must be at least 1.")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware UTC.")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError(f"{label} must use UTC.")
    return value


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _enum(value: object, expected: type[StrEnum], label: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be a {expected.__name__}.")
