# src/gridiron_edge/market/candidate_issuance.py

"""Pure immutable pregame candidate issuance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Final, cast

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_store import validate_forecast_events
from gridiron_edge.ingest.odds.store import (
    OBSERVATION_IDENTITY_COLUMNS,
    OBSERVATION_SORT_COLUMNS,
    validate_quote_rows,
)
from gridiron_edge.market.line_shopping import evaluate_line_shopping_guidance
from gridiron_edge.models.game_prediction.product_validation import (
    validate_weekly_game_product,
)

CANDIDATE_ISSUANCE_SCHEMA_VERSION: Final[int] = 1
_PRODUCT_STORAGE_COLUMNS: Final[tuple[str, ...]] = (
    "product_schema_version",
    "product_id",
    "product_run_id",
    "product_generated_at",
)
CANDIDATE_REFERENCE_DERIVATION_VERSION_V1: Final[int] = 1
CURRENT_CANDIDATE_REFERENCE_DERIVATION_VERSION: Final[int] = (
    CANDIDATE_REFERENCE_DERIVATION_VERSION_V1
)


class UnsupportedCandidateReferenceVersionError(ValueError):
    """Raised when a candidate-reference derivation version is unsupported.

    Covers both an invalid version value and a version with no known
    implementation to derive or re-derive against.
    """


class CandidateIssuanceState(StrEnum):
    """Historical outcome of evaluating one exact quote observation."""

    CANDIDATE = "candidate"
    NOT_CANDIDATE = "not_candidate"
    UNAVAILABLE = "unavailable"


class CandidateIssuanceReason(StrEnum):
    """Evidence-only reason for one issuance state."""

    POSITIVE_EXPECTED_VALUE = "positive_expected_value"
    EXPECTED_VALUE_NOT_POSITIVE = "expected_value_not_positive"
    KICKOFF_UNAVAILABLE = "kickoff_unavailable"
    QUOTE_UNAVAILABLE = "quote_unavailable"
    QUOTE_LIVE = "quote_live"
    QUOTE_NOT_PREGAME = "quote_not_pregame"
    FORECAST_EVENT_UNAVAILABLE = "forecast_event_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNCERTAINTY_UNAVAILABLE = "uncertainty_unavailable"
    MODEL_PROBABILITY_UNAVAILABLE = "model_probability_unavailable"
    EXPECTED_VALUE_UNAVAILABLE = "expected_value_unavailable"


@dataclass(frozen=True, slots=True)
class CandidateIssuanceRow:
    """Immutable evidence for one evaluated quote observation."""

    game_id: str
    market: str
    side: str
    provider: str
    provider_event_id: str | None
    sportsbook: str | None
    line: float | None
    american_price: int | None
    fetched_at: datetime
    sportsbook_updated_at: datetime | None
    kickoff: datetime | None
    is_live: bool
    forecast_event_id: str | None
    forecast_run_id: str | None
    forecast_role: str | None
    forecast_generated_at: datetime | None
    model_name: str | None
    model_type: str | None
    model_probability: float | None
    expected_value: float | None
    state: CandidateIssuanceState
    reason: CandidateIssuanceReason


@dataclass(frozen=True, slots=True)
class CandidateIssuance:
    """One deterministic immutable pregame evaluation invocation."""

    schema_version: int
    issuance_id: str
    product_id: str
    product_run_id: str
    product_generated_at: datetime
    season: str
    week: int
    evaluated_at: datetime
    rows: tuple[CandidateIssuanceRow, ...]


def candidate_issuance_id(
    *,
    product_id: str,
    product_run_id: str,
    season: str,
    week: int,
    evaluated_at: datetime,
) -> str:
    """Return the deterministic identity for one evaluation invocation."""
    evaluated = _require_utc(evaluated_at, label="evaluated_at")
    payload = {
        "schema_version": CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        "product_id": _nonempty(product_id, label="product_id"),
        "product_run_id": _nonempty(product_run_id, label="product_run_id"),
        "season": _nonempty(season, label="season"),
        "week": _positive_week(week),
        "evaluated_at": evaluated.isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def candidate_issuance_row_id(
    issuance_id: str,
    row: CandidateIssuanceRow,
    *,
    version: int = CURRENT_CANDIDATE_REFERENCE_DERIVATION_VERSION,
) -> str:
    """Return one exact row identity under a selected derivation version.

    The version selects an implementation and is not automatically
    included in that implementation's payload. Explicit v1 derivation
    therefore preserves the pre-versioning candidate-reference output
    exactly.
    """
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise UnsupportedCandidateReferenceVersionError(
            f"Candidate reference derivation version {version!r} is invalid."
        )
    if version == CANDIDATE_REFERENCE_DERIVATION_VERSION_V1:
        return _candidate_issuance_row_id_v1(issuance_id, row)
    raise UnsupportedCandidateReferenceVersionError(
        f"Candidate reference derivation version {version} is not supported."
    )


def _candidate_issuance_row_id_v1(issuance_id: str, row: CandidateIssuanceRow) -> str:
    """Version 1 candidate-reference derivation, unchanged from the original.

    The payload preserves the candidate-reference identity originally owned
    by the market closeout boundary so peer consumers share one lasting
    """
    normalized_issuance_id = _nonempty(issuance_id, label="issuance_id")
    identity: dict[str, bool | float | int | str | None] = {
        "issuance_id": normalized_issuance_id,
        "provider": row.provider,
        "provider_event_id": row.provider_event_id,
        "sportsbook": row.sportsbook,
        "game_id": row.game_id,
        "market": row.market,
        "side": row.side,
        "fetched_at": row.fetched_at.isoformat(),
        "sportsbook_updated_at": (
            None if row.sportsbook_updated_at is None else row.sportsbook_updated_at.isoformat()
        ),
        "is_live": row.is_live,
        "american_price": row.american_price,
        "line": row.line,
    }
    digest = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{normalized_issuance_id}:{digest}"


def issue_pregame_candidates(
    *,
    product: DataFrame,
    forecast_events: DataFrame,
    quotes: DataFrame,
    evaluated_at: datetime,
) -> CandidateIssuance:
    """Evaluate and freeze every supplied exact quote before kickoff."""
    evaluated = _require_utc(evaluated_at, label="evaluated_at")
    product_rows = _validate_selected_product(product)
    events = validate_forecast_events(forecast_events)
    _reject_non_live_or_late_events(events, evaluated_at=evaluated)
    quote_rows = validate_quote_rows(quotes)
    _reject_duplicate_quotes(quote_rows)

    product_id = _single_text(product_rows, "product_id")
    product_run_id = _single_text(product_rows, "product_run_id")
    season = _single_text(product_rows, "season")
    week = _single_int(product_rows, "week")
    product_generated_at = _single_datetime(product_rows, "product_generated_at")
    issuance_id = candidate_issuance_id(
        product_id=product_id,
        product_run_id=product_run_id,
        season=season,
        week=week,
        evaluated_at=evaluated,
    )

    if quote_rows.empty:
        return CandidateIssuance(
            CANDIDATE_ISSUANCE_SCHEMA_VERSION,
            issuance_id,
            product_id,
            product_run_id,
            product_generated_at,
            season,
            week,
            evaluated,
            (),
        )

    if (
        not quote_rows["season"].astype(str).eq(season).all()
        or not quote_rows["week"].astype(int).eq(week).all()
    ):
        raise ValueError("Candidate issuance quotes must match the selected product scope.")

    known_kickoffs = quote_rows["commence_time"].dropna()
    if known_kickoffs.le(pd.Timestamp(evaluated)).any():
        raise ValueError("Candidate issuance must occur strictly before kickoff.")

    evaluable_mask = (
        quote_rows["sportsbook"].notna()
        & quote_rows["odds"].notna()
        & quote_rows["commence_time"].notna()
        & quote_rows["is_live"].eq(False)
        & quote_rows["fetched_at"].lt(quote_rows["commence_time"])
    )
    evaluable = quote_rows.loc[evaluable_mask, :].copy()
    evaluated_offers = (
        evaluate_line_shopping_guidance(product_rows, evaluable).offers
        if not evaluable.empty
        else DataFrame()
    )
    offer_by_identity = {_quote_identity(row): row for _, row in evaluated_offers.iterrows()}
    events_by_id = events.set_index("event_id", drop=False)
    product_by_game = product_rows.set_index("game_id", drop=False)

    rows = [
        _issuance_row(
            quote,
            offer=offer_by_identity.get(_quote_identity(quote)),
            product_by_game=product_by_game,
            events_by_id=events_by_id,
        )
        for _, quote in quote_rows.sort_values(
            list(OBSERVATION_SORT_COLUMNS),
            kind="stable",
            na_position="first",
        ).iterrows()
    ]
    return CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        product_id,
        product_run_id,
        product_generated_at,
        season,
        week,
        evaluated,
        tuple(rows),
    )


def _issuance_row(
    quote: Series,
    *,
    offer: Series | None,
    product_by_game: DataFrame,
    events_by_id: DataFrame,
) -> CandidateIssuanceRow:
    kickoff = _optional_datetime(quote["commence_time"])
    event_id = _referenced_event_id(quote, product_by_game)
    event = _event_row(event_id, events_by_id)
    state, reason = _state_and_reason(quote, offer=offer, event=event)
    return CandidateIssuanceRow(
        game_id=str(quote["game_id"]),
        market=str(quote["market"]),
        side=str(quote["side"]),
        provider=str(quote["provider"]),
        provider_event_id=_optional_text(quote["provider_event_id"]),
        sportsbook=_optional_text(quote["sportsbook"]),
        line=_optional_float(quote["line"]),
        american_price=(None if pd.isna(quote["odds"]) else int(float(quote["odds"]))),
        fetched_at=_datetime(quote["fetched_at"]),
        sportsbook_updated_at=_optional_datetime(quote["sportsbook_updated_at"]),
        kickoff=kickoff,
        is_live=bool(quote["is_live"]),
        forecast_event_id=event_id if event is not None else None,
        forecast_run_id=(None if event is None else str(event["run_id"])),
        forecast_role=(None if event is None else str(event["role"])),
        forecast_generated_at=(None if event is None else _datetime(event["generated_at"])),
        model_name=(None if event is None else str(event["model_name"])),
        model_type=(None if event is None else str(event["model_type"])),
        model_probability=(None if offer is None else _optional_float(offer["model_probability"])),
        expected_value=(None if offer is None else _optional_float(offer["expected_value"])),
        state=state,
        reason=reason,
    )


def _state_and_reason(
    quote: Series,
    *,
    offer: Series | None,
    event: Series | None,
) -> tuple[CandidateIssuanceState, CandidateIssuanceReason]:
    """Return the deterministic state and reason for one quote observation."""
    kickoff = _optional_datetime(quote["commence_time"])
    if kickoff is None:
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.KICKOFF_UNAVAILABLE,
        )
    elif bool(quote["is_live"]):
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.QUOTE_LIVE,
        )
    elif _datetime(quote["fetched_at"]) >= kickoff:
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.QUOTE_NOT_PREGAME,
        )
    elif pd.isna(quote["odds"]) or pd.isna(quote["sportsbook"]):
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.QUOTE_UNAVAILABLE,
        )
    elif event is None:
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.FORECAST_EVENT_UNAVAILABLE,
        )
    elif offer is None or str(offer["model_status"]) == "model_unavailable":
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.MODEL_UNAVAILABLE,
        )
    elif str(offer["model_status"]) == "uncertainty_unavailable":
        result = (
            CandidateIssuanceState.UNAVAILABLE,
            CandidateIssuanceReason.UNCERTAINTY_UNAVAILABLE,
        )
    else:
        probability = _optional_float(offer["model_probability"])
        expected_value = _optional_float(offer["expected_value"])
        if probability is None:
            result = (
                CandidateIssuanceState.UNAVAILABLE,
                CandidateIssuanceReason.MODEL_PROBABILITY_UNAVAILABLE,
            )
        elif expected_value is None:
            result = (
                CandidateIssuanceState.UNAVAILABLE,
                CandidateIssuanceReason.EXPECTED_VALUE_UNAVAILABLE,
            )
        elif expected_value > 0:
            result = (
                CandidateIssuanceState.CANDIDATE,
                CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
            )
        else:
            result = (
                CandidateIssuanceState.NOT_CANDIDATE,
                CandidateIssuanceReason.EXPECTED_VALUE_NOT_POSITIVE,
            )
    return result


def _referenced_event_id(quote: Series, product_by_game: DataFrame) -> str | None:
    game_id = str(quote["game_id"])
    if game_id not in product_by_game.index:
        return None
    product = product_by_game.loc[game_id]
    if isinstance(product, DataFrame):
        raise ValueError(f"Selected product game ID is not unique: {game_id}")
    market = str(quote["market"])
    column = {
        "moneyline": "win_event_id",
        "spread": "spread_source_event_id",
        "total": "total_event_id",
    }[market]
    return _optional_text(product[column])


def _event_row(event_id: str | None, events_by_id: DataFrame) -> Series | None:
    if event_id is None or event_id not in events_by_id.index:
        return None
    row = events_by_id.loc[event_id]
    if isinstance(row, DataFrame):
        raise ValueError(f"Forecast event ID is not unique: {event_id}")
    return row


def _validate_selected_product(product: DataFrame) -> DataFrame:
    missing = sorted(set(_PRODUCT_STORAGE_COLUMNS) - set(product.columns))
    if missing:
        raise ValueError("Selected product is missing storage identity: " + ", ".join(missing))
    domain = product.drop(columns=list(_PRODUCT_STORAGE_COLUMNS))
    validate_weekly_game_product(domain)
    if product["game_id"].astype(str).duplicated().any():
        raise ValueError("Selected product contains duplicate game IDs.")
    return product.copy(deep=True)


def _reject_duplicate_quotes(quotes: DataFrame) -> None:
    if quotes.duplicated(subset=list(OBSERVATION_IDENTITY_COLUMNS)).any():
        raise ValueError("Candidate issuance contains duplicate quote observation identities.")


def _reject_non_live_or_late_events(events: DataFrame, *, evaluated_at: datetime) -> None:
    """Reject a non-live forecast role or a forecast generated after evaluation.

    A development or backfilled forecast role is never eligible for a real
    pregame recommendation, and a forecast generated after the issuance's own
    evaluation time could not have been known at that moment -- evaluating
    against it would be look-ahead leakage.
    """
    if events.empty:
        return
    invalid_roles = sorted(set(events["role"].astype(str)) - {ForecastRole.LIVE.value})
    if invalid_roles:
        raise ValueError(
            "Candidate issuance requires exclusively live forecast events; found: "
            + ", ".join(invalid_roles)
        )
    if events["generated_at"].apply(_datetime).gt(evaluated_at).any():
        raise ValueError(
            "Candidate issuance must not reference a forecast generated after evaluated_at."
        )


def _quote_identity(row: Series) -> tuple[object, ...]:
    return tuple(_identity_value(row[column]) for column in OBSERVATION_IDENTITY_COLUMNS)


def _identity_value(value: object) -> object:
    """Canonicalize one validated DataFrame-cell identity value."""
    scalar = cast(Any, value)
    if pd.isna(scalar):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _single_text(rows: DataFrame, column: str) -> str:
    values = rows[column].dropna().astype(str).unique().tolist()
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"Selected product must contain one nonempty {column}.")
    return values[0]


def _single_int(rows: DataFrame, column: str) -> int:
    values = rows[column].dropna().astype(int).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Selected product must contain one {column}.")
    return int(values[0])


def _single_datetime(rows: DataFrame, column: str) -> datetime:
    values = cast(Series, pd.to_datetime(rows[column], utc=True, errors="coerce")).dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Selected product must contain one valid {column}.")
    return _datetime(values[0])


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value


def _nonempty(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    return normalized


def _positive_week(value: int) -> int:
    if value < 1:
        raise ValueError("week must be at least 1.")
    return value


def _datetime(value: object) -> datetime:
    """Normalize one validated DataFrame-cell timestamp to UTC datetime."""
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("Candidate issuance timestamps must be timezone-aware UTC.")
    return timestamp.to_pydatetime()


def _optional_datetime(value: object) -> datetime | None:
    """Normalize one optional validated DataFrame-cell timestamp."""
    return None if pd.isna(cast(Any, value)) else _datetime(value)


def _optional_text(value: object) -> str | None:
    """Normalize one optional validated DataFrame-cell text value."""
    return None if pd.isna(cast(Any, value)) else str(value)


def _optional_float(value: object) -> float | None:
    """Normalize one optional validated DataFrame-cell numeric value."""
    return None if pd.isna(cast(Any, value)) else float(cast(float | int, value))
