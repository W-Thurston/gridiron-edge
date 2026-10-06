# src/gridiron_edge/market/candidate_issuance_store.py

"""Immutable JSON persistence for pregame candidate issuance."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.market.candidate_issuance import (
    CANDIDATE_ISSUANCE_SCHEMA_VERSION,
    CandidateIssuance,
    CandidateIssuanceReason,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_id,
)


def candidate_issuance_root(repo: Path | None = None) -> Path:
    """Return the immutable candidate-issuance storage root."""
    root = repo or get_settings().repo_root
    return root / "data" / "output" / "candidate_issuance"


def candidate_issuance_path(
    issuance_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the immutable path for one deterministic issuance ID."""
    normalized = _validate_issuance_id(issuance_id)
    return candidate_issuance_root(repo) / "issuances" / f"{normalized}.json"


def write_candidate_issuance(
    issuance: CandidateIssuance,
    *,
    repo: Path | None = None,
) -> Path:
    """Create one issuance or accept an exact idempotent replay."""
    validate_candidate_issuance(issuance)
    path: Path = candidate_issuance_path(issuance.issuance_id, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = _payload(issuance)
    encoded: str = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary: Path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing: CandidateIssuance = read_candidate_issuance(path)
            if existing != issuance:
                raise ValueError(
                    "Candidate issuance ID cannot be reused with different content: "
                    f"{issuance.issuance_id}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def candidate_issuances_for_product(
    *,
    season: str,
    week: int,
    product_id: str,
    product_run_id: str,
    repo: Path | None = None,
) -> tuple[CandidateIssuance, ...]:
    """Return every stored issuance of one week's exact product and run, oldest file name first.

    An unreadable or invalid artifact is skipped; callers decide what more than one match means.
    """
    directory = candidate_issuance_root(repo) / "issuances"
    if not directory.is_dir():
        return ()
    matches: list[CandidateIssuance] = []
    for path in sorted(directory.glob("*.json")):
        try:
            issuance = read_candidate_issuance(path)
        except (OSError, ValueError):
            continue
        if (
            issuance.season == season
            and issuance.week == week
            and issuance.product_id == product_id
            and issuance.product_run_id == product_run_id
        ):
            matches.append(issuance)
    return tuple(matches)


def read_candidate_issuance(path: Path) -> CandidateIssuance:
    """Read and validate one exact immutable issuance artifact."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Candidate issuance artifact must contain a JSON object.")
    required = {
        "schema_version",
        "issuance_id",
        "product_id",
        "product_run_id",
        "product_generated_at",
        "season",
        "week",
        "evaluated_at",
        "rows",
    }
    if set(raw) != required:
        raise ValueError("Candidate issuance artifact keys do not match the current schema.")
    raw_rows = raw["rows"]
    if not isinstance(raw_rows, list):
        raise ValueError("Candidate issuance rows must be a list.")
    issuance = CandidateIssuance(
        schema_version=cast(int, raw["schema_version"]),
        issuance_id=str(raw["issuance_id"]),
        product_id=str(raw["product_id"]),
        product_run_id=str(raw["product_run_id"]),
        product_generated_at=_datetime(raw["product_generated_at"]),
        season=str(raw["season"]),
        week=cast(int, raw["week"]),
        evaluated_at=_datetime(raw["evaluated_at"]),
        rows=tuple(_row(item) for item in raw_rows),
    )
    validate_candidate_issuance(issuance)
    if path.stem != issuance.issuance_id:
        raise ValueError("Candidate issuance filename and issuance_id disagree.")
    return issuance


def validate_candidate_issuance(issuance: CandidateIssuance) -> None:
    """Validate identity, schema, ordering, and state invariants."""
    if issuance.schema_version != CANDIDATE_ISSUANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate issuance schema version.")
    expected_id = candidate_issuance_id(
        product_id=issuance.product_id,
        product_run_id=issuance.product_run_id,
        season=issuance.season,
        week=issuance.week,
        evaluated_at=issuance.evaluated_at,
    )
    if issuance.issuance_id != expected_id:
        raise ValueError("Candidate issuance ID does not match its identity payload.")
    observed = [_row_sort_key(row) for row in issuance.rows]
    if observed != sorted(observed):
        raise ValueError("Candidate issuance rows are not deterministically ordered.")
    if len(observed) != len(set(observed)):
        raise ValueError("Candidate issuance contains duplicate row identities.")
    for row in issuance.rows:
        if (
            row.state is CandidateIssuanceState.CANDIDATE
            and row.reason is not CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE
        ):
            raise ValueError("Candidate rows require positive_expected_value.")
        if (
            row.state is CandidateIssuanceState.NOT_CANDIDATE
            and row.reason is not CandidateIssuanceReason.EXPECTED_VALUE_NOT_POSITIVE
        ):
            raise ValueError("Not-candidate rows require expected_value_not_positive.")


def _payload(issuance: CandidateIssuance) -> dict[str, object]:
    return {
        "schema_version": issuance.schema_version,
        "issuance_id": issuance.issuance_id,
        "product_id": issuance.product_id,
        "product_run_id": issuance.product_run_id,
        "product_generated_at": issuance.product_generated_at.isoformat(),
        "season": issuance.season,
        "week": issuance.week,
        "evaluated_at": issuance.evaluated_at.isoformat(),
        "rows": [_row_payload(row) for row in issuance.rows],
    }


def _row_payload(row: CandidateIssuanceRow) -> dict[str, object]:
    payload = asdict(row)
    for column in (
        "fetched_at",
        "sportsbook_updated_at",
        "kickoff",
        "forecast_generated_at",
    ):
        value = payload[column]
        payload[column] = None if value is None else cast(datetime, value).isoformat()
    payload["state"] = row.state.value
    payload["reason"] = row.reason.value
    return payload


def _row(value: object) -> CandidateIssuanceRow:
    if not isinstance(value, dict):
        raise ValueError("Candidate issuance row must be a JSON object.")
    expected = set(asdict(_empty_row()))
    if set(value) != expected:
        raise ValueError("Candidate issuance row keys do not match the current schema.")
    return CandidateIssuanceRow(
        game_id=str(value["game_id"]),
        market=str(value["market"]),
        side=str(value["side"]),
        provider=str(value["provider"]),
        provider_event_id=_optional_text(value["provider_event_id"]),
        sportsbook=_optional_text(value["sportsbook"]),
        line=_optional_float(value["line"]),
        american_price=_optional_int(value["american_price"]),
        fetched_at=_datetime(value["fetched_at"]),
        sportsbook_updated_at=_optional_datetime(value["sportsbook_updated_at"]),
        kickoff=_optional_datetime(value["kickoff"]),
        is_live=bool(value["is_live"]),
        forecast_event_id=_optional_text(value["forecast_event_id"]),
        forecast_run_id=_optional_text(value["forecast_run_id"]),
        forecast_role=_optional_text(value["forecast_role"]),
        forecast_generated_at=_optional_datetime(value["forecast_generated_at"]),
        model_name=_optional_text(value["model_name"]),
        model_type=_optional_text(value["model_type"]),
        model_probability=_optional_float(value["model_probability"]),
        expected_value=_optional_float(value["expected_value"]),
        state=CandidateIssuanceState(str(value["state"])),
        reason=CandidateIssuanceReason(str(value["reason"])),
    )


def _row_sort_key(row: CandidateIssuanceRow) -> tuple[str, ...]:
    return (
        row.fetched_at.isoformat(),
        row.provider,
        row.provider_event_id or "",
        row.sportsbook or "",
        row.game_id,
        row.market,
        row.side,
        "" if row.line is None else repr(row.line),
        "" if row.american_price is None else str(row.american_price),
        "" if row.sportsbook_updated_at is None else row.sportsbook_updated_at.isoformat(),
        str(row.is_live),
    )


def _validate_issuance_id(value: str) -> str:
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("issuance_id must be a lowercase SHA-256 digest.")
    return normalized


def _datetime(value: object) -> datetime:
    """Parse one timezone-aware UTC ISO timestamp."""
    if not isinstance(value, str):
        raise ValueError("Candidate issuance timestamp must be an ISO string.")
    result = datetime.fromisoformat(value)
    offset = result.utcoffset()
    if result.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError("Candidate issuance timestamp must be timezone-aware UTC.")
    return result


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(cast(float | int, value))


def _optional_int(value: object) -> int | None:
    return None if value is None else cast(int, value)


def _empty_row() -> CandidateIssuanceRow:
    timestamp = datetime.fromisoformat("2000-01-01T00:00:00+00:00")
    return CandidateIssuanceRow(
        "game",
        "moneyline",
        "home",
        "provider",
        None,
        None,
        None,
        None,
        timestamp,
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        CandidateIssuanceState.UNAVAILABLE,
        CandidateIssuanceReason.QUOTE_UNAVAILABLE,
    )
