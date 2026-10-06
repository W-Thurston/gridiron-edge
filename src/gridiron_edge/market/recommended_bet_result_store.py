# src/gridiron_edge/market/recommended_bet_result_store.py

"""Immutable JSON persistence for recommended-bet results and evaluations.

Besides writing and reading, it owns the one rule for which persisted result stands for an
exact offer in a week: the latest unambiguous evaluation
(:func:`load_latest_recommended_bet_results`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
from types import UnionType
from typing import cast, get_args, get_origin, get_type_hints
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.market.recommended_bet_result import (
    RecommendedBetEvaluation,
    RecommendedBetResult,
    validate_recommended_bet_evaluation,
    validate_recommended_bet_result,
)


@dataclass(frozen=True, slots=True)
class LoadedRecommendedBetResult:
    """One persisted result paired with its immutable evaluation identity."""

    evaluation_id: str
    result: RecommendedBetResult


RecommendedBetOfferKey = tuple[
    str,
    str | None,
    str | None,
    str,
    str,
    str,
    datetime,
    datetime | None,
    datetime | None,
    bool,
    int | None,
    float | None,
]


def recommended_bet_offer_key(
    result: RecommendedBetResult,
) -> RecommendedBetOfferKey:
    """Return the complete exact-offer key preserved by one result."""
    return (
        result.provider,
        result.provider_event_id,
        result.sportsbook,
        result.game_id,
        result.market,
        result.side,
        result.fetched_at,
        result.sportsbook_updated_at,
        result.kickoff,
        result.is_live,
        result.american_price,
        result.line,
    )


def load_latest_recommended_bet_results(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> tuple[LoadedRecommendedBetResult, ...]:
    """Load the latest unambiguous persisted result of every exact offer in one week.

    Selection uses persisted ``evaluated_at`` values only. Equal-time,
    different-result evidence for one exact offer is rejected as conflicting.
    """
    root = recommended_bet_result_root(repo)
    if not root.is_dir():
        return ()

    selected: dict[RecommendedBetOfferKey, LoadedRecommendedBetResult] = {}
    for path in sorted(root.glob("schema=*/evaluations/*.json")):
        evaluation = read_recommended_bet_evaluation(path)
        for result in evaluation.results:
            if result.season != season or result.week != week:
                continue
            candidate = LoadedRecommendedBetResult(evaluation.evaluation_id, result)
            key = recommended_bet_offer_key(result)
            existing = selected.get(key)
            if existing is None or result.evaluated_at > existing.result.evaluated_at:
                selected[key] = candidate
            elif (
                result.evaluated_at == existing.result.evaluated_at
                and result.result_id != existing.result.result_id
            ):
                raise ValueError(
                    "Conflicting persisted recommendation results share the latest "
                    "evaluation time for one exact offer."
                )

    return tuple(selected[key] for key in sorted(selected, key=_offer_sort_key))


def _offer_sort_key(key: RecommendedBetOfferKey) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)


def recommended_bet_result_root(repo: Path | None = None) -> Path:
    """Return the immutable recommended-bet result storage root."""
    root = repo or get_settings().repo_root
    return root / "data" / "output" / "recommended_bet_results"


def recommended_bet_result_path(
    schema_version: int, result_id: str, *, repo: Path | None = None
) -> Path:
    """Return the identity-addressed path for one persisted result."""
    return (
        recommended_bet_result_root(repo)
        / f"schema={_schema(schema_version)}"
        / "results"
        / f"{_digest(result_id, 'result_id')}.json"
    )


def recommended_bet_evaluation_path(
    schema_version: int, evaluation_id: str, *, repo: Path | None = None
) -> Path:
    """Return the identity-addressed path for one evaluation manifest."""
    return (
        recommended_bet_result_root(repo)
        / f"schema={_schema(schema_version)}"
        / "evaluations"
        / f"{_digest(evaluation_id, 'evaluation_id')}.json"
    )


def write_recommended_bet_result(result: RecommendedBetResult, *, repo: Path | None = None) -> Path:
    """Persist one immutable result or accept an exact replay."""
    validate_recommended_bet_result(result)
    path = recommended_bet_result_path(result.schema_version, result.result_id, repo=repo)
    _immutable_write(path, _encode(result), label="Recommended-bet result")
    return path


def read_recommended_bet_result(path: Path) -> RecommendedBetResult:
    """Read and validate one exact persisted recommended-bet result."""
    raw = _json_object(path)
    result = cast(
        RecommendedBetResult,
        _decode_dataclass(RecommendedBetResult, raw),
    )
    validate_recommended_bet_result(result)
    repo = _artifact_repo(path)
    expected = recommended_bet_result_path(result.schema_version, result.result_id, repo=repo)
    if path.resolve() != expected.resolve():
        raise ValueError("Recommended-bet result path and embedded identity disagree.")
    return result


def write_recommended_bet_evaluation(
    evaluation: RecommendedBetEvaluation, *, repo: Path | None = None
) -> Path:
    """Persist one evaluation manifest and all referenced results."""
    validate_recommended_bet_evaluation(evaluation)
    for result in evaluation.results:
        write_recommended_bet_result(result, repo=repo)
    path = recommended_bet_evaluation_path(
        evaluation.schema_version, evaluation.evaluation_id, repo=repo
    )
    payload = {
        "schema_version": evaluation.schema_version,
        "evaluation_id": evaluation.evaluation_id,
        "issuance_id": evaluation.issuance_id,
        "policy_id": evaluation.policy_id,
        "evaluated_at": evaluation.evaluated_at.isoformat(),
        "result_ids": [result.result_id for result in evaluation.results],
    }
    _immutable_write(path, payload, label="Recommended-bet evaluation")
    return path


def read_recommended_bet_evaluation(path: Path) -> RecommendedBetEvaluation:
    """Read and validate one evaluation and its referenced results."""
    raw = _json_object(path)
    expected_keys = {
        "schema_version",
        "evaluation_id",
        "issuance_id",
        "policy_id",
        "evaluated_at",
        "result_ids",
    }
    if set(raw) != expected_keys:
        raise ValueError("Recommended-bet evaluation keys do not match the schema.")
    schema_version = _strict_int(raw["schema_version"], "schema_version")
    evaluation_id = _strict_text(raw["evaluation_id"], "evaluation_id")
    repo = _artifact_repo(path)
    result_ids = raw["result_ids"]
    if not isinstance(result_ids, list):
        raise ValueError("Recommended-bet evaluation result_ids must be a list.")
    results = tuple(
        read_recommended_bet_result(
            recommended_bet_result_path(schema_version, _strict_text(value, "result_id"), repo=repo)
        )
        for value in result_ids
    )
    evaluation = RecommendedBetEvaluation(
        schema_version,
        evaluation_id,
        _strict_text(raw["issuance_id"], "issuance_id"),
        _strict_text(raw["policy_id"], "policy_id"),
        _strict_datetime(raw["evaluated_at"], "evaluated_at"),
        results,
    )
    validate_recommended_bet_evaluation(evaluation)
    expected = recommended_bet_evaluation_path(schema_version, evaluation_id, repo=repo)
    if path.resolve() != expected.resolve():
        raise ValueError("Recommended-bet evaluation path and identity disagree.")
    return evaluation


def _immutable_write(path: Path, payload: object, *, label: str) -> None:
    encoded: str = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"{label} identity cannot be reused with different content.")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    f"{label} identity cannot be reused with different content."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _encode(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported recommended-bet store value: {type(value).__name__}")


def _decode_dataclass(cls: type[object], raw: object) -> object:
    """Strictly reconstruct one immutable dataclass from exact JSON fields."""
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object.")
    hints = get_type_hints(cls)
    expected = {
        field.name
        for field in fields(
            cls  # pyrefly: ignore [bad-argument-type]
        )
    }
    if set(raw) != expected:
        raise ValueError(f"{cls.__name__} keys do not match the current schema.")
    values = {name: _decode_value(hints[name], raw[name], name) for name in expected}
    return cls(**values)


def _decode_value(annotation: object, value: object, label: str) -> object:
    """Dispatch one stored value to a narrow decoder."""
    origin = get_origin(annotation)
    if origin is UnionType:
        result = _decode_union(get_args(annotation), value, label)
    elif origin is tuple:
        result = _decode_tuple(get_args(annotation), value, label)
    elif isinstance(annotation, type) and is_dataclass(annotation):
        result = _decode_dataclass(annotation, value)
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        result = _decode_enum(annotation, value, label)
    else:
        result = _decode_scalar(annotation, value, label)
    return result


def _decode_union(args: tuple[object, ...], value: object, label: str) -> object:
    if value is None and type(None) in args:
        return None
    candidates = tuple(item for item in args if item is not type(None))
    if len(candidates) == 1:
        return _decode_value(candidates[0], value, label)
    for candidate in candidates:
        try:
            return _decode_value(candidate, value, label)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"{label} does not match its union contract.")


def _decode_tuple(args: tuple[object, ...], value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list.")
    item_type = args[0]
    return tuple(_decode_value(item_type, item, label) for item in value)


def _decode_enum(annotation: type[Enum], value: object, label: str) -> Enum:
    if not isinstance(value, str):
        raise ValueError(f"{label} enum value must be a string.")
    return annotation(value)


def _decode_scalar(annotation: object, value: object, label: str) -> object:
    if annotation is datetime:
        result: object = _strict_datetime(value, label)
    elif annotation is str:
        result = _strict_text(value, label)
    elif annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be boolean.")
        result = value
    elif annotation is int:
        result = _strict_int(value, label)
    elif annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} must be numeric.")
        result = float(value)
    elif value is None:
        result = None
    else:
        raise TypeError(f"Unsupported stored annotation for {label}: {annotation!r}")
    return result


def _json_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Recommended-bet artifact must contain a JSON object.")
    return raw


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = ("data", "output", "recommended_bet_results")
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Recommended-bet artifact is outside the canonical store.")


def _schema(value: int) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError("schema_version must be a positive integer.")
    return value


def _digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _strict_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _strict_datetime(value: object, label: str) -> datetime:
    text = _strict_text(value, label)
    result = datetime.fromisoformat(text)
    offset = result.utcoffset()
    if result.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return result
