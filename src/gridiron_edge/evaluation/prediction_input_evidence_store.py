# src/gridiron_edge/evaluation/prediction_input_evidence_store.py
"""Immutable persistence for selected weekly prediction-input evidence and binary snapshots."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import cast
from uuid import uuid4

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    EloPredictionEventEvidence,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionExecutionKind,
    PredictionFeatureSchema,
    PredictionInputEvidence,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    StatisticalPredictionEventEvidence,
    prediction_input_evidence_payload,
    validate_prediction_input_evidence,
)

_STORE_DIRECTORY = "data/output/prediction_input_evidence"
_STORE_SCHEMA_VERSION = 1


def prediction_input_evidence_root(repo: Path | None = None) -> Path:
    """Return the canonical prediction-input evidence root."""
    return (repo or get_settings().repo_root) / _STORE_DIRECTORY


def binary_snapshot_path(
    content_digest: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the canonical content-addressed binary snapshot path."""
    digest = _digest(content_digest, "content_digest")
    return (
        prediction_input_evidence_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "artifacts"
        / f"{digest}.bin"
    )


def prediction_input_evidence_path(
    evidence_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the canonical identity-addressed family evidence path."""
    identity = _digest(evidence_id, "evidence_id")
    return (
        prediction_input_evidence_root(repo)
        / f"schema={_STORE_SCHEMA_VERSION}"
        / "evidence"
        / f"{identity}.json"
    )


def write_binary_snapshot(
    source: Path,
    *,
    expected_digest: str,
    expected_size_bytes: int,
    repo: Path | None = None,
) -> Path:
    """Copy and create one immutable binary snapshot or accept exact replay."""
    digest = _digest(expected_digest, "expected_digest")
    size = _nonnegative_integer(expected_size_bytes, "expected_size_bytes")
    if not source.is_file():
        raise FileNotFoundError(f"Binary snapshot source is missing: {source}")
    source_digest, source_size = _file_identity(source)
    if source_digest != digest or source_size != size:
        raise ValueError("Binary snapshot source identity does not match its reference.")

    path = binary_snapshot_path(digest, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _authenticate_binary_path(path, digest=digest, size_bytes=size)
        return path

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_stream, temporary.open("xb") as destination:
            shutil.copyfileobj(source_stream, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        _authenticate_binary_path(temporary, digest=digest, size_bytes=size)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _authenticate_binary_path(path, digest=digest, size_bytes=size)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def authenticate_binary_snapshot(
    reference: BinaryArtifactReference,
    *,
    repo: Path | None = None,
) -> Path | None:
    """Authenticate one present snapshot or explicit absent artifact state."""
    if reference.state is PredictionArtifactState.ABSENT:
        if reference.content_digest is not None or reference.size_bytes is not None:
            raise ValueError("Absent binary artifact must not contain digest or size.")
        return None
    if reference.content_digest is None or reference.size_bytes is None:
        raise ValueError("Present binary artifact requires digest and size.")
    path = binary_snapshot_path(reference.content_digest, repo=repo)
    if not path.is_file():
        raise FileNotFoundError(f"Binary snapshot is missing: {path}")
    _authenticate_binary_path(
        path,
        digest=reference.content_digest,
        size_bytes=reference.size_bytes,
    )
    return path


def write_prediction_input_evidence(
    evidence: PredictionInputEvidence,
    *,
    repo: Path | None = None,
) -> Path:
    """Create one immutable family evidence artifact or accept exact replay."""
    validate_prediction_input_evidence(evidence)
    path = prediction_input_evidence_path(evidence.evidence_id, repo=repo)
    encoded = (
        json.dumps(
            {
                "store_schema_version": _STORE_SCHEMA_VERSION,
                "evidence_id": evidence.evidence_id,
                "evidence": prediction_input_evidence_payload(evidence),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                "Prediction-input evidence identity cannot be reused with different content."
            )
        read_prediction_input_evidence(path)
        return path

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(
                    "Prediction-input evidence identity cannot be reused with different content."
                ) from None
        read_prediction_input_evidence(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_prediction_input_evidence(path: Path) -> PredictionInputEvidence:
    """Read and strictly validate one exact immutable family evidence artifact."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prediction-input evidence contains malformed JSON: {path}") from exc
    raw = _object(value, "prediction-input evidence artifact")
    _exact_keys(raw, {"store_schema_version", "evidence_id", "evidence"}, "Artifact")
    store_version = _integer(raw["store_schema_version"], "store_schema_version")
    if store_version != _STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported prediction-input evidence store schema version.")
    embedded_id = _digest(_text(raw["evidence_id"], "evidence_id"), "evidence_id")
    evidence = _evidence(raw["evidence"])
    if embedded_id != evidence.evidence_id:
        raise ValueError("Stored evidence identity does not match evidence content.")
    expected = prediction_input_evidence_path(embedded_id, repo=_artifact_repo(path))
    if path.resolve() != expected.resolve():
        raise ValueError("Prediction-input evidence path and embedded identity disagree.")
    return evidence


def list_prediction_input_evidence_by_run(
    run_id: str,
    *,
    repo: Path | None = None,
) -> tuple[PredictionInputEvidence, ...]:
    """Scan and deterministically list exact family evidence for one run."""
    normalized = _text(run_id, "run_id")
    return tuple(value for value in _all_evidence(repo=repo) if value.run_id == normalized)


def find_prediction_input_evidence_by_event(
    event_id: str,
    *,
    repo: Path | None = None,
) -> PredictionInputEvidence | None:
    """Scan immutable evidence and resolve at most one artifact for an event."""
    normalized = _text(event_id, "event_id")
    matches = tuple(
        evidence for evidence in _all_evidence(repo=repo) if normalized in _event_ids(evidence)
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple prediction-input evidence artifacts claim event {normalized!r}."
        )
    return matches[0] if matches else None


def _all_evidence(*, repo: Path | None) -> tuple[PredictionInputEvidence, ...]:
    directory = (
        prediction_input_evidence_root(repo) / f"schema={_STORE_SCHEMA_VERSION}" / "evidence"
    )
    if not directory.exists():
        return ()
    values = tuple(
        read_prediction_input_evidence(path) for path in sorted(directory.glob("*.json"))
    )
    return tuple(sorted(values, key=lambda value: value.evidence_id))


def _event_ids(evidence: PredictionInputEvidence) -> tuple[str, ...]:
    events = evidence.statistical_events or evidence.elo_events
    return tuple(value.event_id for value in events)


def _authenticate_binary_path(path: Path, *, digest: str, size_bytes: int) -> None:
    actual_digest, actual_size = _file_identity(path)
    if actual_digest != digest or actual_size != size_bytes:
        raise ValueError(f"Binary snapshot identity does not match stored bytes: {path}")


def _file_identity(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _evidence(value: object) -> PredictionInputEvidence:
    raw = _object(value, "evidence")
    _exact_keys(
        raw,
        {
            "schema_version",
            "evidence_id",
            "run_id",
            "season",
            "week",
            "generated_at",
            "model_name",
            "model_type",
            "execution_kind",
            "source_revision",
            "source_artifacts",
            "binary_artifacts",
            "feature_schema",
            "post_processing",
            "statistical_events",
            "elo_events",
        },
        "Evidence",
    )
    evidence = PredictionInputEvidence(
        schema_version=_integer(raw["schema_version"], "schema_version"),
        evidence_id=_digest(_text(raw["evidence_id"], "evidence_id"), "evidence_id"),
        run_id=_text(raw["run_id"], "run_id"),
        season=_text(raw["season"], "season"),
        week=_integer(raw["week"], "week"),
        generated_at=_datetime(raw["generated_at"], "generated_at"),
        model_name=_text(raw["model_name"], "model_name"),
        model_type=_text(raw["model_type"], "model_type"),
        execution_kind=_enum(raw["execution_kind"], PredictionExecutionKind, "execution_kind"),
        source_revision=_source_revision(raw["source_revision"]),
        source_artifacts=tuple(
            _source_reference(item) for item in _list(raw["source_artifacts"], "source_artifacts")
        ),
        binary_artifacts=tuple(
            _binary_reference(item) for item in _list(raw["binary_artifacts"], "binary_artifacts")
        ),
        feature_schema=(
            None if raw["feature_schema"] is None else _feature_schema(raw["feature_schema"])
        ),
        post_processing=(
            None if raw["post_processing"] is None else _post_processing(raw["post_processing"])
        ),
        statistical_events=tuple(
            _statistical_event(item)
            for item in _list(raw["statistical_events"], "statistical_events")
        ),
        elo_events=tuple(_elo_event(item) for item in _list(raw["elo_events"], "elo_events")),
    )
    validate_prediction_input_evidence(evidence)
    return evidence


def _source_revision(value: object) -> SourceRevision:
    raw = _object(value, "source_revision")
    _exact_keys(raw, {"commit", "tracked_worktree_clean"}, "Source revision")
    clean = raw["tracked_worktree_clean"]
    if not isinstance(clean, bool):
        raise ValueError("tracked_worktree_clean must be a boolean.")
    return SourceRevision(commit=_text(raw["commit"], "commit"), tracked_worktree_clean=clean)


def _source_reference(value: object) -> SourceArtifactReference:
    raw = _object(value, "source artifact")
    _exact_keys(
        raw,
        {"relative_path", "state", "content_digest", "size_bytes"},
        "Source artifact",
    )
    return SourceArtifactReference(
        relative_path=_text(raw["relative_path"], "relative_path"),
        state=_enum(raw["state"], PredictionSourceState, "source state"),
        content_digest=_optional_digest(raw["content_digest"], "content_digest"),
        size_bytes=_optional_integer(raw["size_bytes"], "size_bytes"),
    )


def _binary_reference(value: object) -> BinaryArtifactReference:
    raw = _object(value, "binary artifact")
    _exact_keys(
        raw,
        {"kind", "source_relative_path", "state", "content_digest", "size_bytes"},
        "Binary artifact",
    )
    return BinaryArtifactReference(
        kind=_enum(raw["kind"], PredictionArtifactKind, "artifact kind"),
        source_relative_path=_text(raw["source_relative_path"], "source_relative_path"),
        state=_enum(raw["state"], PredictionArtifactState, "artifact state"),
        content_digest=_optional_digest(raw["content_digest"], "content_digest"),
        size_bytes=_optional_integer(raw["size_bytes"], "size_bytes"),
    )


def _feature_schema(value: object) -> PredictionFeatureSchema:
    raw = _object(value, "feature_schema")
    _exact_keys(
        raw,
        {
            "schema_id",
            "model_name",
            "model_type",
            "task",
            "modeling_schema_version",
            "feature_set_name",
            "ordered_columns",
        },
        "Feature schema",
    )
    return PredictionFeatureSchema(
        schema_id=_digest(_text(raw["schema_id"], "schema_id"), "schema_id"),
        model_name=_text(raw["model_name"], "model_name"),
        model_type=_text(raw["model_type"], "model_type"),
        task=_text(raw["task"], "task"),
        modeling_schema_version=_integer(raw["modeling_schema_version"], "modeling_schema_version"),
        feature_set_name=_text(raw["feature_set_name"], "feature_set_name"),
        ordered_columns=tuple(
            _text(item, "feature column")
            for item in _list(raw["ordered_columns"], "ordered_columns")
        ),
    )


def _post_processing(value: object) -> PredictionPostProcessingEvidence:
    raw = _object(value, "post_processing")
    _exact_keys(
        raw,
        {
            "registry_reference",
            "registry_entry_updated_at",
            "sigma",
            "sigma_source",
            "margin_std",
            "margin_std_source",
            "external_calibrator_state",
            "embedded_estimator_calibration",
        },
        "Post-processing evidence",
    )
    embedded = raw["embedded_estimator_calibration"]
    if not isinstance(embedded, bool):
        raise ValueError("embedded_estimator_calibration must be a boolean.")
    return PredictionPostProcessingEvidence(
        registry_reference=_source_reference(raw["registry_reference"]),
        registry_entry_updated_at=_optional_datetime(
            raw["registry_entry_updated_at"], "registry_entry_updated_at"
        ),
        sigma=_optional_float(raw["sigma"], "sigma"),
        sigma_source=_enum(raw["sigma_source"], CalibrationResolutionSource, "sigma_source"),
        margin_std=_optional_float(raw["margin_std"], "margin_std"),
        margin_std_source=_enum(
            raw["margin_std_source"], CalibrationResolutionSource, "margin_std_source"
        ),
        external_calibrator_state=_enum(
            raw["external_calibrator_state"],
            PredictionArtifactState,
            "external_calibrator_state",
        ),
        embedded_estimator_calibration=embedded,
    )


def _statistical_event(value: object) -> StatisticalPredictionEventEvidence:
    raw = _object(value, "statistical event")
    _exact_keys(
        raw,
        {
            "event_id",
            "game_id",
            "raw_feature_values",
            "transformed_feature_values",
            "raw_estimator_output",
            "post_estimator_output",
            "final_outputs",
        },
        "Statistical event",
    )
    return StatisticalPredictionEventEvidence(
        event_id=_text(raw["event_id"], "event_id"),
        game_id=_text(raw["game_id"], "game_id"),
        raw_feature_values=tuple(
            _float(item, "raw feature value")
            for item in _list(raw["raw_feature_values"], "raw_feature_values")
        ),
        transformed_feature_values=tuple(
            _float(item, "transformed feature value")
            for item in _list(raw["transformed_feature_values"], "transformed_feature_values")
        ),
        raw_estimator_output=_float(raw["raw_estimator_output"], "raw_estimator_output"),
        post_estimator_output=_float(raw["post_estimator_output"], "post_estimator_output"),
        final_outputs=_final_outputs(raw["final_outputs"]),
    )


def _elo_event(value: object) -> EloPredictionEventEvidence:
    raw = _object(value, "Elo event")
    _exact_keys(
        raw,
        {
            "event_id",
            "game_id",
            "season",
            "week",
            "away_team",
            "home_team",
            "away_elo",
            "home_elo",
            "formula_id",
            "divisor",
            "away_win_probability",
            "home_win_probability",
            "final_outputs",
        },
        "Elo event",
    )
    return EloPredictionEventEvidence(
        event_id=_text(raw["event_id"], "event_id"),
        game_id=_text(raw["game_id"], "game_id"),
        season=_text(raw["season"], "season"),
        week=_integer(raw["week"], "week"),
        away_team=_text(raw["away_team"], "away_team"),
        home_team=_text(raw["home_team"], "home_team"),
        away_elo=_float(raw["away_elo"], "away_elo"),
        home_elo=_float(raw["home_elo"], "home_elo"),
        formula_id=_text(raw["formula_id"], "formula_id"),
        divisor=_float(raw["divisor"], "divisor"),
        away_win_probability=_float(raw["away_win_probability"], "away_win_probability"),
        home_win_probability=_float(raw["home_win_probability"], "home_win_probability"),
        final_outputs=_final_outputs(raw["final_outputs"]),
    )


def _final_outputs(value: object) -> tuple[tuple[str, bool | int | float | str | None], ...]:
    rows = _list(value, "final_outputs")
    output: list[tuple[str, bool | int | float | str | None]] = []
    for item in rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each final output must be a two-item list.")
        output.append((_text(item[0], "final output name"), _scalar(item[1])))
    return tuple(output)


def _artifact_repo(path: Path) -> Path:
    resolved = path.resolve()
    marker = tuple(Path(_STORE_DIRECTORY).parts)
    parts = resolved.parts
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return Path(*parts[:index])
    raise ValueError("Prediction-input evidence path is outside the canonical store.")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _exact_keys(raw: dict[str, object], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ValueError(f"{label} keys do not match the current schema.")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return result


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _nonnegative_integer(value, label)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _optional_float(value: object, label: str) -> float | None:
    return None if value is None else _float(value, label)


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp string.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp string.") from exc


def _optional_datetime(value: object, label: str) -> datetime | None:
    return None if value is None else _datetime(value, label)


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _optional_digest(value: object, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _enum[T: str](value: object, enum_type: type[T], label: str) -> T:
    text = _text(value, label)
    try:
        # pyrefly: ignore [unnecessary-type-conversion]
        return enum_type(text)
    except ValueError as exc:
        raise ValueError(f"{label} contains an unsupported value: {text!r}.") from exc


def _scalar(value: object) -> bool | int | float | str | None:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ValueError("Final output value must be a JSON scalar.")
