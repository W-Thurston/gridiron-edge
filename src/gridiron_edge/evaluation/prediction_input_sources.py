# src/gridiron_edge/evaluation/prediction_input_sources.py
"""Committed source revision and artifact identities for live prediction evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from typing import Final

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    require_same_source_artifacts,
)

_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_READ_CHUNK_SIZE: Final[int] = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PredictionSourceSpec:
    """Canonical source path and whether live evidence requires its presence."""

    relative_path: str
    required: bool


PREDICTION_SOURCE_SPECS: Final[tuple[PredictionSourceSpec, ...]] = (
    PredictionSourceSpec("data/cleaned/NFL_Team_Elo.csv", required=True),
    PredictionSourceSpec("data/cleaned/NFL_Team_Elo.metadata.json", required=True),
    PredictionSourceSpec("data/cleaned/NFL_stadium_reference.csv", required=False),
    PredictionSourceSpec("data/cleaned/NFL_upcoming_schedule_rich.parquet", required=True),
    PredictionSourceSpec("data/cleaned/NFL_wk_by_wk_cleaned.csv", required=True),
    PredictionSourceSpec("data/cleaned/NFL_wk_by_wk_w_weather.csv", required=False),
    PredictionSourceSpec("data/cleaned/epa_by_game.parquet", required=False),
    PredictionSourceSpec(
        "data/output/calibration/game_model_calibration.json",
        required=False,
    ),
)

RunCommand = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def resolve_clean_source_revision(
    repo: Path,
    *,
    run_command: RunCommand | None = None,
) -> SourceRevision:
    """Resolve Git HEAD and reject staged or unstaged tracked changes.

    Untracked and ignored files are intentionally excluded from the status
    command. Operational artifacts under ``data/`` therefore do not make the
    tracked worktree dirty.
    """
    runner = run_command or _run_git
    resolved_repo = repo.resolve()
    head_result = runner(("git", "rev-parse", "HEAD"), resolved_repo)
    _require_success(head_result, "resolve Git HEAD")
    commit = head_result.stdout.strip()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("Git HEAD must be a lowercase 40-character commit SHA.")

    status_result = runner(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ),
        resolved_repo,
    )
    _require_success(status_result, "inspect tracked worktree state")
    if status_result.stdout.strip():
        raise ValueError(
            "Live weekly prediction requires a clean tracked worktree. "
            "Commit or stash staged and unstaged tracked changes before retrying."
        )

    return SourceRevision(
        commit=commit,
        tracked_worktree_clean=True,
    )


def capture_prediction_source_artifacts(
    repo: Path,
    *,
    specs: tuple[PredictionSourceSpec, ...] = PREDICTION_SOURCE_SPECS,
) -> tuple[SourceArtifactReference, ...]:
    """Capture exact identities and explicit absence for bounded live sources."""
    _validate_specs(specs)
    resolved_repo = repo.resolve()
    references: list[SourceArtifactReference] = []
    for spec in specs:
        path = _resolve_source_path(resolved_repo, spec.relative_path)
        if not path.is_file():
            if spec.required:
                raise FileNotFoundError(f"Required prediction source artifact is missing: {path}")
            references.append(
                SourceArtifactReference(
                    relative_path=spec.relative_path,
                    state=PredictionSourceState.ABSENT,
                    content_digest=None,
                    size_bytes=None,
                )
            )
            continue

        content_digest, size_bytes = file_identity(path)
        references.append(
            SourceArtifactReference(
                relative_path=spec.relative_path,
                state=PredictionSourceState.PRESENT,
                content_digest=content_digest,
                size_bytes=size_bytes,
            )
        )
    return tuple(references)


def recapture_and_require_same_prediction_sources(
    repo: Path,
    before: tuple[SourceArtifactReference, ...],
    *,
    specs: tuple[PredictionSourceSpec, ...] = PREDICTION_SOURCE_SPECS,
) -> tuple[SourceArtifactReference, ...]:
    """Recapture current source identities and reject any execution-time drift."""
    after = capture_prediction_source_artifacts(repo, specs=specs)
    require_same_source_artifacts(before, after)
    return after


def file_identity(path: Path) -> tuple[str, int]:
    """Return SHA-256 and byte size for exact persisted file bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction source artifact is missing: {path}")
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def source_artifact_path(
    repo: Path,
    reference: SourceArtifactReference,
) -> Path | None:
    """Resolve one present source reference or return None for explicit absence."""
    path = _resolve_source_path(repo.resolve(), reference.relative_path)
    if reference.state is PredictionSourceState.ABSENT:
        if reference.content_digest is not None or reference.size_bytes is not None:
            raise ValueError("Absent source reference must not contain digest or size.")
        return None
    if reference.content_digest is None or reference.size_bytes is None:
        raise ValueError("Present source reference requires digest and size.")
    return path


def _run_git(command: Sequence[str], repo: Path) -> subprocess.CompletedProcess[str]:
    """Run one noninteractive Git command without shell interpretation."""
    return subprocess.run(
        list(command),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_success(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode == 0:
        return
    detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
    raise ValueError(f"Could not {action}: {detail}")


def _validate_specs(specs: tuple[PredictionSourceSpec, ...]) -> None:
    if not specs:
        raise ValueError("Prediction source specifications must not be empty.")
    paths: list[str] = []
    for spec in specs:
        if not isinstance(spec, PredictionSourceSpec):
            raise TypeError("Prediction source specifications contain an invalid value.")
        _safe_relative_path(spec.relative_path)
        if not isinstance(spec.required, bool):
            raise ValueError("Prediction source required state must be a boolean.")
        paths.append(spec.relative_path)
    if tuple(paths) != tuple(sorted(set(paths))):
        raise ValueError(
            "Prediction source specifications must be ordered by unique relative path."
        )


def _resolve_source_path(repo: Path, relative_path: str) -> Path:
    normalized = _safe_relative_path(relative_path)
    path = (repo / normalized).resolve()
    try:
        path.relative_to(repo)
    except ValueError as exc:
        raise ValueError("Prediction source path escapes repository root.") from exc
    return path


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Prediction source path must be a nonempty string.")
    normalized = value.strip()
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Prediction source path must be repository-relative and safe.")
    return normalized
