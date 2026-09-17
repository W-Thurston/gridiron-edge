# src/gridiron_edge/betting/_artifact_transaction.py
"""Rollback helpers for coordinated betting-artifact publication.

The betting domain persists the wager ledger and bankroll transaction ledger
as separate files. Higher-level operations that change both files snapshot
their prior bytes before publishing and restore those snapshots if any later
step fails.

These helpers provide compensating rollback, not a cross-file atomic commit.
Callers remain responsible for holding the appropriate writer lock across the
complete snapshot, publication, validation, and restoration sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import uuid


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Exact prior state of one persisted artifact."""

    path: Path
    existed: bool
    content: bytes | None


def snapshot_artifact(path: Path) -> ArtifactSnapshot:
    """Capture whether an artifact exists and its exact bytes.

    Args:
        path: Artifact path to snapshot.

    Returns:
        An immutable snapshot suitable for compensating restoration.
    """
    existed = path.exists()
    return ArtifactSnapshot(
        path=path,
        existed=existed,
        content=path.read_bytes() if existed else None,
    )


def restore_artifact(snapshot: ArtifactSnapshot) -> None:
    """Restore one artifact to its exact snapshotted state.

    An artifact that did not exist when snapshotted is removed. An existing
    artifact is restored through a colocated temporary file and atomic rename,
    so readers do not observe partially restored bytes.

    Args:
        snapshot: Prior artifact state returned by :func:`snapshot_artifact`.

    Raises:
        RuntimeError: If an existing artifact snapshot has no captured bytes.
    """
    path = snapshot.path

    if not snapshot.existed:
        path.unlink(missing_ok=True)
        return

    if snapshot.content is None:
        raise RuntimeError(f"Existing artifact snapshot has no content: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.restore.tmp")
    try:
        temporary.write_bytes(snapshot.content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_artifacts(
    snapshots: tuple[ArtifactSnapshot, ...],
    *,
    failure_message: str,
) -> None:
    """Restore multiple snapshots and report incomplete restoration.

    Every restoration is attempted even if an earlier restoration fails. If
    one or more restorations fail, a single ``RuntimeError`` is raised from the
    first restoration error after all snapshots have been attempted.

    Args:
        snapshots: Artifact snapshots to restore in caller-defined order.
        failure_message: Error message used when any restoration is incomplete.

    Raises:
        RuntimeError: If at least one artifact could not be restored.
    """
    restoration_errors: list[Exception] = []

    for snapshot in snapshots:
        try:
            restore_artifact(snapshot)
        except Exception as error:
            restoration_errors.append(error)

    if restoration_errors:
        raise RuntimeError(failure_message) from restoration_errors[0]
