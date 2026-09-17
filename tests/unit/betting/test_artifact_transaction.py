# tests/unit/betting/test_artifact_transaction.py
"""Tests for compensating betting-artifact restoration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridiron_edge.betting._artifact_transaction import (
    ArtifactSnapshot,
    restore_artifact,
    restore_artifacts,
    snapshot_artifact,
)


def test_snapshot_and_restore_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "data" / "betting" / "ledger.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")

    snapshot = snapshot_artifact(path)
    path.write_bytes(b"replacement")

    restore_artifact(snapshot)

    assert snapshot == ArtifactSnapshot(path, True, b"original")
    assert path.read_bytes() == b"original"


def test_restore_removes_artifact_absent_at_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "data" / "betting" / "ledger.parquet"
    snapshot = snapshot_artifact(path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new")

    restore_artifact(snapshot)

    assert snapshot == ArtifactSnapshot(path, False, None)
    assert not path.exists()


def test_invalid_existing_snapshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"

    with pytest.raises(
        RuntimeError,
        match="Existing artifact snapshot has no content",
    ):
        restore_artifact(ArtifactSnapshot(path, True, None))


def test_restore_artifacts_attempts_every_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ArtifactSnapshot(tmp_path / "first", True, None)
    second_path = tmp_path / "second"
    second_path.write_bytes(b"replacement")
    second = ArtifactSnapshot(second_path, False, None)

    with pytest.raises(RuntimeError, match="restoration incomplete"):
        restore_artifacts(
            (first, second),
            failure_message="restoration incomplete",
        )

    assert not second_path.exists()


def test_restore_cleans_temporary_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ledger.parquet"
    path.write_bytes(b"replacement")
    snapshot = ArtifactSnapshot(path, True, b"original")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(
        "gridiron_edge.betting._artifact_transaction.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="replace failed"):
        restore_artifact(snapshot)

    assert path.read_bytes() == b"replacement"
    assert list(tmp_path.glob("*.restore.tmp")) == []
    assert list(tmp_path.glob(".*.restore.tmp")) == []
