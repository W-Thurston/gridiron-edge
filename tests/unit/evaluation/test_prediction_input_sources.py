# tests/unit/evaluation/test_prediction_input_sources.py
"""Tests for live prediction source revision and artifact capture."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess

import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionSourceState,
    SourceArtifactReference,
)
from gridiron_edge.evaluation.prediction_input_sources import (
    PREDICTION_SOURCE_SPECS,
    PredictionSourceSpec,
    capture_prediction_source_artifacts,
    file_identity,
    recapture_and_require_same_prediction_sources,
    resolve_clean_source_revision,
    source_artifact_path,
)

COMMIT = "a" * 40


def _completed(
    args: tuple[str, ...],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _write(repo: Path, relative_path: str, content: bytes) -> Path:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _minimal_specs() -> tuple[PredictionSourceSpec, ...]:
    return (
        PredictionSourceSpec("data/optional.bin", required=False),
        PredictionSourceSpec("data/required.bin", required=True),
    )


class TestSourceRevision:
    def test_resolves_clean_head_with_exact_commands(self, tmp_path: Path) -> None:
        calls: list[tuple[tuple[str, ...], Path]] = []

        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((command, repo))
            if command == ("git", "rev-parse", "HEAD"):
                return _completed(command, stdout=f"{COMMIT}\n")
            return _completed(command, stdout="")

        revision = resolve_clean_source_revision(
            tmp_path,
            run_command=run_command,
        )

        assert revision.commit == COMMIT
        assert revision.tracked_worktree_clean is True
        assert calls == [
            (("git", "rev-parse", "HEAD"), tmp_path.resolve()),
            (
                (
                    "git",
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=no",
                ),
                tmp_path.resolve(),
            ),
        ]

    @pytest.mark.parametrize(
        "status_output",
        [
            " M src/gridiron_edge/module.py\n",
            "M  src/gridiron_edge/module.py\n",
            "MM src/gridiron_edge/module.py\n",
            " D src/gridiron_edge/module.py\n",
        ],
    )
    def test_rejects_staged_or_unstaged_tracked_changes(
        self,
        tmp_path: Path,
        status_output: str,
    ) -> None:
        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            del repo
            if command == ("git", "rev-parse", "HEAD"):
                return _completed(command, stdout=f"{COMMIT}\n")
            return _completed(command, stdout=status_output)

        with pytest.raises(
            ValueError,
            match="Commit or stash staged and unstaged tracked changes",
        ):
            resolve_clean_source_revision(tmp_path, run_command=run_command)

    def test_untracked_files_are_excluded_by_command_contract(
        self,
        tmp_path: Path,
    ) -> None:
        observed_status_command: tuple[str, ...] | None = None

        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            nonlocal observed_status_command
            del repo
            if command == ("git", "rev-parse", "HEAD"):
                return _completed(command, stdout=f"{COMMIT}\n")
            observed_status_command = command
            return _completed(command, stdout="")

        resolve_clean_source_revision(tmp_path, run_command=run_command)

        assert observed_status_command == (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )

    @pytest.mark.parametrize(
        "head",
        [
            "",
            "abc123",
            "A" * 40,
            "g" * 40,
            "a" * 39,
            "a" * 41,
        ],
    )
    def test_rejects_invalid_head_identity(self, tmp_path: Path, head: str) -> None:
        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            del repo
            return _completed(command, stdout=f"{head}\n")

        with pytest.raises(ValueError, match="lowercase 40-character commit SHA"):
            resolve_clean_source_revision(tmp_path, run_command=run_command)

    def test_head_command_failure_is_explicit(self, tmp_path: Path) -> None:
        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            del repo
            return _completed(command, returncode=128, stderr="not a repository")

        with pytest.raises(
            ValueError,
            match="Could not resolve Git HEAD: not a repository",
        ):
            resolve_clean_source_revision(tmp_path, run_command=run_command)

    def test_status_command_failure_is_explicit(self, tmp_path: Path) -> None:
        def run_command(
            command: tuple[str, ...],
            repo: Path,
        ) -> subprocess.CompletedProcess[str]:
            del repo
            if command == ("git", "rev-parse", "HEAD"):
                return _completed(command, stdout=f"{COMMIT}\n")
            return _completed(command, returncode=1, stderr="status failed")

        with pytest.raises(
            ValueError,
            match="Could not inspect tracked worktree state: status failed",
        ):
            resolve_clean_source_revision(tmp_path, run_command=run_command)


class TestSourceInventory:
    def test_canonical_specs_are_sorted_unique_and_bounded(self) -> None:
        paths = tuple(spec.relative_path for spec in PREDICTION_SOURCE_SPECS)

        assert paths == tuple(sorted(set(paths)))
        assert paths == (
            "data/cleaned/NFL_Team_Elo.csv",
            "data/cleaned/NFL_Team_Elo.metadata.json",
            "data/cleaned/NFL_stadium_reference.csv",
            "data/cleaned/NFL_upcoming_schedule_rich.parquet",
            "data/cleaned/NFL_wk_by_wk_cleaned.csv",
            "data/cleaned/NFL_wk_by_wk_w_weather.csv",
            "data/cleaned/epa_by_game.parquet",
            "data/output/calibration/game_model_calibration.json",
        )

    def test_captures_exact_digest_size_and_optional_absence(
        self,
        tmp_path: Path,
    ) -> None:
        content = b"required source bytes\x00"
        _write(tmp_path, "data/required.bin", content)

        references = capture_prediction_source_artifacts(
            tmp_path,
            specs=_minimal_specs(),
        )

        assert references == (
            SourceArtifactReference(
                relative_path="data/optional.bin",
                state=PredictionSourceState.ABSENT,
                content_digest=None,
                size_bytes=None,
            ),
            SourceArtifactReference(
                relative_path="data/required.bin",
                state=PredictionSourceState.PRESENT,
                content_digest=sha256(content).hexdigest(),
                size_bytes=len(content),
            ),
        )

    def test_missing_required_source_fails_explicitly(self, tmp_path: Path) -> None:
        with pytest.raises(
            FileNotFoundError,
            match="Required prediction source artifact is missing",
        ):
            capture_prediction_source_artifacts(
                tmp_path,
                specs=_minimal_specs(),
            )

    def test_empty_spec_collection_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            capture_prediction_source_artifacts(tmp_path, specs=())

    def test_unsorted_specs_are_rejected(self, tmp_path: Path) -> None:
        specs = (
            PredictionSourceSpec("data/z.bin", required=False),
            PredictionSourceSpec("data/a.bin", required=False),
        )

        with pytest.raises(ValueError, match="ordered by unique relative path"):
            capture_prediction_source_artifacts(tmp_path, specs=specs)

    def test_duplicate_specs_are_rejected(self, tmp_path: Path) -> None:
        specs = (
            PredictionSourceSpec("data/a.bin", required=False),
            PredictionSourceSpec("data/a.bin", required=True),
        )

        with pytest.raises(ValueError, match="ordered by unique relative path"):
            capture_prediction_source_artifacts(tmp_path, specs=specs)

    @pytest.mark.parametrize(
        "relative_path",
        [
            "../outside.bin",
            "/absolute.bin",
            "",
            "   ",
        ],
    )
    def test_unsafe_source_paths_are_rejected(
        self,
        tmp_path: Path,
        relative_path: str,
    ) -> None:
        specs = (PredictionSourceSpec(relative_path, required=False),)

        with pytest.raises(ValueError, match="source path"):
            capture_prediction_source_artifacts(tmp_path, specs=specs)

    def test_file_identity_uses_exact_bytes(self, tmp_path: Path) -> None:
        content = b"abc\x00def\n"
        path = _write(tmp_path, "artifact.bin", content)

        assert file_identity(path) == (sha256(content).hexdigest(), len(content))

    def test_file_identity_rejects_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="artifact is missing"):
            file_identity(tmp_path / "missing.bin")


class TestSourceRecapture:
    def test_unchanged_sources_round_trip(self, tmp_path: Path) -> None:
        _write(tmp_path, "data/required.bin", b"stable")
        before = capture_prediction_source_artifacts(
            tmp_path,
            specs=_minimal_specs(),
        )

        after = recapture_and_require_same_prediction_sources(
            tmp_path,
            before,
            specs=_minimal_specs(),
        )

        assert after == before

    def test_changed_source_bytes_are_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "data/required.bin", b"before")
        before = capture_prediction_source_artifacts(
            tmp_path,
            specs=_minimal_specs(),
        )
        path.write_bytes(b"after")

        with pytest.raises(ValueError, match="changed during execution"):
            recapture_and_require_same_prediction_sources(
                tmp_path,
                before,
                specs=_minimal_specs(),
            )

    def test_optional_source_appearance_is_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, "data/required.bin", b"stable")
        before = capture_prediction_source_artifacts(
            tmp_path,
            specs=_minimal_specs(),
        )
        _write(tmp_path, "data/optional.bin", b"appeared")

        with pytest.raises(ValueError, match="changed during execution"):
            recapture_and_require_same_prediction_sources(
                tmp_path,
                before,
                specs=_minimal_specs(),
            )

    def test_required_source_disappearance_fails_explicitly(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "data/required.bin", b"stable")
        before = capture_prediction_source_artifacts(
            tmp_path,
            specs=_minimal_specs(),
        )
        path.unlink()

        with pytest.raises(FileNotFoundError, match="Required prediction source"):
            recapture_and_require_same_prediction_sources(
                tmp_path,
                before,
                specs=_minimal_specs(),
            )


class TestSourcePathResolution:
    def test_present_reference_resolves_canonical_path(self, tmp_path: Path) -> None:
        content = b"source"
        path = _write(tmp_path, "data/source.bin", content)
        reference = SourceArtifactReference(
            relative_path="data/source.bin",
            state=PredictionSourceState.PRESENT,
            content_digest=sha256(content).hexdigest(),
            size_bytes=len(content),
        )

        assert source_artifact_path(tmp_path, reference) == path.resolve()

    def test_absent_reference_returns_none(self, tmp_path: Path) -> None:
        reference = SourceArtifactReference(
            relative_path="data/optional.bin",
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        )

        assert source_artifact_path(tmp_path, reference) is None

    def test_malformed_absent_reference_is_rejected(self, tmp_path: Path) -> None:
        reference = SourceArtifactReference(
            relative_path="data/optional.bin",
            state=PredictionSourceState.ABSENT,
            content_digest="a" * 64,
            size_bytes=None,
        )

        with pytest.raises(ValueError, match="must not contain digest or size"):
            source_artifact_path(tmp_path, reference)

    def test_malformed_present_reference_is_rejected(self, tmp_path: Path) -> None:
        reference = SourceArtifactReference(
            relative_path="data/source.bin",
            state=PredictionSourceState.PRESENT,
            content_digest=None,
            size_bytes=None,
        )

        with pytest.raises(ValueError, match="requires digest and size"):
            source_artifact_path(tmp_path, reference)
