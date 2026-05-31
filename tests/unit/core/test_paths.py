# tests/unit/core/test_paths.py
"""Tests for gridiron_edge.core.paths."""

from __future__ import annotations

from pathlib import Path

from gridiron_edge.core.paths import repo_root


class TestRepoRoot:
    def test_returns_path(self) -> None:
        assert isinstance(repo_root(), Path)

    def test_is_absolute(self) -> None:
        assert repo_root().is_absolute()

    def test_directory_exists(self) -> None:
        assert repo_root().is_dir()

    def test_contains_pyproject_toml(self) -> None:
        assert (repo_root() / "pyproject.toml").is_file()

    def test_contains_src_directory(self) -> None:
        assert (repo_root() / "src").is_dir()

    def test_consistent_across_calls(self) -> None:
        assert repo_root() == repo_root()
