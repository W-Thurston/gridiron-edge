# tests/integration/conftest.py
"""Integration test fixtures.

Uses MiniRepoBuilder for composable, explicit test data setup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.repos import MiniRepoBuilder


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """Minimal repo tree with games + stadiums for pipeline integration tests."""
    return MiniRepoBuilder(tmp_path).with_games().with_stadiums().build()


@pytest.fixture
def mini_repo_with_elo(tmp_path: Path) -> Path:
    """Minimal repo with games, stadiums, and pre-computed Elo state."""
    return MiniRepoBuilder(tmp_path).with_games().with_stadiums().with_elo_state().build()


@pytest.fixture
def mini_repo_full(tmp_path: Path) -> Path:
    """Repo with all datasets populated — for pipeline-wide tests."""
    return (
        MiniRepoBuilder(tmp_path)
        .with_games()
        .with_stadiums()
        .with_elo_state()
        .with_epa_by_game()
        .with_weather()
        .build()
    )
