# tests/conftest.py
"""Root conftest - shared configuration, fixtures, and auto-markers.

Markers are applied automatically based on the test file's directory:
  tests/unit/       → @pytest.mark.unit
  tests/integration → @pytest.mark.integration
  tests/e2e/        → @pytest.mark.e2e

Tests outside a marker directory are treated as unit tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Terminal color variables change how Typer and Rich render help and error text (typer reads
# them when it is first imported), which broke assertions on plain output. Clear them before
# anything under test is imported so results do not depend on the developer's or CI's shell.
for _color_variable in ("FORCE_COLOR", "PY_COLORS", "GITHUB_ACTIONS"):
    os.environ.pop(_color_variable, None)
os.environ["NO_COLOR"] = "1"

# ---------------------------------------------------------------------------
# Auto-apply markers by directory
# ---------------------------------------------------------------------------

_MARKER_DIRS: dict[str, str] = {
    "unit": "unit",
    "integration": "integration",
    "e2e": "e2e",
}


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-tag tests with markers based on their directory."""
    tests_root = Path(__file__).parent

    for item in items:
        rel = Path(item.fspath).relative_to(tests_root)
        parts = rel.parts  # e.g. ("unit", "features", "test_rest.py")

        matched = False
        for marker_name, dir_name in _MARKER_DIRS.items():
            if parts and parts[0] == dir_name:
                item.add_marker(getattr(pytest.mark, marker_name))
                matched = True
                break

        # Tests outside a marker directory default to the unit marker.
        if not matched:
            item.add_marker(pytest.mark.unit)
