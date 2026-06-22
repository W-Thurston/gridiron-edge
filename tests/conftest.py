# tests/conftest.py
"""Root conftest - shared configuration, fixtures, and auto-markers.

Markers are applied automatically based on the test file's directory:
  tests/unit/       → @pytest.mark.unit
  tests/integration → @pytest.mark.integration
  tests/e2e/        → @pytest.mark.e2e

Existing unmarked tests (during migration) are treated as unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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

        # During migration: unmarked top-level tests get 'unit' by default
        if not matched:
            item.add_marker(pytest.mark.unit)
