"""Tests for cli/main.py stage staleness checking."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


class TestStageStalenessCheck:
    """Verify _check_stage_staleness warns on stale upstream data (main/C1)."""

    def test_warns_when_input_newer_than_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from gridiron_edge.cli.main import _check_stage_staleness

        # Create the file structure expected by the dependency map.
        raw_dir = tmp_path / "data" / "raw"
        cleaned_dir = tmp_path / "data" / "cleaned"
        raw_dir.mkdir(parents=True)
        cleaned_dir.mkdir(parents=True)

        input_path = raw_dir / "games.parquet"
        output_path = cleaned_dir / "games.csv"

        # Create output first (older), then input (newer).
        output_path.write_text("old output")
        import time

        time.sleep(0.05)
        input_path.write_text("new input")

        # Force the helper to look at tmp_path.
        from gridiron_edge.core import settings as settings_mod

        class FakeSettings:
            repo_root = tmp_path

        monkeypatch.setattr(settings_mod, "get_settings", FakeSettings)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-games"})

        warnings = [r for r in caplog.records if "older than its current output" in r.getMessage()]
        assert len(warnings) == 1

    def test_no_warning_when_output_is_newer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from gridiron_edge.cli.main import _check_stage_staleness

        raw_dir = tmp_path / "data" / "raw"
        cleaned_dir = tmp_path / "data" / "cleaned"
        raw_dir.mkdir(parents=True)
        cleaned_dir.mkdir(parents=True)

        input_path = raw_dir / "games.parquet"
        output_path = cleaned_dir / "games.csv"

        # Create input first, then output (newer).
        input_path.write_text("input")
        import time

        time.sleep(0.05)
        output_path.write_text("output")

        from gridiron_edge.core import settings as settings_mod

        class FakeSettings:
            repo_root = tmp_path

        monkeypatch.setattr(settings_mod, "get_settings", FakeSettings)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-games"})

        warnings = [r for r in caplog.records if "older than its current output" in r.getMessage()]
        assert len(warnings) == 0

    def test_no_warning_when_files_dont_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from gridiron_edge.cli.main import _check_stage_staleness

        # Empty data dir — first run scenario.
        from gridiron_edge.core import settings as settings_mod

        class FakeSettings:
            repo_root = tmp_path

        monkeypatch.setattr(settings_mod, "get_settings", FakeSettings)

        caplog.set_level(logging.WARNING, logger="gridiron_edge.cli.main")
        _check_stage_staleness(active={"clean-games"})

        warnings = [r for r in caplog.records if "older than its current output" in r.getMessage()]
        assert len(warnings) == 0
