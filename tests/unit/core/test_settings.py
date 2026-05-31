# tests/unit/core/test_settings.py
"""Tests for gridiron_edge.core.settings."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gridiron_edge.core.settings import Settings, get_settings


class TestSettingsDataclass:
    def test_is_frozen(self) -> None:
        settings: Settings = get_settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.repo_root = Path("/tmp")  # type: ignore[misc]

    def test_has_expected_fields(self) -> None:
        settings: Settings = get_settings()
        expected_fields: set[str] = {
            "repo_root",
            "owm_api_key",
            "data_raw",
            "data_cleaned",
            "data_modeling",
            "data_output",
        }
        actual_fields: set[str] = {f.name for f in dataclasses.fields(settings)}
        assert expected_fields <= actual_fields


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        assert isinstance(get_settings(), Settings)

    def test_repo_root_is_absolute(self) -> None:
        assert get_settings().repo_root.is_absolute()

    def test_data_dirs_under_repo_root(self) -> None:
        s: Settings = get_settings()
        for attr in ("data_raw", "data_cleaned", "data_modeling", "data_output"):
            path = getattr(s, attr)
            assert str(path).startswith(str(s.repo_root)), f"{attr} not under repo_root"

    def test_owm_api_key_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OWM_API_KEY", "test-key-12345")

        assert get_settings().owm_api_key == "test-key-12345"

    def test_owm_api_key_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OWM_API_KEY", raising=False)

        assert get_settings().owm_api_key is None


class TestEnsureDataDirs:
    def test_creates_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from gridiron_edge.core.settings import Settings, ensure_data_dirs

        settings = Settings(
            repo_root=tmp_path,
            owm_api_key=None,
            data_raw=tmp_path / "data" / "raw",
            data_cleaned=tmp_path / "data" / "cleaned",
            data_modeling=tmp_path / "data" / "modeling",
            data_output=tmp_path / "data" / "output",
        )
        ensure_data_dirs(settings)
        assert (tmp_path / "data" / "raw").is_dir()
        assert (tmp_path / "data" / "cleaned").is_dir()
        assert (tmp_path / "data" / "modeling").is_dir()
        assert (tmp_path / "data" / "output").is_dir()


class TestCurrentNflSeason:
    def test_returns_int(self) -> None:
        from gridiron_edge.core.settings import current_nfl_season

        assert isinstance(current_nfl_season(), int)

    def test_reasonable_year(self) -> None:
        from gridiron_edge.core.settings import current_nfl_season

        year: int = current_nfl_season()
        assert 2020 <= year <= 2040, f"Unexpected season year: {year}"
