"""Tests for the stadium synchronization CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# pyrefly: ignore [missing-import]
import typer
from typer.testing import CliRunner

from gridiron_edge.cli.stadiums import stadiums_app

runner = CliRunner()


def _app() -> typer.Typer:
    app = typer.Typer()
    app.add_typer(stadiums_app, name="stadiums")
    return app


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ISSUE": ["missing_franchise_origin", "unresolved_game_site"],
            "HOME_TEAM": ["Kansas City Chiefs", pd.NA],
            "YEAR": ["2026-2027", "2026-2027"],
            "STADIUM": [pd.NA, "New Stadium"],
            "GAME_COUNT": [17, 1],
        }
    )


def _updates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ACTION": ["carry_forward", "unresolved"],
            "REVIEW_STATUS": ["proposed", "unresolved"],
            "HOME_TEAM": ["Kansas City Chiefs", pd.NA],
            "YEAR": ["2026-2027", "2026-2027"],
            "SOURCE_YEAR": ["2025-2026", pd.NA],
            "SOURCE_STADIUM": ["Arrowhead Stadium", pd.NA],
            "STADIUM": ["Arrowhead Stadium", "New Stadium"],
            "LATITUDE": [39.0, pd.NA],
            "LONGITUDE": [-94.0, pd.NA],
            "ROOF": ["outdoors", pd.NA],
            "SURFACE": ["grass", pd.NA],
            "ALTITUDE": [274.0, pd.NA],
            "NOTE": ["carry", "review"],
        }
    )


@patch("gridiron_edge.cli.stadiums.audit_stadium_coverage", return_value=pd.DataFrame())
@patch("gridiron_edge.cli.stadiums.load_schedule_upcoming_rich")
@patch("gridiron_edge.cli.stadiums.load_stadiums")
@patch("gridiron_edge.cli.stadiums.get_settings")
def test_clean_audit_exits_successfully(
    settings: MagicMock,
    _stadiums: MagicMock,
    _schedule: MagicMock,
    audit: MagicMock,
    tmp_path: Path,
) -> None:
    settings.return_value.repo_root = tmp_path
    result = runner.invoke(
        _app(),
        ["stadiums", "audit", "--season", "2026-2027"],
    )
    assert result.exit_code == 0
    audit.assert_called_once()


@patch("gridiron_edge.cli.stadiums.audit_stadium_coverage", return_value=_audit())
@patch("gridiron_edge.cli.stadiums.load_schedule_upcoming_rich")
@patch("gridiron_edge.cli.stadiums.load_stadiums")
@patch("gridiron_edge.cli.stadiums.get_settings")
def test_incomplete_audit_exits_nonzero(
    settings: MagicMock,
    _stadiums: MagicMock,
    _schedule: MagicMock,
    _audit_mock: MagicMock,
    tmp_path: Path,
) -> None:
    settings.return_value.repo_root = tmp_path
    result = runner.invoke(
        _app(),
        ["stadiums", "audit", "--season", "2026-2027"],
    )
    assert result.exit_code == 1
    assert "missing_franchise_origin" in result.output


@patch("gridiron_edge.cli.stadiums.prepare_stadium_updates", return_value=_updates())
@patch("gridiron_edge.cli.stadiums.load_stadium_aliases")
@patch("gridiron_edge.cli.stadiums.load_schedule_upcoming_rich")
@patch("gridiron_edge.cli.stadiums.load_stadiums")
@patch("gridiron_edge.cli.stadiums.get_settings")
def test_prepare_writes_review_only(
    settings: MagicMock,
    _stadiums: MagicMock,
    _schedule: MagicMock,
    _aliases: MagicMock,
    _prepare: MagicMock,
    tmp_path: Path,
) -> None:
    settings.return_value.repo_root = tmp_path
    output = tmp_path / "review.csv"
    result = runner.invoke(
        _app(),
        [
            "stadiums",
            "prepare",
            "--season",
            "2026-2027",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    written = pd.read_csv(output)
    assert written["REVIEW_STATUS"].tolist() == ["proposed", "unresolved"]
    assert not (tmp_path / "data/cleaned/NFL_stadium_reference.csv").exists()


@patch("gridiron_edge.cli.stadiums.audit_stadium_coverage", return_value=pd.DataFrame())
@patch("gridiron_edge.cli.stadiums.apply_approved_stadium_updates")
@patch("gridiron_edge.cli.stadiums.load_schedule_upcoming_rich")
@patch("gridiron_edge.cli.stadiums.load_stadiums")
@patch("gridiron_edge.cli.stadiums.get_settings")
def test_apply_calls_atomic_service_and_reaudits(
    settings: MagicMock,
    _stadiums: MagicMock,
    _schedule: MagicMock,
    apply: MagicMock,
    _audit_mock: MagicMock,
    tmp_path: Path,
) -> None:
    settings.return_value.repo_root = tmp_path
    _stadiums.return_value = pd.DataFrame(
        {
            "HOME_TEAM": ["Kansas City Chiefs"],
        }
    )
    apply.return_value = pd.DataFrame(
        {
            "HOME_TEAM": [
                "Kansas City Chiefs",
                "Buffalo Bills",
            ],
        }
    )
    review = _updates()
    review.loc[0, "REVIEW_STATUS"] = "approved"
    update_path = tmp_path / "review.csv"
    review.to_csv(update_path, index=False)

    result = runner.invoke(
        _app(),
        [
            "stadiums",
            "apply",
            "--updates",
            str(update_path),
            "--season",
            "2026-2027",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "approved rows applied: 1" in result.output
    apply.assert_called_once()


def test_apply_rejects_missing_review_file(tmp_path: Path) -> None:
    result = runner.invoke(
        _app(),
        [
            "stadiums",
            "apply",
            "--updates",
            str(tmp_path / "missing.csv"),
            "--season",
            "2026-2027",
        ],
    )
    assert result.exit_code != 0


def test_stadiums_group_is_registered_on_main_app() -> None:
    from gridiron_edge.cli.main import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stadiums" in result.output


@patch(
    "gridiron_edge.cli.stadiums.audit_stadium_coverage",
    return_value=pd.DataFrame(),
)
@patch(
    "gridiron_edge.cli.stadiums.apply_approved_stadium_updates",
)
@patch(
    "gridiron_edge.cli.stadiums.load_schedule_upcoming_rich",
)
@patch(
    "gridiron_edge.cli.stadiums.load_stadiums",
)
@patch(
    "gridiron_edge.cli.stadiums.get_settings",
)
def test_apply_reports_zero_for_identical_reapplication(
    settings: MagicMock,
    stadiums: MagicMock,
    _schedule: MagicMock,
    apply: MagicMock,
    _audit_mock: MagicMock,
    tmp_path: Path,
) -> None:
    settings.return_value.repo_root = tmp_path

    current = pd.DataFrame(
        {
            "HOME_TEAM": ["Kansas City Chiefs"],
        }
    )
    stadiums.return_value = current
    apply.return_value = current.copy()

    review = _updates()
    review.loc[0, "REVIEW_STATUS"] = "approved"
    update_path = tmp_path / "review.csv"
    review.to_csv(
        update_path,
        index=False,
    )

    result = runner.invoke(
        _app(),
        [
            "stadiums",
            "apply",
            "--updates",
            str(update_path),
            "--season",
            "2026-2027",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "approved rows applied: 0" in result.output
