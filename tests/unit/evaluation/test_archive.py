# tests/evaluation/test_archive.py

"""Tests for the prediction archive store."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from gridiron_edge.evaluation.archive import (
    _ARCHIVE_COLUMNS,
    build_archive_rows,
    load_prediction_log,
    write_archive_rows,
)


def _make_predictions(
    n: int = 3,
    season: str = "2025-2026",
    week: int = 1,
    game_id_prefix: str = "2025",
) -> pd.DataFrame:
    """Minimal predictions DataFrame matching build_predictions_df output."""
    return pd.DataFrame(
        {
            "GAME_ID": [f"{game_id_prefix}_0{i}_KC_LAC" for i in range(1, n + 1)],
            "GAME_DATE": ["2025-09-05"] * n,
            "AWAY_TEAM": ["Kansas City Chiefs"] * n,
            "HOME_TEAM": ["Los Angeles Chargers"] * n,
            "AWAY_TEAM_ELO": [1520.0] * n,
            "HOME_TEAM_ELO": [1480.0] * n,
            "AWAY_WIN_PROB": [0.55] * n,
            "HOME_WIN_PROB": [0.45] * n,
        }
    )


def test_build_archive_rows_schema() -> None:
    df = _make_predictions()
    rows = build_archive_rows(
        df,
        model_version="elo_v1",
        season="2025-2026",
        week=1,
    )
    assert list(rows.columns) == _ARCHIVE_COLUMNS
    assert len(rows) == 3
    assert (rows["model_version"] == "elo_v1").all()
    assert (rows["season"] == "2025-2026").all()
    assert (rows["week"] == 1).all()
    assert (rows["away_win_prob"] == 0.55).all()
    assert (rows["home_win_prob"] == 0.45).all()


def test_build_archive_rows_custom_timestamp() -> None:
    ts = datetime.datetime(2025, 9, 5, 12, 0, 0)
    rows = build_archive_rows(
        _make_predictions(),
        model_version="elo_v1",
        season="2025-2026",
        week=1,
        predicted_at=ts,
    )
    assert (rows["predicted_at"] == ts).all()


def test_write_archive_rows_creates_file(tmp_path: pytest.FixtureValue) -> None:
    rows = build_archive_rows(
        _make_predictions(),
        model_version="elo_v1",
        season="2025-2026",
        week=1,
    )
    # Patch the archive path to use tmp_path
    archive_file = tmp_path / "predictions_log.parquet"
    rows.to_parquet(archive_file, index=False)
    assert archive_file.exists()
    loaded = pd.read_parquet(archive_file)
    assert len(loaded) == 3


def test_write_archive_rows_deduplication(tmp_path: pytest.FixtureValue) -> None:
    """Re-writing the same game_id + model_version replaces rather than duplicates."""
    rows_v1 = build_archive_rows(
        _make_predictions(n=2),
        model_version="elo_v1",
        season="2025-2026",
        week=1,
        predicted_at=datetime.datetime(2025, 9, 1),
    )
    write_archive_rows(rows_v1, repo=tmp_path)

    # Write again with updated probabilities
    df2 = _make_predictions(n=2)
    df2["AWAY_WIN_PROB"] = 0.60
    df2["HOME_WIN_PROB"] = 0.40
    rows_v2 = build_archive_rows(
        df2,
        model_version="elo_v1",
        season="2025-2026",
        week=1,
        predicted_at=datetime.datetime(2025, 9, 5),
    )
    write_archive_rows(rows_v2, repo=tmp_path)

    log = load_prediction_log(repo=tmp_path)
    assert len(log) == 2  # not 4 — deduped
    assert (log["away_win_prob"] == 0.60).all()  # latest wins


def test_write_archive_rows_accumulates_different_models(
    tmp_path: pytest.FixtureValue,
) -> None:
    """Different model versions accumulate independently."""
    rows_v1 = build_archive_rows(
        _make_predictions(n=2),
        model_version="elo_v1",
        season="2025-2026",
        week=1,
    )
    rows_v2 = build_archive_rows(
        _make_predictions(n=2),
        model_version="elo_v2",
        season="2025-2026",
        week=1,
    )
    write_archive_rows(rows_v1, repo=tmp_path)
    write_archive_rows(rows_v2, repo=tmp_path)

    log = load_prediction_log(repo=tmp_path)
    assert len(log) == 4  # 2 games x 2 model versions
    assert set(log["model_version"].unique()) == {"elo_v1", "elo_v2"}


def test_load_prediction_log_filters_by_season(tmp_path: pytest.FixtureValue) -> None:
    rows_25 = build_archive_rows(
        _make_predictions(season="2025-2026", game_id_prefix="2025"),
        model_version="elo_v1",
        season="2025-2026",
        week=1,
    )
    rows_26 = build_archive_rows(
        _make_predictions(season="2026-2027", game_id_prefix="2026"),
        model_version="elo_v1",
        season="2026-2027",
        week=1,
    )
    write_archive_rows(rows_25, repo=tmp_path)
    write_archive_rows(rows_26, repo=tmp_path)

    filtered = load_prediction_log(season="2025-2026", repo=tmp_path)
    assert (filtered["season"] == "2025-2026").all()
    assert len(filtered) == 3


def test_load_prediction_log_empty_if_no_file(tmp_path: pytest.FixtureValue) -> None:
    log = load_prediction_log(repo=tmp_path)
    assert log.empty
    assert list(log.columns) == _ARCHIVE_COLUMNS
