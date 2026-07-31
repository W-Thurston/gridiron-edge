# tests/unit/ingest/nflverse/test_games.py

"""Tests for nflverse historical-game ingestion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ingest.nflverse.games import (
    fetch_nflverse_games_refresh,
    refresh_nflverse_game_seasons,
)


def _raw_games_path(repo: Path) -> Path:
    """Return the registered nflverse games artifact path."""
    return dataset_path(
        repo,
        "games_raw_nflverse",
    )


def _write_existing_games(
    repo: Path,
    *,
    game_ids: list[str],
    seasons: list[int],
) -> Path:
    """Write representative existing raw game rows."""
    path = _raw_games_path(repo)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "week": [1] * len(game_ids),
        }
    ).to_parquet(
        path,
        index=False,
    )

    return path


def _schedule_result(
    frame: pd.DataFrame,
) -> MagicMock:
    """Return a mocked nflreadpy result."""
    result = MagicMock()
    result.to_pandas.return_value = frame
    return result


def test_selected_season_refresh_preserves_other_seasons(
    tmp_path: Path,
) -> None:
    raw_path = _write_existing_games(
        tmp_path,
        game_ids=[
            "2024_01_A_B",
            "2025_01_C_D",
        ],
        seasons=[
            2024,
            2025,
        ],
    )

    refreshed = pd.DataFrame(
        {
            "game_id": [
                "2025_01_E_F",
            ],
            "season": [
                2025,
            ],
            "week": [
                1,
            ],
        }
    )

    with patch(
        "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
        return_value=_schedule_result(refreshed),
    ) as mock_load:
        result = refresh_nflverse_game_seasons(
            seasons=[2025],
            repo=tmp_path,
        )

    assert result == raw_path

    stored = pd.read_parquet(raw_path)

    assert stored["game_id"].tolist() == [
        "2024_01_A_B",
        "2025_01_E_F",
    ]
    mock_load.assert_called_once_with([2025])


def test_multiple_season_refresh_replaces_only_requested_seasons(
    tmp_path: Path,
) -> None:
    raw_path = _write_existing_games(
        tmp_path,
        game_ids=[
            "2023_01_A_B",
            "2024_01_C_D",
            "2025_01_E_F",
        ],
        seasons=[
            2023,
            2024,
            2025,
        ],
    )

    refreshed = pd.DataFrame(
        {
            "game_id": [
                "2024_01_G_H",
                "2025_01_I_J",
            ],
            "season": [
                2024,
                2025,
            ],
            "week": [
                1,
                1,
            ],
        }
    )

    with patch(
        "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
        return_value=_schedule_result(refreshed),
    ) as mock_load:
        refresh_nflverse_game_seasons(
            seasons=[
                2025,
                2024,
            ],
            repo=tmp_path,
        )

    stored = pd.read_parquet(raw_path)

    assert stored["game_id"].tolist() == [
        "2023_01_A_B",
        "2024_01_G_H",
        "2025_01_I_J",
    ]
    mock_load.assert_called_once_with(
        [
            2024,
            2025,
        ]
    )


def test_selected_seasons_are_sorted_and_deduplicated(
    tmp_path: Path,
) -> None:
    refreshed = pd.DataFrame(
        {
            "game_id": [
                "2024_01_A_B",
                "2025_01_C_D",
            ],
            "season": [
                2024,
                2025,
            ],
            "week": [
                1,
                1,
            ],
        }
    )

    with patch(
        "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
        return_value=_schedule_result(refreshed),
    ) as mock_load:
        refresh_nflverse_game_seasons(
            seasons=[
                2025,
                2024,
                2025,
            ],
            repo=tmp_path,
        )

    mock_load.assert_called_once_with(
        [
            2024,
            2025,
        ]
    )


def test_selected_season_refresh_rejects_invalid_list(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
        ) as mock_load,
        pytest.raises(
            ValueError,
            match="No valid seasons to refresh",
        ),
    ):
        refresh_nflverse_game_seasons(
            seasons=[
                1998,
            ],
            repo=tmp_path,
        )

    mock_load.assert_not_called()


def test_selected_season_refresh_writes_new_artifact(
    tmp_path: Path,
) -> None:
    refreshed = pd.DataFrame(
        {
            "game_id": [
                "2025_01_A_B",
            ],
            "season": [
                2025,
            ],
            "week": [
                1,
            ],
        }
    )

    with patch(
        "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
        return_value=_schedule_result(refreshed),
    ):
        result = refresh_nflverse_game_seasons(
            seasons=[2025],
            repo=tmp_path,
        )

    assert result == _raw_games_path(tmp_path)
    assert result.exists()

    stored = pd.read_parquet(result)
    assert stored.to_dict(orient="records") == (refreshed.to_dict(orient="records"))


def test_selected_season_refresh_rejects_duplicate_game_ids(
    tmp_path: Path,
) -> None:
    refreshed = pd.DataFrame(
        {
            "game_id": [
                "2025_01_A_B",
                "2025_01_A_B",
            ],
            "season": [
                2025,
                2025,
            ],
            "week": [
                1,
                1,
            ],
        }
    )

    with (
        patch(
            "gridiron_edge.ingest.nflverse.games.nfl.load_schedules",
            return_value=_schedule_result(refreshed),
        ),
        pytest.raises(
            ValueError,
            match="duplicate game IDs",
        ),
    ):
        refresh_nflverse_game_seasons(
            seasons=[2025],
            repo=tmp_path,
        )


def test_single_season_refresh_delegates_to_selected_seasons(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "raw.parquet"

    with patch(
        "gridiron_edge.ingest.nflverse.games.refresh_nflverse_game_seasons",
        return_value=expected,
    ) as mock_refresh:
        result = fetch_nflverse_games_refresh(
            season=2025,
            repo=tmp_path,
        )

    assert result == expected
    mock_refresh.assert_called_once_with(
        seasons=[2025],
        repo=tmp_path,
    )
