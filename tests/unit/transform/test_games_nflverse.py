# tests/unit/transform/test_games_nflverse.py
"""Tests for nflverse historical-game schema mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.transform.clean.games_nflverse import (
    _validate_home_away_games,
    clean_nflverse_games,
)


def _raw_game(
    *,
    game_id: str = "2025_01_PHI_GB",
    away_team: str = "PHI",
    home_team: str = "GB",
    away_score: int = 20,
    home_score: int = 17,
    location: str = "Home",
) -> dict[str, object]:
    """Return one completed nflverse schedule row."""
    return {
        "game_id": game_id,
        "season": 2025,
        "game_type": "REG",
        "week": 1,
        "gameday": "2025-09-04",
        "weekday": "Thursday",
        "gametime": "20:20",
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_score,
        "home_score": home_score,
        "location": location,
        "result": home_score - away_score,
        "stadium": "Example Stadium",
        "roof": "outdoors",
        "surface": "grass",
        "spread_line": 3.0,
        "total_line": 47.5,
        "div_game": 0,
    }


def _write_raw_games(
    repo: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write synthetic nflverse rows to the registered raw path."""
    path = dataset_path(repo, "games_raw_nflverse")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _cleaned_games(repo: Path) -> pd.DataFrame:
    """Run the cleaner and return its output."""
    clean_nflverse_games(repo=repo)
    return pd.read_csv(dataset_path(repo, "games"))


def _valid_home_away_games() -> pd.DataFrame:
    """Return one valid cleaned home/away game."""
    return pd.DataFrame(
        {
            "GAME_ID": ["2025_01_PHI_GB"],
            "AWAY_TEAM": ["Philadelphia Eagles"],
            "HOME_TEAM": ["Green Bay Packers"],
            "AWAY_SCORE": [20],
            "HOME_SCORE": [17],
            "IS_NEUTRAL_SITE": [0],
            "WINNER": ["Philadelphia Eagles"],
            "LOSER": ["Green Bay Packers"],
            "PTS_WINNER": [20],
            "PTS_LOSER": [17],
            "WIN_OR_TIE": [1.0],
        }
    )


def test_preserves_home_away_identity_for_away_win(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            _raw_game(
                away_score=20,
                home_score=17,
            )
        ],
    )

    games = _cleaned_games(tmp_path)
    row = games.iloc[0]

    assert row["AWAY_TEAM"] == "Philadelphia Eagles"
    assert row["HOME_TEAM"] == "Green Bay Packers"
    assert row["AWAY_SCORE"] == 20
    assert row["HOME_SCORE"] == 17
    assert row["IS_NEUTRAL_SITE"] == 0

    assert row["WINNER"] == "Philadelphia Eagles"
    assert row["LOSER"] == "Green Bay Packers"
    assert row["GAME_LOCATION"] == "@"


def test_preserves_home_away_identity_for_home_win(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            _raw_game(
                away_score=17,
                home_score=24,
            )
        ],
    )

    games = _cleaned_games(tmp_path)
    row = games.iloc[0]

    assert row["AWAY_TEAM"] == "Philadelphia Eagles"
    assert row["HOME_TEAM"] == "Green Bay Packers"
    assert row["AWAY_SCORE"] == 17
    assert row["HOME_SCORE"] == 24
    assert row["IS_NEUTRAL_SITE"] == 0

    assert row["WINNER"] == "Green Bay Packers"
    assert row["LOSER"] == "Philadelphia Eagles"
    assert row["GAME_LOCATION"] == "H"


def test_preserves_source_orientation_for_neutral_game(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            _raw_game(
                game_id="2025_21_PHI_GB",
                away_score=27,
                home_score=24,
                location="Neutral",
            )
        ],
    )

    games = _cleaned_games(tmp_path)
    row = games.iloc[0]

    assert row["AWAY_TEAM"] == "Philadelphia Eagles"
    assert row["HOME_TEAM"] == "Green Bay Packers"
    assert row["AWAY_SCORE"] == 27
    assert row["HOME_SCORE"] == 24
    assert row["IS_NEUTRAL_SITE"] == 1
    assert row["GAME_LOCATION"] == "N"

    # The winner remains result-oriented, while Away/Home remains stable.
    assert row["WINNER"] == "Philadelphia Eagles"


def test_preserves_source_orientation_for_tie(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            _raw_game(
                away_score=21,
                home_score=21,
            )
        ],
    )

    games = _cleaned_games(tmp_path)
    row = games.iloc[0]

    assert row["AWAY_TEAM"] == "Philadelphia Eagles"
    assert row["HOME_TEAM"] == "Green Bay Packers"
    assert row["AWAY_SCORE"] == 21
    assert row["HOME_SCORE"] == 21
    assert row["WIN_OR_TIE"] == pytest.approx(0.5)

    # Existing historical convention remains separate from orientation.
    assert row["WINNER"] == "Green Bay Packers"
    assert row["LOSER"] == "Philadelphia Eagles"


def test_empty_first_run_writes_extended_schema(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            {
                **_raw_game(),
                "away_score": None,
                "home_score": None,
                "result": None,
            }
        ],
    )

    path = clean_nflverse_games(repo=tmp_path)
    games = pd.read_csv(path)

    assert games.empty
    assert list(games.columns[:10]) == [
        "GAME_ID",
        "WEEK_NUM",
        "GAME_DAY_OF_WEEK",
        "GAME_DATE",
        "GAMETIME",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_SCORE",
        "HOME_SCORE",
        "IS_NEUTRAL_SITE",
    ]


def test_empty_refresh_does_not_clobber_existing_history(
    tmp_path: Path,
) -> None:
    _write_raw_games(
        tmp_path,
        [
            {
                **_raw_game(),
                "away_score": None,
                "home_score": None,
                "result": None,
            }
        ],
    )

    cleaned_path = dataset_path(tmp_path, "games")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(
        {
            "GAME_ID": ["existing-game"],
            "AWAY_TEAM": ["Away"],
            "HOME_TEAM": ["Home"],
        }
    )
    existing.to_csv(cleaned_path, index=False)

    result = clean_nflverse_games(repo=tmp_path)

    assert result == cleaned_path
    loaded = pd.read_csv(cleaned_path)
    assert loaded.to_dict(orient="records") == (existing.to_dict(orient="records"))


def test_home_away_validation_rejects_duplicate_game_ids() -> None:
    games = pd.concat(
        [
            _valid_home_away_games(),
            _valid_home_away_games(),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game IDs",
    ):
        _validate_home_away_games(games)


def test_home_away_validation_rejects_same_team() -> None:
    games = _valid_home_away_games()
    games.loc[0, "HOME_TEAM"] = games.loc[0, "AWAY_TEAM"]

    with pytest.raises(
        ValueError,
        match="Away and home team must differ",
    ):
        _validate_home_away_games(games)


def test_home_away_validation_rejects_score_mismatch() -> None:
    games = _valid_home_away_games()
    games.loc[0, "PTS_WINNER"] = 21

    with pytest.raises(
        ValueError,
        match="PTS_WINNER does not reconcile",
    ):
        _validate_home_away_games(games)


def test_home_away_validation_rejects_invalid_tie_state() -> None:
    games = _valid_home_away_games()
    games.loc[0, "WIN_OR_TIE"] = 0.5

    with pytest.raises(
        ValueError,
        match="scores do not reconcile with WIN_OR_TIE",
    ):
        _validate_home_away_games(games)
