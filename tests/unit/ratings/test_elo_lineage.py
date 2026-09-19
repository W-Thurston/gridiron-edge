"""Tests for strict Elo reconstruction lineage."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.lineage import (
    ELO_LINEAGE_SCHEMA_VERSION,
    EloArtifactReference,
    EloLineage,
    build_elo_lineage,
    elo_lineage_path,
    file_content_digest,
    load_elo_lineage,
    verify_current_elo_lineage,
    write_elo_lineage,
)

GENERATED_AT = datetime(
    2026,
    9,
    18,
    18,
    30,
    tzinfo=UTC,
)


def _games() -> pd.DataFrame:
    """Create representative persisted canonical games."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": "1999_01_A_B",
                "YEAR": "1999-2000",
                "WEEK_NUM": 1,
                "AWAY_TEAM": "Team A",
                "HOME_TEAM": "Team B",
                "AWAY_SCORE": 20,
                "HOME_SCORE": 24,
            },
            {
                "GAME_ID": "2026_01_B_A",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "AWAY_TEAM": "Team B",
                "HOME_TEAM": "Team A",
                "AWAY_SCORE": 17,
                "HOME_SCORE": 27,
            },
        ]
    )


def _elo() -> pd.DataFrame:
    """Create representative persisted canonical Elo state."""
    return pd.DataFrame(
        [
            {
                "NFL_TEAM": "Team A",
                "NFL_YEAR": "1999-2000",
                "NFL_WEEK": 1,
                "ELO": 1500.0,
            },
            {
                "NFL_TEAM": "Team B",
                "NFL_YEAR": "1999-2000",
                "NFL_WEEK": 1,
                "ELO": 1500.0,
            },
            {
                "NFL_TEAM": "Team A",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1540.0,
            },
            {
                "NFL_TEAM": "Team B",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1460.0,
            },
            {
                "NFL_TEAM": "Team A",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1550.0,
            },
            {
                "NFL_TEAM": "Team B",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1450.0,
            },
        ]
    )


def _write_artifacts(
    repo: Path,
) -> tuple[Path, Path]:
    """Write representative games and Elo artifacts."""
    games_path = dataset_path(repo, "games")
    elo_path = dataset_path(repo, "elo_state")

    games_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    elo_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _games().to_csv(
        games_path,
        index=False,
    )
    _elo().to_csv(
        elo_path,
        index=False,
    )

    return games_path, elo_path


def _write_valid_lineage(
    repo: Path,
) -> EloLineage:
    """Write lineage matching representative artifacts."""
    _write_artifacts(repo)

    lineage = build_elo_lineage(
        repo=repo,
        generated_at=GENERATED_AT,
    )
    write_elo_lineage(
        lineage,
        repo=repo,
    )
    return lineage


def test_builds_exact_games_and_elo_references(
    tmp_path: Path,
) -> None:
    games_path, elo_path = _write_artifacts(tmp_path)

    lineage = build_elo_lineage(
        repo=tmp_path,
        generated_at=GENERATED_AT,
    )

    assert lineage.schema_version == (ELO_LINEAGE_SCHEMA_VERSION)
    assert lineage.generated_at == GENERATED_AT

    assert lineage.source_games.relative_path == ("data/cleaned/NFL_wk_by_wk_cleaned.csv")
    assert lineage.source_games.content_digest == (file_content_digest(games_path))
    assert lineage.source_games.row_count == 2
    assert lineage.source_games.columns == tuple(_games().columns)
    assert lineage.source_games.first_season == "1999-2000"
    assert lineage.source_games.latest_season == "2026-2027"
    assert lineage.source_games.latest_week == 1

    assert lineage.elo_state.relative_path == ("data/cleaned/NFL_Team_Elo.csv")
    assert lineage.elo_state.content_digest == (file_content_digest(elo_path))
    assert lineage.elo_state.row_count == 6
    assert lineage.elo_state.columns == tuple(_elo().columns)
    assert lineage.elo_state.first_season == "1999-2000"
    assert lineage.elo_state.latest_season == "2026-2027"
    assert lineage.elo_state.latest_week == 2


def test_writes_and_loads_deterministic_lineage(
    tmp_path: Path,
) -> None:
    expected = _write_valid_lineage(tmp_path)

    first_bytes = elo_lineage_path(tmp_path).read_bytes()
    actual = load_elo_lineage(repo=tmp_path)

    write_elo_lineage(
        actual,
        repo=tmp_path,
    )
    second_bytes = elo_lineage_path(tmp_path).read_bytes()

    assert actual == expected
    assert first_bytes == second_bytes


def test_valid_current_artifacts_are_verified(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    assert verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_missing_lineage_is_unavailable(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "games",
        "elo_state",
    ],
)
def test_missing_referenced_artifact_is_unavailable(
    tmp_path: Path,
    artifact: str,
) -> None:
    _write_valid_lineage(tmp_path)
    dataset_path(tmp_path, artifact).unlink()

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_changed_games_content_invalidates_lineage(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    games_path = dataset_path(tmp_path, "games")
    games = pd.read_csv(games_path)
    games.loc[0, "HOME_SCORE"] = 25
    games.to_csv(
        games_path,
        index=False,
    )

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_changed_elo_content_invalidates_lineage(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    elo_path = dataset_path(tmp_path, "elo_state")
    elo = pd.read_csv(elo_path)
    elo.loc[0, "ELO"] = 1501.0
    elo.to_csv(
        elo_path,
        index=False,
    )

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_changed_row_count_invalidates_lineage(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    games_path = dataset_path(tmp_path, "games")
    games = pd.read_csv(games_path)
    games.iloc[:1].to_csv(
        games_path,
        index=False,
    )

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_changed_columns_invalidates_lineage(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    elo_path = dataset_path(tmp_path, "elo_state")
    elo = pd.read_csv(elo_path)
    elo["EXTRA"] = 1
    elo.to_csv(
        elo_path,
        index=False,
    )

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_changed_season_scope_invalidates_lineage(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)

    games_path = dataset_path(tmp_path, "games")
    games = pd.read_csv(games_path)
    games.loc[1, "YEAR"] = "2025-2026"
    games.to_csv(
        games_path,
        index=False,
    )

    assert not verify_current_elo_lineage(
        repo=tmp_path,
    )


def test_malformed_json_raises(
    tmp_path: Path,
) -> None:
    path = elo_lineage_path(tmp_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="malformed JSON",
    ):
        verify_current_elo_lineage(
            repo=tmp_path,
        )


def test_unsupported_schema_version_raises(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Elo lineage schema_version: 999",
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


def test_malformed_digest_raises(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_games"]["content_digest"] = "invalid"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="content_digest must be a lowercase SHA-256",
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


@pytest.mark.parametrize(
    "generated_at",
    [
        "not-a-timestamp",
        "2026-09-18T18:30:00",
    ],
)
def test_invalid_generated_at_raises(
    tmp_path: Path,
    generated_at: str,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generated_at"] = generated_at
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=("ISO-8601" if generated_at == "not-a-timestamp" else "timezone-aware"),
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        (
            "/tmp/NFL_Team_Elo.csv",
            "safe relative path",
        ),
        (
            "../NFL_Team_Elo.csv",
            "safe relative path",
        ),
        (
            "",
            "must be a nonempty string",
        ),
    ],
)
def test_invalid_artifact_path_raises(
    tmp_path: Path,
    relative_path: str,
    message: str,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["elo_state"]["relative_path"] = relative_path
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_row_count_raises(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="row_count must be a positive integer",
    ):
        EloArtifactReference(
            relative_path=("data/cleaned/NFL_Team_Elo.csv"),
            content_digest="a" * 64,
            row_count=value,
            columns=(
                "NFL_TEAM",
                "NFL_YEAR",
                "NFL_WEEK",
                "ELO",
            ),
            first_season="1999-2000",
            latest_season="2026-2027",
            latest_week=2,
        )


def test_boolean_row_count_is_rejected(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["elo_state"]["row_count"] = True
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="row_count must be an integer",
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


def test_unexpected_fields_are_rejected(
    tmp_path: Path,
) -> None:
    _write_valid_lineage(tmp_path)
    path = elo_lineage_path(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = "value"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="fields do not match schema",
    ):
        load_elo_lineage(
            repo=tmp_path,
        )


def test_absolute_canonical_reference_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="safe relative path",
    ):
        EloArtifactReference(
            relative_path="/tmp/NFL_Team_Elo.csv",
            content_digest="a" * 64,
            row_count=1,
            columns=(
                "NFL_TEAM",
                "NFL_YEAR",
                "NFL_WEEK",
                "ELO",
            ),
            first_season="1999-2000",
            latest_season="1999-2000",
            latest_week=1,
        )


def test_naive_generated_at_is_rejected() -> None:
    reference = EloArtifactReference(
        relative_path=("data/cleaned/NFL_wk_by_wk_cleaned.csv"),
        content_digest="a" * 64,
        row_count=1,
        columns=(
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "AWAY_TEAM",
            "HOME_TEAM",
            "AWAY_SCORE",
            "HOME_SCORE",
        ),
        first_season="1999-2000",
        latest_season="1999-2000",
        latest_week=1,
    )
    elo_reference = EloArtifactReference(
        relative_path="data/cleaned/NFL_Team_Elo.csv",
        content_digest="b" * 64,
        row_count=1,
        columns=(
            "NFL_TEAM",
            "NFL_YEAR",
            "NFL_WEEK",
            "ELO",
        ),
        first_season="1999-2000",
        latest_season="1999-2000",
        latest_week=1,
    )

    with pytest.raises(
        ValueError,
        match="generated_at must be timezone-aware",
    ):
        EloLineage(
            schema_version=ELO_LINEAGE_SCHEMA_VERSION,
            generated_at=datetime(2026, 9, 18, 18, 30),
            source_games=reference,
            elo_state=elo_reference,
        )
