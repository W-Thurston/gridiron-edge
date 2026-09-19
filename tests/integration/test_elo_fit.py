# tests/integration/test_elo_fit.py
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from tests.fixtures.dataframes import make_complete_elo_games
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.fit import fit_elo
from gridiron_edge.ratings.elo.lineage import (
    ELO_LINEAGE_SCHEMA_VERSION,
    elo_lineage_path,
    file_content_digest,
    load_elo_lineage,
    verify_current_elo_lineage,
)


def _elo_repo(
    tmp_path: Path,
) -> Path:
    """Build a repository with complete history for Elo reconstruction."""
    return MiniRepoBuilder(tmp_path).with_complete_elo_games().build()


def test_fit_elo_writes_state_table(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)

    fit_elo(repo=repo)

    elo_path = dataset_path(repo, "elo_state")
    assert elo_path.exists()

    elo = pd.read_csv(elo_path)
    assert {
        "NFL_TEAM",
        "NFL_YEAR",
        "NFL_WEEK",
        "ELO",
    }.issubset(elo.columns)
    assert {
        "Team A",
        "Team B",
    }.issubset(set(elo["NFL_TEAM"]))


def test_fit_elo_is_deterministic(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)

    fit_elo(repo=repo)
    first = pd.read_csv(dataset_path(repo, "elo_state"))

    fit_elo(repo=repo)
    second = pd.read_csv(dataset_path(repo, "elo_state"))

    assert_frame_equal(first, second)


def test_fit_elo_rejects_partial_history_without_replacing_state(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)
    fit_elo(repo=repo)

    elo_path = dataset_path(repo, "elo_state")
    before = elo_path.read_bytes()

    lineage_path = elo_lineage_path(repo)
    lineage_before = lineage_path.read_bytes()

    partial = make_complete_elo_games(
        latest_season=2026,
    )
    partial = partial.loc[
        partial["YEAR"].astype(str).eq("2026-2027"),
        :,
    ].copy()
    partial.to_csv(
        dataset_path(repo, "games"),
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("must begin with season 1999-2000; earliest represented season is 2026-2027"),
    ):
        fit_elo(repo=repo)

    assert elo_path.read_bytes() == before
    assert lineage_path.read_bytes() == lineage_before
    assert verify_current_elo_lineage(repo=repo) is False


def test_fit_elo_preserves_prior_strength_into_partial_latest_season(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)

    fit_elo(repo=repo)
    elo = pd.read_csv(dataset_path(repo, "elo_state"))

    team_a_week_one = elo.loc[
        elo["NFL_TEAM"].astype(str).eq("Team A")
        & elo["NFL_YEAR"].astype(str).eq("2026-2027")
        & pd.to_numeric(
            elo["NFL_WEEK"],
            errors="coerce",
        ).eq(1),
        "ELO",
    ].iloc[0]

    team_a_week_two = elo.loc[
        elo["NFL_TEAM"].astype(str).eq("Team A")
        & elo["NFL_YEAR"].astype(str).eq("2026-2027")
        & pd.to_numeric(
            elo["NFL_WEEK"],
            errors="coerce",
        ).eq(2),
        "ELO",
    ].iloc[0]

    assert team_a_week_one != pytest.approx(1500.0)
    assert team_a_week_two > team_a_week_one


def test_fit_elo_writes_matching_lineage(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)

    fit_elo(repo=repo)

    lineage_path = elo_lineage_path(repo)
    assert lineage_path.is_file()

    lineage = load_elo_lineage(repo=repo)

    assert lineage.schema_version == (ELO_LINEAGE_SCHEMA_VERSION)
    assert lineage.source_games.relative_path == ("data/cleaned/NFL_wk_by_wk_cleaned.csv")
    assert lineage.elo_state.relative_path == ("data/cleaned/NFL_Team_Elo.csv")

    games_path = dataset_path(repo, "games")
    elo_path = dataset_path(repo, "elo_state")

    assert lineage.source_games.content_digest == (file_content_digest(games_path))
    assert lineage.elo_state.content_digest == (file_content_digest(elo_path))
    assert verify_current_elo_lineage(repo=repo)


def test_fit_elo_repeated_reconstruction_preserves_content_identity(
    tmp_path: Path,
) -> None:
    repo = _elo_repo(tmp_path)

    fit_elo(repo=repo)
    first = load_elo_lineage(repo=repo)

    fit_elo(repo=repo)
    second = load_elo_lineage(repo=repo)

    assert second.generated_at >= first.generated_at
    assert second.source_games == first.source_games
    assert second.elo_state == first.elo_state
    assert verify_current_elo_lineage(repo=repo)
