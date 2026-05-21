from pathlib import Path

import pandas as pd

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.fit import fit_elo


def test_fit_elo_writes_state_table(mini_repo: Path) -> None:
    fit_elo(all_years=True, repo=mini_repo)
    elo_path = dataset_path(mini_repo, "elo_state")
    assert elo_path.exists()
    elo = pd.read_csv(elo_path)
    assert {"NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"}.issubset(elo.columns)
    assert elo["NFL_TEAM"].isin(["Team A", "Team B"]).any()


def test_fit_elo_idempotent_rebuild(mini_repo: Path) -> None:
    fit_elo(all_years=True, repo=mini_repo)
    first = pd.read_csv(dataset_path(mini_repo, "elo_state"))
    fit_elo(all_years=True, repo=mini_repo)
    second = pd.read_csv(dataset_path(mini_repo, "elo_state"))
    pd.testing.assert_frame_equal(first, second)
