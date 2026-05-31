from pathlib import Path

import pandas as pd

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.pipeline import build_model_inputs
from gridiron_edge.ratings.elo.fit import fit_elo


def test_build_model_inputs_full_rebuild(mini_repo: Path) -> None:
    fit_elo(all_years=True, repo=mini_repo)
    build_model_inputs(all_years=True, repo=mini_repo)

    base = pd.read_parquet(dataset_path(mini_repo, "modeling_base"))
    full = pd.read_parquet(dataset_path(mini_repo, "modeling_full"))

    assert len(base) == 4  # two rows per game
    assert len(full) == 4
    assert "HOME_FIELD" in full.columns
    assert "TEAM_A_ELO" in full.columns
    assert "TEAM_A_KM_TRAVELED" in full.columns
