# src/gridiron_edge/ratings/elo/fit.py

from pathlib import Path

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders, writers
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ratings.elo.table import (
    build_elo_state_table_all_years,
    update_elo_state_incremental,
)


def fit_elo(
    *,
    all_years: bool,
    repo: Path | None = None,
) -> None:
    """Build the Elo state table from cleaned game results.

    Args:
        all_years: If ``True``, rebuilds the full Elo table from scratch
            using all historical games. If ``False``, applies an incremental
            update to the existing state table.
        repo: Absolute path to the repository root. Defaults to the
            value returned by ``repo_root()``.
    """
    resolved_repo: Path = repo or repo_root()

    games: pd.DataFrame = loaders.load_games(resolved_repo)

    elo_path: Path = dataset_path(resolved_repo, "elo_state")
    if all_years or not elo_path.exists():
        elo_df: pd.DataFrame = build_elo_state_table_all_years(games)
    else:
        elo_existing: pd.DataFrame = loaders.load_elo_state(resolved_repo)
        elo_df = update_elo_state_incremental(
            games=games,
            elo_state_existing=elo_existing,
        )

    writers.write_csv(resolved_repo, "elo_state", elo_df)
