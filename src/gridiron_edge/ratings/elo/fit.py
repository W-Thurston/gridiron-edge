from pathlib import Path

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders, writers
from gridiron_edge.ratings.elo.table import build_elo_state_table_all_years


def fit_elo(
    *,
    repo: Path | None = None,
) -> None:
    """Rebuild the canonical Elo state from complete historical games.

    The complete cleaned games history is validated before simulation. A
    validation or simulation failure occurs before the registered Elo artifact
    is replaced.

    Args:
        repo: Absolute path to the repository root. Defaults to the value
            returned by ``repo_root()``.
    """
    resolved_repo: Path = repo or repo_root()
    games: pd.DataFrame = loaders.load_games(resolved_repo)
    elo_df: pd.DataFrame = build_elo_state_table_all_years(games)
    writers.write_csv(resolved_repo, "elo_state", elo_df)
