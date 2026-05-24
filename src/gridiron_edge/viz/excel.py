# src/gridiron_edge/viz/excel.py
"""Elo rankings CSV output.

The Excel workbook has been retired. Rankings are now written as
versioned CSVs to data/output/rankings/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.registry import dataset_path


def write_elo_rankings_csv(*, year: str, week: int, repo: Path | None = None) -> Path:
    """Write week-over-week Elo rankings to a versioned CSV.

    Args:
        year: NFL season label (e.g. ``"2026-2027"``).
        week: Week number to compare (week vs week+1).
        repo: Repository root path.

    Returns:
        Path to the written CSV file.
    """
    from gridiron_edge.core.paths import repo_root as _repo_root

    resolved_repo: Path = repo or _repo_root()
    elo_path: Path = dataset_path(resolved_repo, "elo_state")
    df_elo: DataFrame = pd.read_csv(elo_path)
    df_elo = df_elo.loc[
        (df_elo["NFL_YEAR"] == year) & (df_elo["NFL_WEEK"].isin([week, week + 1])), :
    ].copy()

    df1 = (
        df_elo.loc[df_elo["NFL_WEEK"] == week, ["NFL_TEAM", "ELO"]]
        .sort_values("ELO", ascending=False)
        .reset_index(drop=True)
    )
    df1["RANK_WK"] = np.arange(1, len(df1) + 1)

    df2 = (
        df_elo.loc[df_elo["NFL_WEEK"] == week + 1, ["NFL_TEAM", "ELO"]]
        .sort_values("ELO", ascending=False)
        .reset_index(drop=True)
    )
    df2["RANK_WK1"] = np.arange(1, len(df2) + 1)
    df2 = df2.rename(columns={"ELO": "ELO_WK1"})

    out = df1.merge(df2, on="NFL_TEAM", how="outer")
    out["RANK_CHANGE"] = out["RANK_WK"] - out["RANK_WK1"]

    out_dir: Path = resolved_repo / "data" / "output" / "rankings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"elo_rankings_{year[:4]}_wk{week:02d}.csv"
    out.to_csv(out_path, index=False)
    return out_path
