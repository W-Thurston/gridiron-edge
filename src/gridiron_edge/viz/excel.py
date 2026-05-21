# src/gridiron_edge/viz/excel.py

from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path

PREDICTIONS_SHEET = "Upcoming Weeks Predictions"
RANKS_SHEET = "ELO Ranking Changes"
DEFAULT_SHEETS = (PREDICTIONS_SHEET, RANKS_SHEET)


def ensure_excel_workbook(
    path: Path,
    *,
    sheet_names: tuple[str, ...] = DEFAULT_SHEETS,
) -> None:
    """Create an empty workbook with expected sheet names if missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        for name in sheet_names:
            pd.DataFrame().to_excel(writer, sheet_name=name, index=False)


def _excel_writer(path: Path, *, sheet_name: str) -> pd.ExcelWriter:
    ensure_excel_workbook(path)
    return pd.ExcelWriter(path, mode="a", if_sheet_exists="overlay", engine="openpyxl")


def write_predictions_sheet(df: pd.DataFrame, *, excel_path: Path | None = None) -> None:
    """Write the predictions DataFrame to the upcoming weeks sheet.

    Writes starting at row 3, column 15 (O3) to align with the existing
    Excel template layout.

    Args:
        df: DataFrame of game predictions to write.
        excel_path: Path to the Excel workbook. Defaults to the path
            from ``get_settings().ranks_excel``.
    """
    path: Path = excel_path or get_settings().ranks_excel

    with _excel_writer(path, sheet_name=PREDICTIONS_SHEET) as writer:
        df.to_excel(
            excel_writer=writer,
            sheet_name=PREDICTIONS_SHEET,
            index=False,
            header=False,
            startrow=2,
            startcol=14,
        )


def write_elo_rank_changes(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
    excel_path: Path | None = None,
) -> None:
    """Write week-over-week Elo rank changes to the Excel workbook."""
    from gridiron_edge.core.paths import repo_root

    repo = repo or repo_root()
    settings = get_settings()
    path = excel_path or settings.ranks_excel
    elo_path = dataset_path(repo, "elo_state")

    print(f"> Adjusting elo ranks based on {year} and {week}")
    df_elo = pd.read_csv(elo_path)
    df_elo = df_elo.loc[
        (df_elo["NFL_YEAR"] == year) & (df_elo["NFL_WEEK"].isin([week, week + 1])),
        :,
    ]

    df1 = (
        df_elo.loc[
            (df_elo["NFL_YEAR"] == year) & (df_elo["NFL_WEEK"] == week),
            ["NFL_TEAM", "ELO"],
        ]
        .sort_values(["ELO"], ascending=False)
        .reset_index(drop=True)
    )
    df1["Rank"] = np.arange(1, df1.shape[0] + 1)
    df1["EMPTY"] = np.nan
    df2 = (
        df_elo.loc[
            (df_elo["NFL_YEAR"] == year) & (df_elo["NFL_WEEK"] == week + 1),
            ["NFL_TEAM", "ELO"],
        ]
        .sort_values(["ELO"], ascending=False)
        .reset_index(drop=True)
    )
    df2["Rank"] = np.arange(1, df2.shape[0] + 1)
    df3 = pd.concat([df1, df2], axis=1)

    with _excel_writer(path, sheet_name=RANKS_SHEET) as writer:
        df3.to_excel(
            excel_writer=writer,
            sheet_name=RANKS_SHEET,
            index=False,
            header=False,
            startrow=3,
            startcol=14,
        )
    print(f"> Saving updated Elo Ranks to: {path}")
