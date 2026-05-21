from pathlib import Path

import pandas as pd

from gridiron_edge.viz.excel import (
    PREDICTIONS_SHEET,
    RANKS_SHEET,
    ensure_excel_workbook,
    write_predictions_sheet,
)


def test_ensure_excel_workbook_creates_sheets(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    ensure_excel_workbook(path)
    assert path.exists()
    sheets = pd.ExcelFile(path).sheet_names
    assert PREDICTIONS_SHEET in sheets
    assert RANKS_SHEET in sheets


def test_write_predictions_on_new_workbook(tmp_path: Path) -> None:
    path = tmp_path / "ranks.xlsx"
    df = pd.DataFrame({"AWAY_TEAM": ["X"], "HOME_TEAM": ["Y"]})
    write_predictions_sheet(df, excel_path=path)
    assert path.exists()
