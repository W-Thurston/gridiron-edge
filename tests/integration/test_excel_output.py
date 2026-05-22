from pathlib import Path

import pandas as pd


def test_write_elo_rankings_csv(tmp_path: Path) -> None:
    """write_elo_rankings_csv produces a versioned CSV with expected columns."""
    from gridiron_edge.viz.excel import write_elo_rankings_csv

    # Build a minimal elo_state fixture
    elo_data = pd.DataFrame(
        {
            "NFL_TEAM": ["Team A", "Team B", "Team A", "Team B"],
            "NFL_YEAR": ["2026-2027"] * 4,
            "NFL_WEEK": [1, 1, 2, 2],
            "ELO": [1520.0, 1480.0, 1525.0, 1475.0],
        }
    )
    elo_path = tmp_path / "data" / "cleaned" / "NFL_Team_Elo.csv"
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    elo_data.to_csv(elo_path, index=False)

    out = write_elo_rankings_csv(year="2026-2027", week=1, repo=tmp_path)

    assert out.exists()
    assert out.suffix == ".csv"
    df = pd.read_csv(out)
    assert "NFL_TEAM" in df.columns
    assert "ELO" in df.columns
    assert "RANK_CHANGE" in df.columns
    assert len(df) == 2


def test_write_elo_rankings_csv_filename(tmp_path: Path) -> None:
    """Output filename includes year and week."""
    from gridiron_edge.viz.excel import write_elo_rankings_csv

    elo_data = pd.DataFrame(
        {
            "NFL_TEAM": ["Team A", "Team B", "Team A", "Team B"],
            "NFL_YEAR": ["2026-2027"] * 4,
            "NFL_WEEK": [5, 5, 6, 6],
            "ELO": [1510.0, 1490.0, 1515.0, 1485.0],
        }
    )
    elo_path = tmp_path / "data" / "cleaned" / "NFL_Team_Elo.csv"
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    elo_data.to_csv(elo_path, index=False)

    out = write_elo_rankings_csv(year="2026-2027", week=5, repo=tmp_path)

    assert "2026" in out.name
    assert "wk05" in out.name
