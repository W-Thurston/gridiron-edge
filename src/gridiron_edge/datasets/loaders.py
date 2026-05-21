# src/gridiron_edge/datasets/loaders.py

from pathlib import Path
from typing import Any

import pandas as pd

from .registry import DatasetKey, dataset_path


def load_csv(repo_root: Path, key: DatasetKey, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a registered dataset from disk as a DataFrame.

    Args:
        repo_root: Absolute path to the repository root.
        key: A ``DatasetKey`` identifying which dataset to load.
        **read_csv_kwargs: Additional keyword arguments forwarded to
            ``pandas.read_csv``.

    Returns:
        The dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the resolved CSV path does not exist.
    """
    path: Path = dataset_path(repo_root, key)
    return pd.read_csv(path, **read_csv_kwargs)


def load_games(repo_root: Path) -> pd.DataFrame:
    """Load the cleaned historical NFL games dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame of cleaned historical game results.
    """
    return load_csv(repo_root, "games")


def load_schedule_upcoming(repo_root: Path) -> pd.DataFrame:
    """Load the cleaned upcoming schedule dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame of upcoming scheduled games.
    """
    return load_csv(repo_root, "schedule_upcoming")


def load_elo_state(repo_root: Path) -> pd.DataFrame:
    """Load the Elo ratings state table.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame with per-team Elo ratings indexed by season and week.
    """
    return load_csv(repo_root, "elo_state")


def load_stadiums(repo_root: Path) -> pd.DataFrame:
    """Load the stadium reference dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame with stadium metadata including coordinates and altitude.
    """
    return load_csv(repo_root, "stadiums")


def load_moneylines(repo_root: Path) -> pd.DataFrame:
    """Load the historical moneylines dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame of historical NFL moneyline odds.
    """
    return load_csv(repo_root, "moneylines")


def load_teams_long_short(repo_root: Path) -> pd.DataFrame:
    """Load the team name long-to-short mapping dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame mapping full team names to short codes.
    """
    return load_csv(repo_root, "teams_long_short")


def load_divisions(repo_root: Path) -> pd.DataFrame:
    """Load the conference and division assignment dataset.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame mapping each team to its conference and division.
    """
    return load_csv(repo_root, "divisions")
