# src/gridiron_edge/datasets/loaders.py

from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

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


def load_epa_by_game(repo_root: Path) -> pd.DataFrame:
    """Load the pre-aggregated game-level EPA statistics.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        DataFrame with one row per (team, game) containing EPA metrics.
        Empty DataFrame if the file does not exist yet.
    """
    path: Path = repo_root / "data" / "cleaned" / "epa_by_game.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_modeling_file(
    repo_root: Path,
    *,
    expected_columns: list[str] | None = None,
    required_schema_version: int | None = None,
    context: str = "",
) -> pd.DataFrame:
    """Load the full feature matrix with optional manifest validation.

    Args:
        repo_root: Absolute path to the repository root.
        expected_columns: If provided, asserts these columns are present.
        required_schema_version: If provided, asserts the manifest schema
            version matches.
        context: Optional label for error messages (e.g. model name).

    Returns:
        Full feature matrix DataFrame.

    Raises:
        FileNotFoundError: If the modeling file or manifest does not exist.
        ValueError: If column or schema version validation fails.
    """
    from gridiron_edge.datasets.registry import dataset_path
    from gridiron_edge.features.manifest import (
        read_manifest,
        validate_columns,
        validate_schema_version,
    )

    df: DataFrame = load_csv(repo_root, "modeling_full")

    if expected_columns is not None or required_schema_version is not None:
        modeling_dir: Path = dataset_path(repo_root, "modeling_full").parent
        manifest: dict[str, Any] = read_manifest(modeling_dir)

        if required_schema_version is not None:
            validate_schema_version(
                manifest,
                required_version=required_schema_version,
                context=context,
            )

        if expected_columns is not None:
            validate_columns(df, expected_columns=expected_columns, context=context)

    return df
