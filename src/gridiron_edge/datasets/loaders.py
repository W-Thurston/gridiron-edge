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
    """Load the cleaned historical NFL games dataset."""
    return load_csv(repo_root, "games")


def load_schedule_upcoming_rich(repo_root: Path) -> pd.DataFrame:
    """Load the rich schedule-complete upcoming-game artifact."""
    path: Path = dataset_path(repo_root, "schedule_upcoming_rich")
    return pd.read_parquet(path)


def load_elo_state(repo_root: Path) -> pd.DataFrame:
    """Load the Elo ratings state table."""
    return load_csv(repo_root, "elo_state")


def load_stadiums(repo_root: Path) -> pd.DataFrame:
    """Load the stadium reference dataset."""
    return load_csv(repo_root, "stadiums")


def load_moneylines(repo_root: Path) -> pd.DataFrame:
    """Load the historical moneylines dataset."""
    return load_csv(repo_root, "moneylines")


def load_teams_long_short(repo_root: Path) -> pd.DataFrame:
    """Load team long-to-short name mapping (from unified metadata)."""
    df: pd.DataFrame = load_csv(repo_root, "team_metadata")
    return df.loc[:, ["NFL_LONG_NAME", "NFL_SHORT_NAME"]].copy()


def load_divisions(repo_root: Path) -> pd.DataFrame:
    """Load conference and division assignment (from unified metadata)."""
    df: pd.DataFrame = load_csv(repo_root, "team_metadata")

    div_letter_to_name: dict[str, str] = {
        "N": "North",
        "S": "South",
        "E": "East",
        "W": "West",
    }

    div_names: pd.Series = df["div"].map(div_letter_to_name)
    return pd.DataFrame(
        {
            "NFL_TEAM": df["NFL_LONG_NAME"],
            "CONFERENCE": df["conf"],
            "DIVISION": df["conf"].str.cat(div_names, sep=" "),
        }
    )


def load_epa_by_game(repo_root: Path) -> pd.DataFrame:
    """Load the pre-aggregated game-level EPA statistics."""
    path: Path = repo_root / "data" / "cleaned" / "epa_by_game.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    """Load a Parquet file if it exists; return ``None`` otherwise."""
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_modeling_file(
    repo_root: Path,
    *,
    expected_columns: list[str] | None = None,
    required_schema_version: int | None = None,
    context: str = "",
) -> pd.DataFrame:
    """Load the full feature matrix with optional manifest validation."""
    from gridiron_edge.datasets.registry import dataset_path
    from gridiron_edge.features.manifest import (
        read_manifest,
        validate_columns,
        validate_schema_version,
    )

    df: DataFrame = pd.read_parquet(dataset_path(repo_root, "modeling_full"))

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


def load_weekly_product(
    repo_root: Path,
    product_id: str,
) -> DataFrame:
    """Load one exact immutable weekly game product.

    Loading uses the weekly-product serialization boundary and performs no
    prediction, feature, model, calibration, or forecast computation.
    """
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        load_weekly_product as load_stored_weekly_product,
    )

    return load_stored_weekly_product(
        product_id,
        repo=repo_root,
    )


def load_current_weekly_product(
    repo_root: Path,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Load the explicitly selected current product for one weekly scope."""
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        load_current_weekly_product as load_current_stored_weekly_product,
    )

    return load_current_stored_weekly_product(
        season=season,
        week=week,
        repo=repo_root,
    )
