# src/gridiron_edge/datasets/writers.py

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity

from .registry import DatasetKey, dataset_path


def write_csv(
    repo_root: Path,
    key: DatasetKey,
    df: pd.DataFrame,
    *,
    index: bool = False,
    **to_csv_kwargs: Any,
) -> Path:
    """Write a DataFrame to the registered path for a dataset key.

    Creates any missing parent directories before writing.

    Args:
        repo_root: Absolute path to the repository root.
        key: A ``DatasetKey`` identifying the destination dataset.
        df: The DataFrame to write.
        index: Whether to include the DataFrame index in the CSV.
            Defaults to ``False``.
        **to_csv_kwargs: Additional keyword arguments forwarded to
            ``DataFrame.to_csv``.

    Returns:
        The absolute path of the file that was written.
    """
    path: Path = dataset_path(repo_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **to_csv_kwargs)
    return path


def write_parquet(
    repo_root: Path,
    key: DatasetKey,
    df: pd.DataFrame,
    *,
    index: bool = False,
) -> Path:
    """Write a DataFrame to the registered path for a dataset key as Parquet.

    Creates any missing parent directories before writing.

    Args:
        repo_root: Absolute path to the repository root.
        key: A ``DatasetKey`` identifying the destination dataset.
        df: The DataFrame to write.
        index: Whether to include the DataFrame index. Defaults to False.

    Returns:
        The absolute path of the file that was written.
    """
    path: Path = dataset_path(repo_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    return path


def write_weekly_product(
    repo_root: Path,
    df: pd.DataFrame,
    *,
    identity: WeeklyProductIdentity,
) -> Path:
    """Write one validated immutable weekly game product."""
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        write_weekly_product as write_stored_weekly_product,
    )

    return write_stored_weekly_product(
        df,
        identity=identity,
        repo=repo_root,
    )


def select_current_weekly_product(
    repo_root: Path,
    product_id: str,
    *,
    season: str,
    week: int,
    selected_at: datetime,
) -> None:
    """Explicitly select one persisted product as current."""
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        select_current_weekly_product as select_stored_weekly_product,
    )

    select_stored_weekly_product(
        product_id,
        season=season,
        week=week,
        selected_at=selected_at,
        repo=repo_root,
    )
