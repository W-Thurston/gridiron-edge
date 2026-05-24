# src/gridiron_edge/datasets/writers.py

from pathlib import Path
from typing import Any

import pandas as pd

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
