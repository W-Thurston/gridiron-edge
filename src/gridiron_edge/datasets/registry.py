# src/gridiron_edge/datasets/registry.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetKey = Literal[
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse",
    "schedule_upcoming_raw_nflverse",
    # ---- Canonical cleaned datasets ----
    "games",
    "schedule_upcoming",
    "weather_enriched",
    "elo_state",
    "stadiums",
    "moneylines",
    "teams_long_short",
    "divisions",
    # ---- Derived modeling artifacts ----
    "modeling_base",
    "modeling_full",
    # ---- Output Directories ----
    "predictions_csv",
    "elo_rankings_csv",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a single registered dataset.

    Attributes:
        relpath: Path to the dataset file, relative to the repository root.
    """

    relpath: str  # relative to repo root


DATASETS: dict[DatasetKey, DatasetSpec] = {
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse": DatasetSpec("data/raw/NFL_wk_by_wk_nflverse.parquet"),
    "schedule_upcoming_raw_nflverse": DatasetSpec(
        "data/raw/NFL_upcoming_schedule_nflverse.parquet",
    ),
    # ---- Canonical cleaned datasets ----
    "games": DatasetSpec("data/cleaned/NFL_wk_by_wk_cleaned.csv"),
    "schedule_upcoming": DatasetSpec("data/cleaned/NFL_upcoming_schedule_cleaned.csv"),
    "weather_enriched": DatasetSpec("data/cleaned/NFL_wk_by_wk_w_weather.csv"),
    "stadiums": DatasetSpec("data/cleaned/NFL_stadium_reference.csv"),
    "moneylines": DatasetSpec("data/cleaned/NFL_historical_moneylines.csv"),
    "teams_long_short": DatasetSpec("data/cleaned/NFL_long_to_short_name.csv"),
    "divisions": DatasetSpec("data/cleaned/NFL_conference_division.csv"),
    # ---- Ratings / state ----
    "elo_state": DatasetSpec("data/cleaned/NFL_Team_Elo.csv"),
    # ---- Derived modeling artifacts ----
    "modeling_base": DatasetSpec("data/modeling/base_modeling_file.parquet"),
    "modeling_full": DatasetSpec("data/modeling/modeling_file.parquet"),
    "predictions_csv": DatasetSpec("data/output/predictions"),  # directory, not a single file
    "elo_rankings_csv": DatasetSpec("data/output/rankings"),  # directory
}


def dataset_path(repo_root: Path, key: DatasetKey) -> Path:
    """Resolve the absolute path for a registered dataset.

    Args:
        repo_root: Absolute path to the repository root.
        key: A ``DatasetKey`` identifying the dataset.

    Returns:
        Absolute path to the dataset file.

    Raises:
        KeyError: If ``key`` is not found in ``DATASETS``.
    """
    return repo_root / DATASETS[key].relpath
