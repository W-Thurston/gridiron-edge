# src/gridiron_edge/datasets/registry.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetKey = Literal[
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse",
    "schedule_upcoming_raw_nflverse",
    # ---- Raw ingest (legacy PFR — kept for backwards compat during transition) ----
    "games_raw",
    "schedule_upcoming_raw",
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
        key: The ``DatasetKey`` identifier for this dataset.
        relpath: Path to the dataset file, relative to the repository root.
    """

    key: DatasetKey
    relpath: str  # relative to repo root


DATASETS: dict[DatasetKey, DatasetSpec] = {
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse": DatasetSpec(
        "games_raw_nflverse",
        "data/raw/NFL_wk_by_wk_nflverse.parquet",
    ),
    "schedule_upcoming_raw_nflverse": DatasetSpec(
        "schedule_upcoming_raw_nflverse",
        "data/raw/NFL_upcoming_schedule_nflverse.parquet",
    ),
    # ---- Raw ingest (legacy PFR) ----
    "games_raw": DatasetSpec("games_raw", "data/raw/NFL_wk_by_wk.csv"),
    "schedule_upcoming_raw": DatasetSpec(
        "schedule_upcoming_raw",
        "data/raw/NFL_upcoming_schedule.csv",
    ),
    # ---- Canonical cleaned datasets ----
    "games": DatasetSpec("games", "data/cleaned/NFL_wk_by_wk_cleaned.csv"),
    "schedule_upcoming": DatasetSpec(
        "schedule_upcoming",
        "data/cleaned/NFL_upcoming_schedule_cleaned.csv",
    ),
    "weather_enriched": DatasetSpec(
        "weather_enriched",
        "data/cleaned/NFL_wk_by_wk_w_weather.csv",
    ),
    "stadiums": DatasetSpec("stadiums", "data/cleaned/NFL_stadium_reference.csv"),
    "moneylines": DatasetSpec(
        "moneylines",
        "data/cleaned/NFL_historical_moneylines.csv",
    ),
    "teams_long_short": DatasetSpec(
        "teams_long_short",
        "data/cleaned/NFL_long_to_short_name.csv",
    ),
    "divisions": DatasetSpec("divisions", "data/cleaned/NFL_conference_division.csv"),
    # ---- Ratings / state ----
    "elo_state": DatasetSpec("elo_state", "data/cleaned/NFL_Team_Elo.csv"),
    # ---- Derived modeling artifacts ----
    "modeling_base": DatasetSpec(
        "modeling_base",
        "data/modeling/base_modeling_file.csv",
    ),
    "modeling_full": DatasetSpec(
        "modeling_full",
        "data/modeling/modeling_file.csv",
    ),
    "predictions_csv": DatasetSpec(
        "predictions_csv",
        "data/output/predictions",  # directory, not a single file
    ),
    "elo_rankings_csv": DatasetSpec(
        "elo_rankings_csv",
        "data/output/rankings",  # directory
    ),
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
