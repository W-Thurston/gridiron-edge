# src/gridiron_edge/datasets/registry.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetKey = Literal[
    "games_raw",
    "games",
    "schedule_upcoming_raw",
    "schedule_upcoming",
    "weather_enriched",
    "elo_state",
    "stadiums",
    "moneylines",
    "teams_long_short",
    "divisions",
    "modeling_base",
    "modeling_full",
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
    # ---- Raw ingest ----
    "games_raw": DatasetSpec("games_raw", "data/raw/NFL_wk_by_wk.csv"),
    "schedule_upcoming_raw": DatasetSpec(
        "schedule_upcoming_raw",
        "data/raw/NFL_upcoming_schedule.csv",
    ),
    # ---- Clean canonical datasets ----
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
