from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetKey = Literal[
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse",
    "schedule_upcoming_raw_nflverse",
    # ---- Canonical cleaned datasets ----
    "games",
    "schedule_upcoming_rich",
    "weather_enriched",
    "elo_state",
    "stadiums",
    "moneylines",
    "team_metadata",
    "epa_by_game",
    "player_game_logs",
    # ---- Derived modeling artifacts ----
    "modeling_base",
    "modeling_full",
    # ---- Archive logs ----
    "prediction_log",
    "prop_prediction_log",
    "bet_ledger",
    "bankroll_txn",
    # ---- Output directories ----
    "predictions_csv",
    "elo_rankings_csv",
    "weekly_products",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a single registered dataset.

    Attributes:
        relpath: Path to the dataset file, relative to the repository root.
    """

    relpath: str


DATASETS: dict[DatasetKey, DatasetSpec] = {
    # ---- Raw ingest (nflverse) ----
    "games_raw_nflverse": DatasetSpec("data/raw/NFL_wk_by_wk_nflverse.parquet"),
    "schedule_upcoming_raw_nflverse": DatasetSpec(
        "data/raw/NFL_upcoming_schedule_nflverse.parquet",
    ),
    # ---- Canonical cleaned datasets ----
    "games": DatasetSpec("data/cleaned/NFL_wk_by_wk_cleaned.csv"),
    "schedule_upcoming_rich": DatasetSpec("data/cleaned/NFL_upcoming_schedule_rich.parquet"),
    "weather_enriched": DatasetSpec("data/cleaned/NFL_wk_by_wk_w_weather.csv"),
    "stadiums": DatasetSpec("data/cleaned/NFL_stadium_reference.csv"),
    "moneylines": DatasetSpec("data/cleaned/NFL_historical_moneylines.csv"),
    "team_metadata": DatasetSpec("data/cleaned/NFL_team_metadata.csv"),
    "epa_by_game": DatasetSpec("data/cleaned/epa_by_game.parquet"),
    "player_game_logs": DatasetSpec("data/cleaned/player_game_logs.parquet"),
    # ---- Ratings / state ----
    "elo_state": DatasetSpec("data/cleaned/NFL_Team_Elo.csv"),
    # ---- Derived modeling artifacts ----
    "modeling_base": DatasetSpec("data/modeling/base_modeling_file.parquet"),
    "modeling_full": DatasetSpec("data/modeling/modeling_file.parquet"),
    # ---- Archive logs ----
    "prediction_log": DatasetSpec("data/output/predictions/predictions_log.parquet"),
    "prop_prediction_log": DatasetSpec("data/output/props/prop_predictions_log.parquet"),
    "bet_ledger": DatasetSpec("data/betting/bet_ledger.parquet"),
    "bankroll_txn": DatasetSpec("data/betting/bankroll_txn.parquet"),
    # ---- Output directories ----
    "predictions_csv": DatasetSpec("data/output/predictions"),
    "elo_rankings_csv": DatasetSpec("data/output/rankings"),
    "weekly_products": DatasetSpec("data/output/weekly_products"),
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
