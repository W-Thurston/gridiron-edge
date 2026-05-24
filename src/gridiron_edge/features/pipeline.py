# src/gridiron_edge/features/pipeline.py

from pathlib import Path
from typing import Final

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders, writers
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.manifest import write_manifest
from gridiron_edge.features.registry import FeatureRegistry, run_features
import gridiron_edge.features.team.elo
import gridiron_edge.features.team.epa
import gridiron_edge.features.team.home_field
import gridiron_edge.features.team.travel  # noqa: F401

# Feature order matters:
# - home_field should run before travel (travel uses HOME_FIELD)
# - elo can run anytime
FEATURES: Final[list[str]] = ["home_field", "team_elo", "travel", "epa"]


def _feature_columns(feature_names: list[str]) -> list[str]:
    cols: list[str] = []
    for name in feature_names:
        cols.extend(FeatureRegistry.get(name)().spec.produces)
    return cols


def build_base_modeling_table(games: pd.DataFrame) -> pd.DataFrame:
    """Replicates legacy prep_data_modeling_file() behavior:.

    Produces two rows per game:
      - TEAM_A=winner, TEAM_B=loser, RESULT=1
      - TEAM_A=loser,  TEAM_B=winner, RESULT=0

    """
    df: pd.DataFrame = games.copy()

    # Keep only the columns needed (matches your legacy intent)
    df = df.loc[:, ["GAME_ID", "WINNER", "LOSER", "YEAR", "WEEK_NUM"]].copy()

    df["RESULT"] = 1
    df.columns = ["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM", "RESULT"]

    flipped = df.loc[:, ["GAME_ID", "TEAM_B", "TEAM_A", "YEAR", "WEEK_NUM"]].copy()
    flipped["RESULT"] = 0
    flipped.columns = ["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM", "RESULT"]

    return (
        pd.concat([df, flipped], ignore_index=True)
        .sort_values(["YEAR", "WEEK_NUM", "TEAM_A"], kind="stable")
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def build_model_inputs(*, all_years: bool, repo: Path | None = None) -> None:
    """Build modeling inputs (base + full).

    all_years=True:
      - rebuild base and full from scratch

    all_years=False:
      - append only new GAME_ID rows to base/full using incremental logic
    """
    repo = repo or repo_root()
    datasets = DatasetAccessor(repo)

    # Load canonical inputs
    games: pd.DataFrame = loaders.load_games(repo)

    # Paths for outputs
    base_path: Path = dataset_path(repo, "modeling_base")
    full_path: Path = dataset_path(repo, "modeling_full")

    # Build base
    base_all: pd.DataFrame = build_base_modeling_table(games)

    if all_years or (not base_path.exists()) or (not full_path.exists()):
        # Full rebuild
        base_out: pd.DataFrame = base_all
        full_out: pd.DataFrame = run_features(
            df=base_out,
            feature_names=FEATURES,
            datasets=datasets,
        )

        writers.write_csv(repo, "modeling_base", base_out)
        writers.write_csv(repo, "modeling_full", full_out)
        write_manifest(
            full_out,
            feature_names=list(FEATURES),
            feature_columns=_feature_columns(list(FEATURES)),
            modeling_dir=dataset_path(repo, "modeling_full").parent,
        )
        return

    # Incremental build: only process unseen GAME_ID rows
    base_existing: pd.DataFrame | None = _load_csv_if_exists(base_path)
    full_existing: pd.DataFrame | None = _load_csv_if_exists(full_path)
    if base_existing is None or full_existing is None:
        # Safety fallback: rebuild if either is missing/unreadable
        base_out = base_all
        full_out = run_features(df=base_out, feature_names=FEATURES, datasets=datasets)

        writers.write_csv(repo, "modeling_base", base_out)
        writers.write_csv(repo, "modeling_full", full_out)
        return

    existing_game_ids: set = set(base_existing["GAME_ID"].unique().tolist())
    new_mask: pd.Series[bool] = ~base_all["GAME_ID"].isin(existing_game_ids)
    base_new: pd.DataFrame = base_all.loc[new_mask, :].copy()

    if base_new.empty:
        # No new games → write manifest if missing, then return
        _manifest_path: Path = (
            dataset_path(repo, "modeling_full").parent / "modeling_file_manifest.json"
        )
        if not _manifest_path.exists():
            write_manifest(
                full_existing,
                feature_names=list(FEATURES),
                feature_columns=_feature_columns(list(FEATURES)),
                modeling_dir=dataset_path(repo, "modeling_full").parent,
            )
        return

    # Compute features only for the new rows
    full_new: pd.DataFrame = run_features(
        df=base_new,
        feature_names=FEATURES,
        datasets=datasets,
    )

    # Append + de-dupe (stable ordering)
    base_out = (
        pd.concat([base_existing, base_new], ignore_index=True)
        .drop_duplicates(subset=["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM"])
        .reset_index(drop=True)
    )
    full_out = (
        pd.concat([full_existing, full_new], ignore_index=True)
        .drop_duplicates(subset=["GAME_ID", "TEAM_A", "TEAM_B", "YEAR", "WEEK_NUM"])
        .reset_index(drop=True)
    )

    writers.write_csv(repo, "modeling_base", base_out)
    writers.write_csv(repo, "modeling_full", full_out)
    write_manifest(
        full_out,
        feature_names=list(FEATURES),
        feature_columns=_feature_columns(list(FEATURES)),
        modeling_dir=dataset_path(repo, "modeling_full").parent,
    )
