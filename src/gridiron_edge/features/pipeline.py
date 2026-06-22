# src/gridiron_edge/features/pipeline.py

from pathlib import Path
from typing import Final

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders, writers
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.loaders import load_parquet_if_exists
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.manifest import (
    CURRENT_DATA_VERSION,
    read_manifest,
    write_manifest,
)
from gridiron_edge.features.registry import FeatureRegistry, run_features, validate_ordering

# Side-effect imports: each module registers its feature class with
# FeatureRegistry via the @FeatureRegistry.register(...) decorator.
# The imports must be preserved even though no name is referenced directly.
import gridiron_edge.features.team.divisional
import gridiron_edge.features.team.elo
import gridiron_edge.features.team.epa
import gridiron_edge.features.team.home_field
import gridiron_edge.features.team.primetime
import gridiron_edge.features.team.record
import gridiron_edge.features.team.rest
import gridiron_edge.features.team.schedule_strength
import gridiron_edge.features.team.travel
import gridiron_edge.features.team.venue_hfa
import gridiron_edge.features.team.weather  # noqa: F401

# Feature order matters - dependencies between features:
# - home_field before travel (travel reads HOME_FIELD)
# - travel before venue_hfa (venue_hfa reads IS_NEUTRAL_SITE)
# - team_elo before schedule_strength (SOS/SOV join on elo_state)
# - record and schedule_strength can run any time after home_field
# - primetime has no dependencies
FEATURES: Final[list[str]] = [
    "home_field",
    "team_elo",
    "travel",
    "epa",
    "rest",
    "weather",
    "divisional",
    "venue_hfa",
    "record",
    "schedule_strength",
    "primetime",
]

# Validate that the ordering above satisfies all depends_on constraints.
# This runs at import time - a mis-ordering raises ValueError immediately
# rather than silently producing wrong features during training.
validate_ordering(FEATURES)


def _feature_columns(feature_names: list[str]) -> list[str]:
    cols: list[str] = []
    for name in feature_names:
        cols.extend(FeatureRegistry.get(name)().spec.produces)
    return cols


def build_base_modeling_table(games: pd.DataFrame) -> pd.DataFrame:
    """Build the symmetric two-row-per-game base modeling table.

    Produces two rows per game:
      - TEAM_A=winner, TEAM_B=loser, RESULT=1
      - TEAM_A=loser,  TEAM_B=winner, RESULT=0

    This design ensures every model learns win probability symmetrically
    across both team perspectives.
    """
    df: pd.DataFrame = games.copy()

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


def _data_version_changed(modeling_dir: Path) -> bool:
    """Return True when the manifest's data_version differs from current.

    Used by ``build_model_inputs`` to decide whether an incremental update
    is safe or whether a full rebuild is required because the on-disk data
    was produced by older (possibly buggy) feature code.

    Returns True (force rebuild) when:
        - The manifest is missing
        - The manifest lacks a data_version field (pre-versioning)
        - The manifest's data_version differs from CURRENT_DATA_VERSION
    """
    try:
        manifest = read_manifest(modeling_dir)
    except FileNotFoundError:
        return True

    stored = manifest.get("data_version")
    if not isinstance(stored, int):
        return True
    return stored != CURRENT_DATA_VERSION


def build_model_inputs(*, all_years: bool, repo: Path | None = None) -> None:
    """Build modeling inputs (base + full) as Parquet files.

    Schema version 3: adds rest and weather features;
    converts modeling files from CSV to Parquet for faster load times
    and correct dtype preservation.

    all_years=True:
      - full rebuild from scratch
    all_years=False:
      - append only new GAME_ID rows (incremental build)
    """
    repo = repo or repo_root()
    datasets = DatasetAccessor(repo)

    games: pd.DataFrame = loaders.load_games(repo)

    base_path: Path = dataset_path(repo, "modeling_base")
    full_path: Path = dataset_path(repo, "modeling_full")

    base_all: pd.DataFrame = build_base_modeling_table(games)

    if all_years or not base_path.exists() or not full_path.exists():
        # Full rebuild
        base_out: pd.DataFrame = base_all
        full_out: pd.DataFrame = run_features(
            df=base_out,
            feature_names=FEATURES,
            datasets=datasets,
        )
        writers.write_parquet(repo, "modeling_base", base_out)
        writers.write_parquet(repo, "modeling_full", full_out)
        write_manifest(
            full_out,
            feature_names=list(FEATURES),
            feature_columns=_feature_columns(list(FEATURES)),
            modeling_dir=full_path.parent,
        )
        return

    # Incremental build: only process unseen GAME_ID rows
    base_existing: pd.DataFrame | None = load_parquet_if_exists(base_path)
    full_existing: pd.DataFrame | None = load_parquet_if_exists(full_path)
    if base_existing is None or full_existing is None:
        base_out = base_all
        full_out = run_features(df=base_out, feature_names=FEATURES, datasets=datasets)
        writers.write_parquet(repo, "modeling_base", base_out)
        writers.write_parquet(repo, "modeling_full", full_out)
        write_manifest(
            full_out,
            feature_names=list(FEATURES),
            feature_columns=_feature_columns(list(FEATURES)),
            modeling_dir=full_path.parent,
        )
        return

    # Check whether the existing modeling file's data_version matches the
    # current code. If not, incremental updates would silently preserve
    # stale rows produced by older (potentially buggy) feature code. Force
    # a full rebuild in that case.
    if _data_version_changed(full_path.parent):
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Modeling file data_version is stale (manifest version differs from "
            "CURRENT_DATA_VERSION=%d). Forcing a full rebuild to refresh all "
            "feature values from current code.",
            CURRENT_DATA_VERSION,
        )
        base_out = base_all
        full_out = run_features(df=base_out, feature_names=FEATURES, datasets=datasets)
        writers.write_parquet(repo, "modeling_base", base_out)
        writers.write_parquet(repo, "modeling_full", full_out)
        write_manifest(
            full_out,
            feature_names=list(FEATURES),
            feature_columns=_feature_columns(list(FEATURES)),
            modeling_dir=full_path.parent,
        )
        return

    existing_game_ids: set = set(base_existing["GAME_ID"].unique().tolist())
    new_mask: pd.Series[bool] = ~base_all["GAME_ID"].isin(existing_game_ids)
    base_new: pd.DataFrame = base_all.loc[new_mask, :].copy()

    if base_new.empty:
        _manifest_path: Path = full_path.parent / "modeling_file_manifest.json"
        if not _manifest_path.exists():
            write_manifest(
                full_existing,
                feature_names=list(FEATURES),
                feature_columns=_feature_columns(list(FEATURES)),
                modeling_dir=full_path.parent,
            )
        return

    full_new: pd.DataFrame = run_features(
        df=base_new,
        feature_names=FEATURES,
        datasets=datasets,
    )

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

    writers.write_parquet(repo, "modeling_base", base_out)
    writers.write_parquet(repo, "modeling_full", full_out)
    write_manifest(
        full_out,
        feature_names=list(FEATURES),
        feature_columns=_feature_columns(list(FEATURES)),
        modeling_dir=full_path.parent,
    )
