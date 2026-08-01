# src/gridiron_edge/features/pipeline.py

import logging
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders, writers
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.loaders import load_parquet_if_exists
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.manifest import (
    CURRENT_DATA_VERSION,
    CURRENT_SCHEMA_VERSION,
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
from gridiron_edge.models.game_prediction.game_schema import (
    GAME_IDENTITY_COLUMNS,
    GAME_SCORE_COLUMNS,
    GAME_TARGET_COLUMNS,
)

# Canonical feature order is explicit and validated at import time.
# Each feature consumes and produces stable Away/Home-oriented columns.
CANONICAL_FEATURES: Final[list[str]] = [
    "home_away_elo",
    "home_away_epa",
    "home_away_rest",
    "home_away_record",
    "home_away_schedule_strength",
    "home_away_travel",
    "home_away_venue_hfa",
    "home_away_divisional",
    "home_away_primetime",
    "home_away_weather",
]

# Validate that the ordering above satisfies all depends_on constraints.
# This runs at import time - a mis-ordering raises ValueError immediately
# rather than silently producing wrong features during training.
validate_ordering(CANONICAL_FEATURES)


def _feature_columns(feature_names: list[str]) -> list[str]:
    cols: list[str] = []
    for name in feature_names:
        cols.extend(FeatureRegistry.get(name)().spec.produces)
    return cols


def canonical_feature_columns() -> list[str]:
    """Return the ordered, unique canonical feature output columns.

    Raises:
        ValueError: If more than one canonical feature declares the same
            output column.
    """
    columns: list[str] = _feature_columns(CANONICAL_FEATURES)
    duplicated: list[str] = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated:
        raise ValueError(
            "Canonical features declare duplicate output columns: " + ", ".join(duplicated)
        )
    return columns


_HOME_AWAY_MODELING_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "GAME_DATE",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
    "IS_NEUTRAL_SITE",
)

_HOME_AWAY_MODELING_COLUMNS: Final[tuple[str, ...]] = (
    *GAME_IDENTITY_COLUMNS,
    "GAME_DATE",
    *GAME_SCORE_COLUMNS,
    "IS_NEUTRAL_SITE",
    *GAME_TARGET_COLUMNS,
)


def _validate_modeling_identity(
    source: DataFrame,
) -> None:
    """Validate game and team identities."""
    if source["GAME_ID"].isna().any():
        raise ValueError("GAME_ID must not contain nulls.")

    empty_game_ids = source["GAME_ID"].astype(str).str.strip().eq("")
    if empty_game_ids.any():
        raise ValueError("GAME_ID must not contain empty values.")

    duplicated = source["GAME_ID"].duplicated(
        keep=False,
    )
    if duplicated.any():
        duplicate_ids = sorted(
            source.loc[
                duplicated,
                "GAME_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError("Historical games contain duplicate game IDs: " + ", ".join(duplicate_ids))

    for column in (
        "YEAR",
        "AWAY_TEAM",
        "HOME_TEAM",
    ):
        if source[column].isna().any():
            raise ValueError(f"{column} must not contain nulls.")

        empty = source[column].astype(str).str.strip().eq("")
        if empty.any():
            raise ValueError(f"{column} must not contain empty values.")

    same_team = source["AWAY_TEAM"].astype(str) == source["HOME_TEAM"].astype(str)
    if same_team.any():
        game_ids = sorted(
            source.loc[
                same_team,
                "GAME_ID",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError("Away and home team must differ for games: " + ", ".join(game_ids))


def _coerce_modeling_week(
    source: DataFrame,
) -> None:
    """Validate and normalize week numbers in place."""
    # pyrefly: ignore [bad-assignment]
    week_values: Series = pd.to_numeric(
        source["WEEK_NUM"],
        errors="raise",
    )

    if week_values.isna().any():
        raise ValueError("WEEK_NUM must not contain nulls.")

    if (week_values < 1).any():
        raise ValueError("WEEK_NUM must be at least 1.")

    source["WEEK_NUM"] = week_values.astype(int)


def _coerce_modeling_scores(
    source: DataFrame,
) -> None:
    """Validate and normalize Away and Home scores in place."""
    for column in (
        "AWAY_SCORE",
        "HOME_SCORE",
    ):
        # pyrefly: ignore [bad-assignment]
        values: Series = pd.to_numeric(
            source[column],
            errors="raise",
        )

        if values.isna().any():
            raise ValueError(f"{column} must not contain nulls.")

        if (values < 0).any():
            raise ValueError(f"{column} must not contain negative values.")

        source[column] = values.astype(int)


def _coerce_neutral_site(
    source: DataFrame,
) -> None:
    """Validate and normalize neutral-site state in place."""
    # pyrefly: ignore [bad-assignment]
    neutral_values: Series = pd.to_numeric(
        source["IS_NEUTRAL_SITE"],
        errors="raise",
    )

    if neutral_values.isna().any():
        raise ValueError("IS_NEUTRAL_SITE must not contain nulls.")

    invalid = ~neutral_values.isin(
        [
            0,
            1,
        ]
    )
    if invalid.any():
        raise ValueError("IS_NEUTRAL_SITE must contain only 0 or 1.")

    source["IS_NEUTRAL_SITE"] = neutral_values.astype(int)


def _attach_modeling_targets(
    source: DataFrame,
) -> None:
    """Attach nullable Home Win, margin, and total targets in place."""
    home_wins = source["HOME_SCORE"] > source["AWAY_SCORE"]
    away_wins = source["AWAY_SCORE"] > source["HOME_SCORE"]

    home_win = Series(
        pd.NA,
        index=source.index,
        dtype="Int64",
    )
    home_win.loc[home_wins] = 1
    home_win.loc[away_wins] = 0

    source["HOME_WIN"] = home_win
    source["ACTUAL_MARGIN"] = source["HOME_SCORE"] - source["AWAY_SCORE"]
    source["ACTUAL_TOTAL"] = source["HOME_SCORE"] + source["AWAY_SCORE"]


def _require_home_away_modeling_columns(
    games: DataFrame,
) -> None:
    """Require explicit historical home/away source fields."""
    missing: list[str] = sorted(set(_HOME_AWAY_MODELING_SOURCE_COLUMNS) - set(games.columns))
    if missing:
        raise ValueError(
            "Historical games are missing required home/away columns: " + ", ".join(missing)
        )


def build_home_away_modeling_table(
    games: DataFrame,
) -> DataFrame:
    """Build one canonical home/away modeling row per historical game.

    The input must contain explicit schedule-oriented Away and Home
    identities and scores. Winner/loser fields, game-location inference,
    game-ID parsing, and perspective duplication are not used.

    ``HOME_WIN`` is nullable:

    - ``1`` when the home team won;
    - ``0`` when the away team won;
    - ``pd.NA`` when the game was tied.

    ``ACTUAL_MARGIN`` is always Home Score minus Away Score.
    ``ACTUAL_TOTAL`` is always the sum of Away and Home scores.

    Args:
        games: Cleaned completed historical games.

    Returns:
        One chronologically ordered row per game in the canonical
        home/away modeling schema.

    Raises:
        ValueError: If required fields are missing, game identities are
            invalid or duplicated, teams are invalid, scores are invalid,
            or neutral-site values are not binary.
    """
    _require_home_away_modeling_columns(games)

    source = games.loc[
        :,
        list(_HOME_AWAY_MODELING_SOURCE_COLUMNS),
    ].copy()

    _validate_modeling_identity(source)
    _coerce_modeling_week(source)
    _coerce_modeling_scores(source)
    _coerce_neutral_site(source)
    _attach_modeling_targets(source)

    return source.loc[
        :,
        list(_HOME_AWAY_MODELING_COLUMNS),
    ].sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "GAME_ID",
        ],
        kind="stable",
        ignore_index=True,
    )


def _modeling_artifact_is_stale(
    modeling_dir: Path,
) -> bool:
    """Return whether the persisted modeling artifact must be rebuilt."""
    try:
        manifest = read_manifest(modeling_dir)
    except FileNotFoundError:
        return True

    schema_version = manifest.get("schema_version")
    data_version = manifest.get("data_version")

    return schema_version != CURRENT_SCHEMA_VERSION or data_version != CURRENT_DATA_VERSION


def build_model_inputs(*, all_years: bool, repo: Path | None = None) -> None:
    """Build canonical modeling inputs as Parquet artifacts.

    Produces one Away/Home-oriented row per completed game.

    When ``all_years`` is true, performs a full canonical rebuild.
    Otherwise, appends unseen game IDs when the persisted schema and
    data versions match, or performs a full rebuild when they do not.
    """
    repo = repo or repo_root()
    datasets = DatasetAccessor(repo)

    games: pd.DataFrame = loaders.load_games(repo)

    base_path: Path = dataset_path(repo, "modeling_base")
    full_path: Path = dataset_path(repo, "modeling_full")

    base_all: pd.DataFrame = build_home_away_modeling_table(games)

    if all_years or not base_path.exists() or not full_path.exists():
        # Full rebuild
        base_out: pd.DataFrame = base_all
        full_out: pd.DataFrame = run_features(
            df=base_out,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        writers.write_parquet(repo, "modeling_base", base_out)
        writers.write_parquet(repo, "modeling_full", full_out)
        write_manifest(
            full_out,
            feature_names=list(CANONICAL_FEATURES),
            feature_columns=(canonical_feature_columns()),
            modeling_dir=full_path.parent,
        )
        return

    # Incremental build: only process unseen GAME_ID rows
    base_existing: pd.DataFrame | None = load_parquet_if_exists(base_path)
    full_existing: pd.DataFrame | None = load_parquet_if_exists(full_path)
    if base_existing is None or full_existing is None:
        base_out = base_all
        full_out = run_features(
            df=base_out,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        writers.write_parquet(
            repo,
            "modeling_base",
            base_out,
        )
        writers.write_parquet(
            repo,
            "modeling_full",
            full_out,
        )
        write_manifest(
            full_out,
            feature_names=list(CANONICAL_FEATURES),
            feature_columns=(canonical_feature_columns()),
            modeling_dir=full_path.parent,
        )
        return

    # Check whether the existing modeling file's data_version matches the
    # current code. If not, incremental updates would silently preserve
    # stale rows produced by older (potentially buggy) feature code. Force
    # a full rebuild in that case.
    if _modeling_artifact_is_stale(full_path.parent):
        logger = logging.getLogger(__name__)
        logger.warning(
            "Modeling artifact versions are stale. "
            "Forcing a full canonical rebuild "
            "with schema_version=%d and "
            "data_version=%d.",
            CURRENT_SCHEMA_VERSION,
            CURRENT_DATA_VERSION,
        )

        base_out = base_all
        full_out = run_features(
            df=base_out,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        writers.write_parquet(
            repo,
            "modeling_base",
            base_out,
        )
        writers.write_parquet(
            repo,
            "modeling_full",
            full_out,
        )
        write_manifest(
            full_out,
            feature_names=list(CANONICAL_FEATURES),
            feature_columns=(canonical_feature_columns()),
            modeling_dir=full_path.parent,
        )
        return

    existing_game_ids: set = set(base_existing["GAME_ID"].unique().tolist())
    new_mask: pd.Series = ~base_all["GAME_ID"].isin(existing_game_ids)
    base_new: pd.DataFrame = base_all.loc[
        new_mask,
        :,
    ].copy()

    if base_new.empty:
        manifest_path = full_path.parent / "modeling_file_manifest.json"
        if not manifest_path.exists():
            write_manifest(
                full_existing,
                feature_names=list(CANONICAL_FEATURES),
                feature_columns=(canonical_feature_columns()),
                modeling_dir=full_path.parent,
            )
        return

    full_new: pd.DataFrame = run_features(
        df=base_new,
        feature_names=CANONICAL_FEATURES,
        datasets=datasets,
    )

    base_out = (
        pd.concat(
            [
                base_existing,
                base_new,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["GAME_ID"],
            keep="last",
        )
        .sort_values(
            [
                "YEAR",
                "WEEK_NUM",
                "GAME_DATE",
                "GAME_ID",
            ],
            kind="stable",
            ignore_index=True,
        )
    )

    full_out = (
        pd.concat(
            [
                full_existing,
                full_new,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["GAME_ID"],
            keep="last",
        )
        .sort_values(
            [
                "YEAR",
                "WEEK_NUM",
                "GAME_DATE",
                "GAME_ID",
            ],
            kind="stable",
            ignore_index=True,
        )
    )

    writers.write_parquet(
        repo,
        "modeling_base",
        base_out,
    )
    writers.write_parquet(
        repo,
        "modeling_full",
        full_out,
    )
    write_manifest(
        full_out,
        feature_names=list(CANONICAL_FEATURES),
        feature_columns=(canonical_feature_columns()),
        modeling_dir=full_path.parent,
    )
