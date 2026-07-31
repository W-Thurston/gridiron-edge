# src/gridiron_edge/features/pipeline.py

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
validate_ordering(FEATURES)
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
