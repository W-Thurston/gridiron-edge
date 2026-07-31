# tests/integration/test_features_pipeline.py

"""Integration tests for canonical modeling artifact generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridiron_edge.datasets.registry import (
    dataset_path,
)
from gridiron_edge.features.manifest import (
    CURRENT_DATA_VERSION,
    CURRENT_SCHEMA_VERSION,
    read_manifest,
)
from gridiron_edge.features.pipeline import (
    CANONICAL_FEATURES,
    build_model_inputs,
    canonical_feature_columns,
)
from gridiron_edge.ratings.elo.fit import (
    fit_elo,
)


def test_build_model_inputs_full_rebuild(
    mini_repo: Path,
) -> None:
    fit_elo(
        all_years=True,
        repo=mini_repo,
    )
    build_model_inputs(
        all_years=True,
        repo=mini_repo,
    )

    base = pd.read_parquet(
        dataset_path(
            mini_repo,
            "modeling_base",
        )
    )
    full = pd.read_parquet(
        dataset_path(
            mini_repo,
            "modeling_full",
        )
    )

    assert len(base) == 2
    assert len(full) == 2

    assert base["GAME_ID"].is_unique
    assert full["GAME_ID"].is_unique

    for column in (
        "AWAY_TEAM",
        "HOME_TEAM",
        "HOME_WIN",
        "ACTUAL_MARGIN",
        "ACTUAL_TOTAL",
    ):
        assert column in base.columns

    for column in (
        "AWAY_ELO",
        "HOME_ELO",
        "AWAY_KM_TRAVELED",
        "HOME_KM_TRAVELED",
        "GAME_SITE_ALTITUDE",
    ):
        assert column in full.columns

    for retired in (
        "TEAM_A",
        "TEAM_B",
        "HOME_FIELD",
        "RESULT",
    ):
        assert retired not in base.columns
        assert retired not in full.columns

    assert (base["ACTUAL_MARGIN"] == (base["HOME_SCORE"] - base["AWAY_SCORE"])).all()

    assert (base["ACTUAL_TOTAL"] == (base["HOME_SCORE"] + base["AWAY_SCORE"])).all()

    manifest = read_manifest(
        dataset_path(
            mini_repo,
            "modeling_full",
        ).parent
    )

    assert manifest["feature_names"] == list(CANONICAL_FEATURES)
    assert manifest["feature_columns"] == (canonical_feature_columns())
    assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
    assert manifest["data_version"] == CURRENT_DATA_VERSION
    assert manifest["row_count"] == 2
