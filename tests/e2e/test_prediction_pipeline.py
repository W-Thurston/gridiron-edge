# tests/e2e/test_prediction_pipeline.py
"""E2E tests for the canonical feature pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from tests.fixtures.repos import (
    MiniRepoBuilder,
)

from gridiron_edge.datasets.registry import (
    dataset_path,
)
from gridiron_edge.features.pipeline import (
    build_model_inputs,
)
from gridiron_edge.ratings.elo.fit import (
    fit_elo,
)


def _build_repo(
    tmp_path: Path,
    *,
    weather: bool = False,
) -> Path:
    """Build a repository for canonical pipeline tests."""
    builder = MiniRepoBuilder(tmp_path).with_games().with_stadiums()

    if weather:
        builder = builder.with_weather()

    repo = builder.build()

    fit_elo(
        all_years=True,
        repo=repo,
    )
    build_model_inputs(
        all_years=True,
        repo=repo,
    )

    return repo


class TestFeaturePipelineEndToEnd:
    """Exercise the active one-row canonical feature pipeline."""

    def test_pipeline_produces_modeling_base(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _build_repo(tmp_path)

        base_path = dataset_path(
            repo,
            "modeling_base",
        )

        assert base_path.is_file()

        base = pd.read_parquet(base_path)

        assert len(base) == 2
        assert base["GAME_ID"].is_unique

    def test_pipeline_produces_modeling_full(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _build_repo(tmp_path)

        full_path = dataset_path(
            repo,
            "modeling_full",
        )

        assert full_path.is_file()

        full = pd.read_parquet(full_path)

        assert len(full) == 2
        assert full["GAME_ID"].is_unique

    def test_modeling_full_has_canonical_features(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _build_repo(tmp_path)

        full: DataFrame = pd.read_parquet(
            dataset_path(
                repo,
                "modeling_full",
            )
        )

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
            assert retired not in full.columns

    def test_modeling_tables_have_one_row_per_game(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _build_repo(tmp_path)

        base: DataFrame = pd.read_parquet(
            dataset_path(
                repo,
                "modeling_base",
            )
        )
        full: DataFrame = pd.read_parquet(
            dataset_path(
                repo,
                "modeling_full",
            )
        )

        assert len(base) == 2
        assert len(full) == 2
        assert base["GAME_ID"].is_unique
        assert full["GAME_ID"].is_unique

    def test_pipeline_with_weather(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _build_repo(
            tmp_path,
            weather=True,
        )

        full: DataFrame = pd.read_parquet(
            dataset_path(
                repo,
                "modeling_full",
            )
        )

        assert len(full) == 2
        assert full["GAME_ID"].is_unique
        assert "IS_DOME" in full.columns
        assert "TEMP_F" in full.columns
