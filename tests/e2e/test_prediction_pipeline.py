# tests/e2e/test_prediction_pipeline.py
"""E2E: features pipeline - fit elo → build model inputs → verify output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.features.pipeline import build_model_inputs
from gridiron_edge.ratings.elo.fit import fit_elo


class TestFeaturePipelineEndToEnd:
    """Run the full feature pipeline on synthetic data via MiniRepoBuilder."""

    def test_pipeline_produces_modeling_base(self, tmp_path: Path) -> None:
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().build()
        fit_elo(all_years=True, repo=repo)
        build_model_inputs(all_years=True, repo=repo)

        base_path: Path = dataset_path(repo, "modeling_base")
        assert base_path.is_file(), "modeling_base not created"
        base: DataFrame = pd.read_parquet(base_path)
        assert len(base) > 0

    def test_pipeline_produces_modeling_full(self, tmp_path: Path) -> None:
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().build()
        fit_elo(all_years=True, repo=repo)
        build_model_inputs(all_years=True, repo=repo)

        full_path: Path = dataset_path(repo, "modeling_full")
        assert full_path.is_file(), "modeling_full not created"
        full: DataFrame = pd.read_parquet(full_path)
        assert len(full) > 0

    def test_modeling_full_has_feature_columns(self, tmp_path: Path) -> None:
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().build()
        fit_elo(all_years=True, repo=repo)
        build_model_inputs(all_years=True, repo=repo)

        full: DataFrame = pd.read_parquet(dataset_path(repo, "modeling_full"))
        # Core feature columns from the pipeline
        assert "HOME_FIELD" in full.columns
        assert "TEAM_A_ELO" in full.columns

    def test_modeling_tables_have_two_rows_per_game(self, tmp_path: Path) -> None:
        """Each game produces a TEAM_A and TEAM_B row."""
        repo = (
            MiniRepoBuilder(tmp_path)
            .with_games()  # default: 2 games
            .with_stadiums()
            .build()
        )
        fit_elo(all_years=True, repo=repo)
        build_model_inputs(all_years=True, repo=repo)

        base: DataFrame = pd.read_parquet(dataset_path(repo, "modeling_base"))
        # 2 games x 2 rows per game = 4 rows
        assert len(base) == 4

    def test_pipeline_with_elo_and_weather(self, tmp_path: Path) -> None:
        """Pipeline runs successfully with additional datasets populated."""
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().with_weather().build()
        fit_elo(all_years=True, repo=repo)
        build_model_inputs(all_years=True, repo=repo)

        full: DataFrame = pd.read_parquet(dataset_path(repo, "modeling_full"))
        assert len(full) > 0
        # Weather features should be present
        assert "IS_DOME" in full.columns
