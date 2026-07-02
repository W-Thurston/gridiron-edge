# tests/fixtures/repos.py
"""Composable MiniRepoBuilder for integration and e2e tests.

Unifies the two existing ``mini_repo`` fixtures (root conftest and
integration conftest) into a single builder-pattern class.

Usage::

    from tests.fixtures.repos import MiniRepoBuilder


    def test_pipeline(tmp_path):
        repo = MiniRepoBuilder(tmp_path).with_games().with_stadiums().with_elo_state().build()
        # repo is a Path - use it as the repo= argument
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.fixtures.dataframes import (
    make_elo_state,
    make_epa_by_game,
    make_games,
    make_stadiums,
    make_weather_enriched,
)

from gridiron_edge.datasets.registry import DATASETS
from gridiron_edge.datasets.writers import write_csv


class MiniRepoBuilder:
    """Composable test repository builder.

    Creates the directory skeleton expected by ``datasets.registry`` and
    populates only the datasets you need for a given test.  This keeps
    integration tests fast (no unnecessary I/O) and explicit about their
    data dependencies.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._root = tmp_path
        self._ensure_dirs()

    # -- internal helpers --------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create all directories defined in the dataset registry."""
        for spec in DATASETS.values():
            (self._root / spec.relpath).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, key: str, df: pd.DataFrame) -> MiniRepoBuilder:
        """Write a DataFrame to the registry-defined path.

        Uses write_csv for .csv targets and to_parquet for .parquet targets.
        """
        spec = DATASETS[key]
        path = self._root / spec.relpath
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            write_csv(self._root, key, df)
        return self

    # -- public builder API ------------------------------------------------

    def with_games(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a games dataset.  Uses factory defaults if *df* is None."""
        games = (
            df
            if df is not None
            else make_games(
                [
                    {
                        "GAME_ID": "2025_01_A_B",
                        "YEAR": "2025-2026",
                        "WEEK_NUM": 1,
                        "WINNER": "Team A",
                        "LOSER": "Team B",
                        "WIN_OR_TIE": 1,
                        "GAME_DATE": "2025-09-07",
                        "GAME_LOCATION": "NULL_VALUE",
                        "STADIUM": "Stadium A",
                        "PTS_WINNER": 27,
                        "PTS_LOSER": 20,
                        "VEGAS_LINE": -3.0,
                        "OVER_UNDER": 45.0,
                        "FAVORITED": "Team A",
                    },
                    {
                        "GAME_ID": "2025_02_B_A",
                        "YEAR": "2025-2026",
                        "WEEK_NUM": 2,
                        "WINNER": "Team B",
                        "LOSER": "Team A",
                        "WIN_OR_TIE": 1,
                        "GAME_DATE": "2025-09-14",
                        "GAME_LOCATION": "NULL_VALUE",
                        "STADIUM": "Stadium B",
                        "PTS_WINNER": 31,
                        "PTS_LOSER": 17,
                        "VEGAS_LINE": -7.0,
                        "OVER_UNDER": 48.0,
                        "FAVORITED": "Team B",
                    },
                ]
            )
        )
        return self._write("games", games)

    def with_stadiums(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a stadiums dataset.  Uses factory defaults if *df* is None."""
        return self._write("stadiums", df if df is not None else make_stadiums())

    def with_elo_state(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add an Elo state dataset.  Uses factory defaults if *df* is None."""
        return self._write("elo_state", df if df is not None else make_elo_state())

    def with_epa_by_game(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add an EPA-by-game dataset. Uses factory defaults if *df* is None."""
        return self._write(
            "epa_by_game",
            df if df is not None else make_epa_by_game(),
        )

    def with_weather(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a weather-enriched dataset.  Uses factory defaults if *df* is None."""
        return self._write(
            "weather_enriched",
            df if df is not None else make_weather_enriched(),
        )

    def with_modeling_file(
        self,
        df: pd.DataFrame | None = None,
        *,
        schema_version: int = 4,
    ) -> MiniRepoBuilder:
        """Add a modeling file (parquet) and its manifest.

        Writes the modeling parquet via the dataset registry's
        ``modeling_full`` path and writes the matching
        ``modeling_file_manifest.json`` next to it. The manifest is required
        by :func:`load_modeling_file` when called with
        ``required_schema_version=...`` (the case for predict paths in
        :class:`GamesPredictor`).

        Args:
            df: Modeling DataFrame. Uses :func:`make_games_modeling_df`
                defaults if None.
            schema_version: Schema version to declare in the manifest.

        Returns:
            Self, for builder chaining.
        """
        import json

        from tests.fixtures.dataframes import (
            make_games_modeling_df,
            make_modeling_manifest,
        )

        modeling: pd.DataFrame = df if df is not None else make_games_modeling_df()

        parquet_path: Path = self._root / DATASETS["modeling_full"].relpath
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        modeling.to_parquet(parquet_path, index=False)

        manifest_path: Path = parquet_path.parent / "modeling_file_manifest.json"
        manifest: dict = make_modeling_manifest(
            schema_version=schema_version,
            columns=list(modeling.columns),
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return self

    def with_player_stats(self, df: pd.DataFrame | None = None) -> MiniRepoBuilder:
        """Add a player-game logs parquet for the prop training path.

        Writes ``data/cleaned/player_game_logs.parquet``, the canonical
        path :func:`build_prop_features` reads from.

        Args:
            df: Player-game DataFrame. Uses :func:`make_props_modeling_df`
                defaults if None.

        Returns:
            Self, for builder chaining.
        """
        from tests.fixtures.dataframes import make_props_modeling_df

        stats: pd.DataFrame = df if df is not None else make_props_modeling_df()

        cleaned_dir: Path = self._root / "data" / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        path: Path = cleaned_dir / "player_game_logs.parquet"
        stats.to_parquet(path, index=False)

        return self

    def with_full_games_setup(self) -> MiniRepoBuilder:
        """Add games + EPA + modeling file + stadiums + Elo state.

        Convenience method for tests that need the full game-side fixture
        suite. Equivalent to chaining each builder individually with
        defaults; provided so tests don't need to know which datasets
        the game pipeline reads.

        Returns:
            Self, for builder chaining.
        """
        return (
            self.with_games()
            .with_stadiums()
            .with_elo_state()
            .with_epa_by_game()
            .with_modeling_file()
        )

    def with_champion_manifest(
        self,
        *,
        win_prob_model_type: str = "elo",
        win_prob_metrics: dict | None = None,
        extra_models: dict[str, dict] | None = None,
    ) -> MiniRepoBuilder:
        """Add a champion manifest.

        Writes ``data/output/champions/champions.json`` with a single
        win_prob entry by default. Callers can override the win_prob
        model_type and metrics, and can add additional model_name
        entries (e.g., total, qb_pass_yards) via ``extra_models``.

        Args:
            win_prob_model_type: Registered model_type for win_prob.
                Defaults to "elo" (matches CLI default backfill).
            win_prob_metrics: Metrics dict for the win_prob entry.
                Defaults to sensible test values.
            extra_models: Additional entries keyed by model_name.

        Returns:
            Self, for builder chaining.
        """
        import json

        manifest_dir: Path = self._root / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        default_metrics: dict[str, float] = {"brier": 0.213, "ece": 0.041, "auc": 0.721}
        models: dict[str, dict] = {
            "win_prob": {
                "model_type": win_prob_model_type,
                "promoted_at": "2026-07-01T14:00:00",
                "source_run_id": "TEST_RUN",
                "metrics": win_prob_metrics or default_metrics,
            },
        }
        if extra_models:
            models.update(extra_models)

        manifest: dict[str, dict[str, dict] | int | str] = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": models,
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest, indent=2))
        return self

    def with_predictions_archive(
        self,
        df: pd.DataFrame,
    ) -> MiniRepoBuilder:
        """Add a predictions archive parquet.

        Writes ``data/output/predictions/predictions_log.parquet``, the
        path that ``evaluation.archive.load_prediction_log`` reads from.

        Args:
            df: Predictions DataFrame matching the archive schema.

        Returns:
            Self, for builder chaining.
        """
        predictions_dir: Path = self._root / "data" / "output" / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(predictions_dir / "predictions_log.parquet", index=False)
        return self

    def build(self) -> Path:
        """Return the repository root path."""
        return self._root
