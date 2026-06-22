# tests/integration/test_games_backfill.py

"""Integration tests for the game backfill flow.

Exercises the full lifecycle from a trained artifact through the
prediction archive to evaluation_df construction. Verifies dedup
behavior, --overwrite semantics, and the join with outcomes.

The tiny-model fixture is module-scoped so we train once per file rather
than per test (~5 seconds saved per additional test in the module).

Marked ``@pytest.mark.integration`` for separate selection from unit
tests. Runtime budget: under 1 minute total.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.dataframes import (
    make_games_from_modeling_df,
    make_games_modeling_df,
)
from tests.fixtures.helpers import (
    assert_archive_schema_valid,
    patch_minimal_param_grid,
)
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.evaluation.archive import _ARCHIVE_COLUMNS, load_prediction_log
from gridiron_edge.evaluation.backfill import backfill_model
from gridiron_edge.evaluation.metrics import build_evaluation_df
from gridiron_edge.models.game_prediction.predictor import (
    WinProbLogisticPredictor,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


# ---------------------------------------------------------------------------
# Module-scoped fixture: tiny trained model + populated repo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped repo with one trained win_prob_logistic artifact.

    Built once per test module. Each test in this file can call
    ``backfill_model(...)`` against this repo without re-training between
    tests.
    """
    tmp_path: Path = tmp_path_factory.mktemp("games_backfill")
    modeling = make_games_modeling_df()
    games = make_games_from_modeling_df(modeling)

    repo: Path = (
        MiniRepoBuilder(tmp_path)
        .with_games(games)
        .with_stadiums()
        .with_elo_state()
        .with_epa_by_game()
        .with_modeling_file(modeling)
        .build()
    )

    # Train a single tiny model. Each test in this module reuses this
    # trained artifact via the backfill path.
    predictor = WinProbLogisticPredictor()
    with patch_minimal_param_grid():
        predictor.train(modeling, repo=repo)

    return repo


# ---------------------------------------------------------------------------
# Test 1: backfill writes to the archive
# ---------------------------------------------------------------------------


class TestBackfillWritesArchive:
    """Verify backfill produces archive rows for the trained model."""

    def test_writes_archive_rows(self, trained_repo: Path) -> None:
        n_written: int = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        assert n_written > 0, "backfill_model returned 0 - no rows archived"

        log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        assert not log.empty
        assert (log["model_name"] == "win_prob").all()
        assert (log["model_type"] == "logistic").all()

    def test_archive_schema_valid(self, trained_repo: Path) -> None:
        """Archive rows conform to the canonical archive schema."""
        # If TestBackfillWritesArchive::test_writes_archive_rows hasn't run yet,
        # archive will be empty. Trigger one backfill first.
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        assert_archive_schema_valid(log)

    def test_archive_columns_match_canonical(self, trained_repo: Path) -> None:
        """Every column in _ARCHIVE_COLUMNS is present in the loaded log."""
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        for col in _ARCHIVE_COLUMNS:
            assert col in log.columns, f"missing archive column: {col}"


# ---------------------------------------------------------------------------
# Test 2: --overwrite replaces existing rows
# ---------------------------------------------------------------------------


class TestBackfillOverwrite:
    """Verify --overwrite semantics replace prior archive rows."""

    def test_overwrite_replaces_rows(self, trained_repo: Path) -> None:
        # First backfill - establishes baseline
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )
        first_log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        first_count: int = len(first_log)
        first_predicted_at: pd.Timestamp = first_log["predicted_at"].iloc[0]

        # Second backfill with --overwrite - should replace, not duplicate
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
            overwrite=True,
        )
        second_log: pd.DataFrame = load_prediction_log(repo=trained_repo)

        # Row count is identical (no duplicates)
        assert len(second_log) == first_count

        # predicted_at advanced (new run replaced old rows)
        second_predicted_at: pd.Timestamp = second_log["predicted_at"].iloc[0]
        assert second_predicted_at >= first_predicted_at


# ---------------------------------------------------------------------------
# Test 3: no --overwrite skips already-archived games
# ---------------------------------------------------------------------------


class TestBackfillIdempotent:
    """Verify backfill without --overwrite is a no-op when archive is current."""

    def test_no_op_when_already_archived(self, trained_repo: Path) -> None:
        # First backfill - establishes baseline
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )
        first_log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        first_count: int = len(first_log)

        # Second backfill without --overwrite - should write 0 new rows
        n_written: int = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        assert n_written == 0, f"expected 0 new rows on idempotent backfill, got {n_written}"

        # Archive row count unchanged
        second_log: pd.DataFrame = load_prediction_log(repo=trained_repo)
        assert len(second_log) == first_count


# ---------------------------------------------------------------------------
# Test 4: evaluation_df joins archive to outcomes
# ---------------------------------------------------------------------------


class TestBuildEvaluationDfJoin:
    """Verify build_evaluation_df produces a non-empty, well-formed result."""

    def test_evaluation_df_joins_outcomes(self, trained_repo: Path) -> None:
        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        df: pd.DataFrame = build_evaluation_df(
            model_name="win_prob",
            model_type="logistic",
            repo=trained_repo,
        )

        assert not df.empty
        assert "away_win_prob" in df.columns
        assert "away_team_won" in df.columns
        assert "model_name" in df.columns
        assert "model_type" in df.columns

        # All rows should have the right (name, type) pair
        assert (df["model_name"] == "win_prob").all()
        assert (df["model_type"] == "logistic").all()

        # Outcome column should be 0/1 integers
        assert df["away_team_won"].isin([0, 1]).all()

        # away_win_prob should be in [0, 1]
        finite_probs = df["away_win_prob"].dropna()
        if len(finite_probs) > 0:
            assert (finite_probs >= 0.0).all()
            assert (finite_probs <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 5: Integration tests for walk-forward backfill.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_walk_forward_smoke_logistic(tmp_path: Path) -> None:
    """Smoke test: walk-forward backfill completes without error for logistic.

    Uses a small season range to keep runtime manageable.
    """
    # This test requires a full modeling file fixture; it lives as a slow
    # integration test rather than a unit test because of the data dependency.
    # Skip gracefully if the test repo doesn't have a modeling file.
    pytest.skip(
        "Walk-forward smoke test requires fixture modeling data; "
        "implement when fixture infrastructure is available."
    )
