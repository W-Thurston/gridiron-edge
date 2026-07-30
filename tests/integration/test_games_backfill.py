# tests/integration/test_games_backfill.py

"""Integration tests for immutable historical game forecast runs.

Exercises the lifecycle from a trained artifact through canonical
historical prediction generation and immutable forecast-event storage.
Verifies schema validity, invocation-level run identity, and coexistence
across repeated backfill runs.

The tiny-model fixture is module-scoped so the model is trained once per
test module.

Marked ``@pytest.mark.integration`` for separate test selection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures.dataframes import (
    make_games_from_modeling_df,
    make_games_modeling_df,
)
from tests.fixtures.helpers import patch_minimal_param_grid
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.evaluation.backfill import backfill_model
from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    load_forecast_events,
)
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
# backfill writes to the archive
# ---------------------------------------------------------------------------


class TestBackfillWritesForecastEvents:
    """Verify backfill writes one immutable historical forecast run."""

    def test_writes_backfilled_forecast_events(
        self,
        trained_repo: Path,
    ) -> None:
        n_written = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )

        assert n_written > 0

        events = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        assert len(events) >= n_written
        assert not events.empty
        assert list(events.columns) == FORECAST_EVENT_COLUMNS
        assert (events["role"] == ForecastRole.BACKFILLED.value).all()
        assert (events["model_name"] == "win_prob").all()
        assert (events["model_type"] == "logistic").all()

    def test_one_invocation_uses_one_run_id(
        self,
        trained_repo: Path,
    ) -> None:
        before = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        n_written = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )

        after = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        new_events = after.loc[
            ~after["event_id"].isin(before["event_id"]),
            :,
        ]

        assert len(new_events) == n_written
        assert new_events["run_id"].nunique() == 1
        assert new_events["generated_at"].nunique() == 1
        assert new_events["event_id"].is_unique


# ---------------------------------------------------------------------------
# --overwrite replaces existing rows
# ---------------------------------------------------------------------------


class TestRepeatedBackfillRuns:
    """Verify repeated reconstruction runs coexist immutably."""

    def test_repeated_runs_preserve_prior_events(
        self,
        trained_repo: Path,
    ) -> None:
        before = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        first_written = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )
        after_first = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        second_written = backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )
        after_second = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        first_events = after_first.loc[
            ~after_first["event_id"].isin(before["event_id"]),
            :,
        ]
        second_events = after_second.loc[
            ~after_second["event_id"].isin(after_first["event_id"]),
            :,
        ]

        assert first_written > 0
        assert second_written > 0
        assert len(first_events) == first_written
        assert len(second_events) == second_written

        first_run_ids = set(first_events["run_id"])
        second_run_ids = set(second_events["run_id"])

        assert len(first_run_ids) == 1
        assert len(second_run_ids) == 1
        assert first_run_ids.isdisjoint(second_run_ids)

        assert set(first_events["event_id"]).issubset(set(after_second["event_id"]))

    def test_same_game_and_model_can_exist_across_runs(
        self,
        trained_repo: Path,
    ) -> None:
        before = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )
        after_first = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        backfill_model(
            model_name="win_prob",
            model_type="logistic",
            mode="current-model",
            repo=trained_repo,
        )
        after_second = load_forecast_events(
            model_name="win_prob",
            model_type="logistic",
            role=ForecastRole.BACKFILLED,
            repo=trained_repo,
        )

        first_events = after_first.loc[
            ~after_first["event_id"].isin(before["event_id"]),
            :,
        ]
        second_events = after_second.loc[
            ~after_second["event_id"].isin(after_first["event_id"]),
            :,
        ]

        shared_games = set(first_events["game_id"]).intersection(second_events["game_id"])

        assert shared_games
        assert first_events["event_id"].is_unique
        assert second_events["event_id"].is_unique


# ---------------------------------------------------------------------------
# Integration tests for walk-forward backfill.
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
