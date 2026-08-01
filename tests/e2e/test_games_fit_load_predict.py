# tests/e2e/test_games_fit_load_predict.py

"""End-to-end tests for the games training and prediction lifecycle.

Each test trains a tiny model on synthetic data, persists the artifact
through ``ArtifactStore``, loads it back through ``GamesModel``, and
asserts the predictions are reasonable. These tests catch the class of
bugs where training reports a clean Brier but production prediction
produces nonsense (the scaler-not-applied-at-predict-time pattern).

The hyperparameter grids are minimized via
:func:`patch_minimal_param_grid` so each test runs in single-digit
seconds rather than minutes.

Marked ``@pytest.mark.slow`` - runs on PR but not on every commit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_games_modeling_df
from tests.fixtures.helpers import (
    assert_predictions_reasonable,
    patch_minimal_param_grid,
)
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.elo.model import WinProbEloModel
from gridiron_edge.models.game_prediction.model import (
    TotalRandomForestModel,
    TotalXGBoostModel,
    WinProbLogisticModel,
    WinProbRandomForestModel,
    WinProbXGBoostModel,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def games_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped repo with games + modeling file + EPA data.

    The canonical modeling DataFrame is generated first, then a matching
    cleaned-games fixture is derived from its explicit Away/Home identity
    and scores.
    """
    from tests.fixtures.dataframes import (
        make_games_from_modeling_df,
        make_games_modeling_df,
    )

    tmp_path: Path = tmp_path_factory.mktemp("games_e2e")
    modeling = make_games_modeling_df()
    games = make_games_from_modeling_df(modeling)

    return (
        MiniRepoBuilder(tmp_path)
        .with_games(games)
        .with_stadiums()
        .with_elo_state()
        .with_epa_by_game()
        .with_modeling_file(modeling)
        .build()
    )


@pytest.fixture
def modeling_df() -> pd.DataFrame:
    """Synthetic modeling DataFrame ready for training."""
    return make_games_modeling_df()


# ---------------------------------------------------------------------------
# Helper: train + verify artifact + load + predict
# ---------------------------------------------------------------------------


def _fit_load_predict_classification(
    model: (WinProbLogisticModel | WinProbRandomForestModel | WinProbXGBoostModel),
    *,
    repo: Path,
    modeling_df: pd.DataFrame,
    enforce_scaler_band: bool = False,
) -> None:
    """Train classification model, persist, load, predict, assert reasonable.

    Shared by all three win_prob tests since their lifecycles are identical
    once the model class differs. Splits the train and predict_historical
    calls so the artifact is genuinely loaded from disk (not cached in
    memory) before predictions are made.
    """
    with patch_minimal_param_grid():
        metadata = model.train(modeling_df, repo=repo)

    # Verify artifact landed on disk
    store = ArtifactStore(repo)
    assert store.is_trained(model.model_name, model.model_type), (
        f"artifact missing at {store.artifact_dir(model.model_name, model.model_type)}"
    )
    assert metadata.task == "classification"

    holdout_brier = metadata.metrics.get("brier")

    assert holdout_brier is not None
    assert holdout_brier > 0.0
    assert holdout_brier < 0.50
    assert "brier" in metadata.metrics
    assert not hasattr(
        metadata,
        "holdout_brier",
    )

    # Now run the predict path - this is where the scaler bug surfaced.
    # Fresh model instance to confirm we're loading from disk, not
    # using in-memory state from training.
    fresh_predictor = type(model)()
    result_df = fresh_predictor.predict_historical(pd.DataFrame(), repo=repo)

    assert not result_df.empty

    expected_columns = {
        "season",
        "week",
        "game_id",
        "game_date",
        "away_team",
        "home_team",
        "away_win_prob",
        "home_win_prob",
    }

    assert expected_columns <= set(result_df.columns)
    assert result_df["game_id"].is_unique

    assert (result_df["away_win_prob"] + result_df["home_win_prob"]).to_numpy() == pytest.approx(
        1.0
    )

    # The critical regression assertion: catches the scaler-not-applied bug.
    # On the broken path, std was ~0.485; on the fixed path, it's ~0.15.
    assert_predictions_reasonable(
        result_df["away_win_prob"],
        task="classification",
        allow_extreme=not enforce_scaler_band,
        name=(f"{model.model_name}/{model.model_type}"),
    )

    assert result_df["away_win_prob"].nunique() > 1


def _fit_load_predict_regression(
    model: TotalRandomForestModel | TotalXGBoostModel,
    *,
    repo: Path,
    modeling_df: pd.DataFrame,
) -> None:
    """Train regression model, persist, load, predict, assert reasonable."""
    with patch_minimal_param_grid():
        metadata = model.train(modeling_df, repo=repo)

    store = ArtifactStore(repo)
    assert store.is_trained(model.model_name, model.model_type), (
        f"artifact missing at {store.artifact_dir(model.model_name, model.model_type)}"
    )
    assert metadata.task == "regression"

    holdout_mae = metadata.metrics.get("mae")

    assert holdout_mae is not None
    assert holdout_mae > 0.0
    assert "mae" in metadata.metrics
    assert not hasattr(
        metadata,
        "holdout_mae",
    )

    # Fresh model for the load path
    fresh_predictor = type(model)()
    result_df = fresh_predictor.predict_historical(pd.DataFrame(), repo=repo)

    assert not result_df.empty

    expected_columns = {
        "season",
        "week",
        "game_id",
        "game_date",
        "away_team",
        "home_team",
        "model_total",
    }

    assert expected_columns <= set(result_df.columns)
    assert result_df["game_id"].is_unique

    assert_predictions_reasonable(
        result_df["model_total"],
        task="regression",
        name=f"{model.model_name}/{model.model_type}",
    )


# ---------------------------------------------------------------------------
# Classification: win_prob_logistic
# ---------------------------------------------------------------------------


class TestWinProbLogistic:
    """End-to-end lifecycle for win_prob_logistic.

    This test is the explicit regression for the scaler-not-applied bug:
    logistic is the only win_prob model that uses a StandardScaler at
    training time. Without the scaler being loaded and applied at predict
    time, probabilities slam to corners (std ~0.5) instead of staying in
    a calibrated band (std ~0.15).
    """

    def test_fit_load_predict(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        model = WinProbLogisticModel()
        _fit_load_predict_classification(
            model,
            repo=games_repo,
            modeling_df=modeling_df,
            enforce_scaler_band=True,
        )

    def test_scaler_artifact_persisted(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        """Verify the scaler is actually written to disk alongside the model."""
        model = WinProbLogisticModel()
        with patch_minimal_param_grid():
            model.train(modeling_df, repo=games_repo)

        store = ArtifactStore(games_repo)
        scaler = store.load_scaler(model.model_name, model.model_type)
        # If the scaler doesn't get persisted, the load-side bug returns -
        # this catches a regression in the artifact persistence layer.
        assert scaler is not None, "scaler artifact missing - load_scaler returned None"


# ---------------------------------------------------------------------------
# Classification: win_prob_random_forest
# ---------------------------------------------------------------------------


class TestWinProbRandomForest:
    """End-to-end lifecycle for win_prob_random_forest."""

    def test_fit_load_predict(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        model = WinProbRandomForestModel()
        _fit_load_predict_classification(
            model,
            repo=games_repo,
            modeling_df=modeling_df,
        )


# ---------------------------------------------------------------------------
# Classification: win_prob_xgboost
# ---------------------------------------------------------------------------


class TestWinProbXGBoost:
    """End-to-end lifecycle for win_prob_xgboost."""

    def test_fit_load_predict(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        model = WinProbXGBoostModel()
        _fit_load_predict_classification(
            model,
            repo=games_repo,
            modeling_df=modeling_df,
        )


# ---------------------------------------------------------------------------
# Regression: total_random_forest
# ---------------------------------------------------------------------------


class TestTotalRandomForest:
    """End-to-end lifecycle for total_random_forest."""

    def test_fit_load_predict(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        model = TotalRandomForestModel()
        _fit_load_predict_regression(
            model,
            repo=games_repo,
            modeling_df=modeling_df,
        )


# ---------------------------------------------------------------------------
# Regression: total_xgboost
# ---------------------------------------------------------------------------


class TestTotalXGBoost:
    """End-to-end lifecycle for total_xgboost."""

    def test_fit_load_predict(self, games_repo: Path, modeling_df: pd.DataFrame) -> None:
        model = TotalXGBoostModel()
        _fit_load_predict_regression(
            model,
            repo=games_repo,
            modeling_df=modeling_df,
        )


# ---------------------------------------------------------------------------
# Elo: load + predict only (no training)
# ---------------------------------------------------------------------------


class TestWinProbElo:
    """End-to-end lifecycle for win_prob_elo.

    Elo doesn't fit (it's analytic). The test verifies that the model
    correctly reads Elo state and produces reasonable predictions. Catches
    regressions in the Elo migration path - e.g., if the model's load
    path stops reading the Elo state table correctly.
    """

    def test_predict_historical(self, games_repo: Path) -> None:
        from gridiron_edge.datasets import loaders

        model = WinProbEloModel()
        # Elo's predict_historical reads games from the games DataFrame
        # rather than the modeling file. Loading via the same path the
        # backfill CLI uses keeps the test honest about real usage.
        games_raw: DataFrame = loaders.load_games(games_repo)
        games = games_raw.loc[games_raw["WIN_OR_TIE"].notna(), :].copy()
        result_df: DataFrame = model.predict_historical(games, repo=games_repo)

        assert not result_df.empty

        expected_columns = {
            "season",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "away_elo",
            "home_elo",
            "away_win_prob",
            "home_win_prob",
            "model_spread",
        }

        assert expected_columns <= set(result_df.columns)
        assert result_df["game_id"].is_unique

        assert (
            result_df["away_win_prob"] + result_df["home_win_prob"]
        ).to_numpy() == pytest.approx(1.0)

        # Allow tighter std for Elo because the synthetic ratings are
        # very similar (both ~1500) so Elo correctly predicts probabilities
        # near 0.5. This test verifies the lifecycle, not Elo's predictive
        # power; production Elo uses ratings with much wider variance.
        assert_predictions_reasonable(
            result_df["away_win_prob"],
            task="classification",
            allow_extreme=True,
            name="win_prob/elo",
        )
