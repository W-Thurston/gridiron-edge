# tests/unit/models/test_games_predictor.py

"""Tests for GamesPredictor + composite-key registrations + build_game_predictions.

Covers the static surface of the predictor classes plus the module-level
``build_game_predictions`` helper:
    - All 5 composite keys are registered.
    - Each subclass has the right (model_name, model_type, spec).
    - GamesPredictor delegates train() to the right trainer.
    - is_trained() delegates to ArtifactStore with the (name, type) pair.
    - predict_historical() / predict_upcoming() return empty DataFrames
      when artifacts are missing (graceful fallback).
    - _maybe_predict_totals() returns None when:
        * called from a total predictor (no recursion).
        * the configured total model is not trained.
    - build_game_predictions() constructs the standard archive schema.

End-to-end fit-and-predict smoke tests against real modeling data are
deferred to slow integration tests; this unit-test file exercises the
static surface only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.models.base import PredictorSpec, Trainable
from gridiron_edge.models.game_prediction.base import (
    GameModelType,
)
from gridiron_edge.models.game_prediction.predictor import (
    _TRAINER_FOR_NAME,
    GamesPredictor,
    TotalRandomForestPredictor,
    TotalXGBoostPredictor,
    WinProbLogisticPredictor,
    WinProbRandomForestPredictor,
    WinProbXGBoostPredictor,
    build_game_predictions,
)
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer
from gridiron_edge.models.registry import PredictorRegistry

# ---------------------------------------------------------------------------
# Trainer dispatch table
# ---------------------------------------------------------------------------


class TestTrainerDispatch:
    """``_TRAINER_FOR_NAME`` maps both model_names to the right trainer."""

    def test_win_prob_maps_to_win_prob_trainer(self) -> None:
        assert _TRAINER_FOR_NAME["win_prob"] is WinProbTrainer

    def test_total_maps_to_total_trainer(self) -> None:
        assert _TRAINER_FOR_NAME["total"] is TotalTrainer

    def test_exactly_two_entries(self) -> None:
        assert set(_TRAINER_FOR_NAME.keys()) == {"win_prob", "total"}


# ---------------------------------------------------------------------------
# Composite-key registrations
# ---------------------------------------------------------------------------


_EXPECTED_REGISTRATIONS: list[tuple[str, type[GamesPredictor], str, str]] = [
    ("win_prob_logistic", WinProbLogisticPredictor, "win_prob", "logistic"),
    (
        "win_prob_random_forest",
        WinProbRandomForestPredictor,
        "win_prob",
        "random_forest",
    ),
    ("win_prob_xgboost", WinProbXGBoostPredictor, "win_prob", "xgboost"),
    ("total_random_forest", TotalRandomForestPredictor, "total", "random_forest"),
    ("total_xgboost", TotalXGBoostPredictor, "total", "xgboost"),
]


class TestCompositeRegistrations:
    """All 5 composite keys are registered and resolve to the right class."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_registered(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        # Importing predictor.py at test-module load time triggered registration.
        cls = PredictorRegistry.get(registry_key)
        assert cls is predictor_cls

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_class_attributes(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.model_name == model_name
        assert predictor_cls.model_type == model_type
        assert predictor_cls.spec.name == registry_key

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_is_trainable(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.spec.trainable is True

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_implements_trainable_protocol(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        instance: GamesPredictor = predictor_cls()
        assert isinstance(instance, Trainable)


# ---------------------------------------------------------------------------
# GamesPredictor accessors
# ---------------------------------------------------------------------------


class TestGamesPredictorAccessors:
    """``_trainer``, ``_game_model_spec``, ``_task`` resolve correctly."""

    def test_win_prob_trainer_instance(self) -> None:
        pred = WinProbRandomForestPredictor()
        trainer = pred._trainer()
        assert isinstance(trainer, WinProbTrainer)

    def test_total_trainer_instance(self) -> None:
        pred = TotalRandomForestPredictor()
        trainer = pred._trainer()
        assert isinstance(trainer, TotalTrainer)

    def test_win_prob_task(self) -> None:
        assert WinProbLogisticPredictor()._task() == "classification"
        assert WinProbRandomForestPredictor()._task() == "classification"
        assert WinProbXGBoostPredictor()._task() == "classification"

    def test_total_task(self) -> None:
        assert TotalRandomForestPredictor()._task() == "regression"
        assert TotalXGBoostPredictor()._task() == "regression"

    def test_default_total_model_type(self) -> None:
        # Default is random_forest for all win_prob variants.
        assert WinProbLogisticPredictor.default_total_model_type == "random_forest"
        assert WinProbRandomForestPredictor.default_total_model_type == "random_forest"
        assert WinProbXGBoostPredictor.default_total_model_type == "random_forest"


# ---------------------------------------------------------------------------
# is_trained delegation
# ---------------------------------------------------------------------------


class TestIsTrained:
    """``is_trained`` returns False when the artifact directory is empty."""

    def test_returns_false_when_no_artifact(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        assert pred.is_trained(repo=tmp_path) is False

    def test_returns_false_for_each_composite(self, tmp_path: Path) -> None:
        for _key, cls, _name, _type in _EXPECTED_REGISTRATIONS:
            assert cls().is_trained(repo=tmp_path) is False


# ---------------------------------------------------------------------------
# predict_historical / predict_upcoming graceful fallback
# ---------------------------------------------------------------------------


class TestPredictGracefulFallback:
    """When no artifact exists, predict methods return an empty DataFrame."""

    def test_predict_historical_classification_empty(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        empty_games = pd.DataFrame()
        out = pred.predict_historical(empty_games, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_historical_regression_empty(self, tmp_path: Path) -> None:
        pred = TotalRandomForestPredictor()
        empty_games = pd.DataFrame()
        out = pred.predict_historical(empty_games, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_upcoming_classification_empty(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        empty_schedule = pd.DataFrame()
        out = pred.predict_upcoming(empty_schedule, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_upcoming_regression_empty(self, tmp_path: Path) -> None:
        pred = TotalRandomForestPredictor()
        empty_schedule = pd.DataFrame()
        out = pred.predict_upcoming(empty_schedule, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty


# ---------------------------------------------------------------------------
# _maybe_predict_totals
# ---------------------------------------------------------------------------


class TestMaybePredictTotals:
    """``_maybe_predict_totals`` skips recursion and missing artifacts."""

    def test_total_predictor_returns_none(self, tmp_path: Path) -> None:
        """A total predictor never tries to predict its own totals."""
        pred = TotalRandomForestPredictor()
        empty_df = pd.DataFrame()
        result = pred._maybe_predict_totals(empty_df, repo=tmp_path)
        assert result is None

    def test_win_prob_returns_none_when_total_not_trained(self, tmp_path: Path) -> None:
        """Win-prob silently omits totals when the total model is missing."""
        pred = WinProbRandomForestPredictor()
        empty_df = pd.DataFrame()
        result = pred._maybe_predict_totals(empty_df, repo=tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# Type-coverage smoke: every predictor's spec round-trips through PredictorSpec
# ---------------------------------------------------------------------------


class TestPredictorSpecShape:
    """Every composite spec is a fully-formed ``PredictorSpec``."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_type(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert isinstance(predictor_cls.spec, PredictorSpec)

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_description_nonempty(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.spec.description.strip() != ""


# ---------------------------------------------------------------------------
# Model type round-trip through GameModelType enum
# ---------------------------------------------------------------------------


class TestModelTypeEnumRoundtrip:
    """Each predictor's ``model_type`` string is a valid ``GameModelType``."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_model_type_is_valid_enum(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        # GameModelType("logistic") / ("random_forest") / ("xgboost") must succeed.
        assert GameModelType(predictor_cls.model_type) in GameModelType


# ---------------------------------------------------------------------------
# Module-level helper: build_game_predictions
# ---------------------------------------------------------------------------


class TestBuildGamePredictions:
    """Tests for canonical classification prediction rows.

    The function lives at module scope in ``predictor.py`` (not a method
    of ``GamesPredictor``) because it's a pure data-shape helper used by
    the classification prediction path.
    """

    def _make_modeling_df(self) -> pd.DataFrame:
        """Minimal modeling DataFrame with two rows per game (home/away)."""
        return pd.DataFrame(
            {
                "GAME_ID": ["G1", "G1", "G2", "G2"],
                "TEAM_A": ["Chiefs", "Ravens", "Bills", "Dolphins"],
                "TEAM_B": ["Ravens", "Chiefs", "Dolphins", "Bills"],
                "YEAR": ["2024-2025"] * 4,
                "WEEK_NUM": [1, 1, 1, 1],
                "HOME_FIELD": [0, 1, 0, 1],
                "GAME_DATE": [
                    "2024-09-05",
                    "2024-09-05",
                    "2024-09-06",
                    "2024-09-06",
                ],
                "TEAM_A_ELO": [
                    1520.0,
                    1480.0,
                    1510.0,
                    1500.0,
                ],
                "TEAM_B_ELO": [
                    1480.0,
                    1520.0,
                    1500.0,
                    1510.0,
                ],
            }
        )

    def test_one_row_per_game(self) -> None:
        """Output has one row per game, not two."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(
            df,
            probs,
        )
        assert len(result) == 2

    def test_away_team_perspective(self) -> None:
        """Away team probability matches the HOME_FIELD==0 row."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(
            df,
            probs,
        )
        g1 = result[result["game_id"] == "G1"].iloc[0]
        assert g1["away_win_prob"] == pytest.approx(0.45)
        assert g1["home_win_prob"] == pytest.approx(0.55)

    def test_totals_included_when_provided(self) -> None:
        """model_total column present when totals are passed."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])
        totals = pd.Series([44.0, 44.0, 48.0, 48.0], index=df.index)

        result = build_game_predictions(
            df,
            probs,
            totals=totals,
        )
        assert "model_total" in result.columns
        assert result["model_total"].notna().all()

    def test_totals_absent_when_not_provided(self) -> None:
        """model_total column absent when totals are not passed."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(
            df,
            probs,
        )
        assert "model_total" not in result.columns

    def test_required_columns_present(self) -> None:
        """Output contains all base archive columns."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(
            df,
            probs,
        )
        required: set[str] = {
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
        }
        assert required.issubset(set(result.columns))
        assert required.issubset(set(result.columns))
        assert "predicted_at" not in result.columns
        assert "is_backfilled" not in result.columns
        assert "model_name" not in result.columns
        assert "model_type" not in result.columns

    def _make_neutral_site_df(self) -> pd.DataFrame:
        """Modeling DataFrame with a neutral-site game (both rows HOME_FIELD=0)."""
        return pd.DataFrame(
            {
                "GAME_ID": ["G_NEUTRAL", "G_NEUTRAL"],
                "TEAM_A": ["Chiefs", "Ravens"],
                "TEAM_B": ["Ravens", "Chiefs"],
                "YEAR": ["2024-2025"] * 2,
                "WEEK_NUM": [5, 5],
                "HOME_FIELD": [0, 0],  # Neutral: both 0
            }
        )

    def test_neutral_site_produces_one_row(self) -> None:
        """Neutral games must produce exactly one prediction row, not two."""
        df = self._make_neutral_site_df()
        probs = np.array([0.55, 0.45])  # P(Chiefs beat Ravens), P(Ravens beat Chiefs)
        result = build_game_predictions(
            df,
            probs,
        )
        assert len(result) == 1

    def test_neutral_site_deterministic_labeling(self) -> None:
        """Neutral games label the alphabetically-first team as away."""
        df = self._make_neutral_site_df()
        probs = np.array([0.55, 0.45])
        result = build_game_predictions(
            df,
            probs,
        )
        # Chiefs < Ravens alphabetically, so Chiefs is labeled "away"
        assert result["away_team"].iloc[0] == "Chiefs"
        assert result["home_team"].iloc[0] == "Ravens"

    def test_neutral_site_probability_matches_labeling(self) -> None:
        """The away_win_prob must match the labeled away team's win probability."""
        df = self._make_neutral_site_df()
        # Row 1: TEAM_A=Chiefs, prob = P(Chiefs beats Ravens) = 0.55
        # Row 2: TEAM_A=Ravens, prob = P(Ravens beats Chiefs) = 0.45
        probs = np.array([0.55, 0.45])
        result = build_game_predictions(
            df,
            probs,
        )
        # Chiefs is labeled away; away_win_prob should be P(Chiefs beats Ravens)
        assert result["away_win_prob"].iloc[0] == pytest.approx(0.55)
        assert result["home_win_prob"].iloc[0] == pytest.approx(0.45)

    def test_neutral_site_stable_across_input_order(self) -> None:
        """Reversing input row order must produce the same output."""
        df1 = self._make_neutral_site_df()
        probs1 = np.array([0.55, 0.45])

        # Reverse input rows
        df2 = df1.iloc[::-1].reset_index(drop=True)
        probs2 = probs1[::-1]

        result1 = build_game_predictions(
            df1,
            probs1,
        )
        result2 = build_game_predictions(
            df2,
            probs2,
        )

        # Same away team, same probability, regardless of input order
        assert result1["away_team"].iloc[0] == result2["away_team"].iloc[0]
        assert result1["away_win_prob"].iloc[0] == pytest.approx(result2["away_win_prob"].iloc[0])

    def test_mixed_standard_and_neutral_games(self) -> None:
        """Standard and neutral games can coexist in one call."""
        df = pd.DataFrame(
            {
                "GAME_ID": ["G_STD", "G_STD", "G_NEUTRAL", "G_NEUTRAL"],
                "TEAM_A": ["Bills", "Dolphins", "Chiefs", "Ravens"],
                "TEAM_B": ["Dolphins", "Bills", "Ravens", "Chiefs"],
                "YEAR": ["2024-2025"] * 4,
                "WEEK_NUM": [1, 1, 5, 5],
                "HOME_FIELD": [0, 1, 0, 0],  # Std: Dolphins home. Neutral: both 0.
            }
        )
        probs = np.array([0.60, 0.40, 0.55, 0.45])
        result = build_game_predictions(
            df,
            probs,
        )
        assert len(result) == 2

        std_row = result[result["game_id"] == "G_STD"].iloc[0]
        # Bills (HOME_FIELD=0) is the away team
        assert std_row["away_team"] == "Bills"
        assert std_row["away_win_prob"] == pytest.approx(0.60)

        neutral_row = result[result["game_id"] == "G_NEUTRAL"].iloc[0]
        # Chiefs alphabetically first, labeled away
        assert neutral_row["away_team"] == "Chiefs"
        assert neutral_row["away_win_prob"] == pytest.approx(0.55)

    def test_preserves_game_date_and_elo_values(self) -> None:
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(
            df,
            probs,
        )

        g1 = result.loc[result["game_id"] == "G1"].iloc[0]

        assert g1["game_date"] == "2024-09-05"
        assert g1["away_elo"] == pytest.approx(1520.0)
        assert g1["home_elo"] == pytest.approx(1480.0)
