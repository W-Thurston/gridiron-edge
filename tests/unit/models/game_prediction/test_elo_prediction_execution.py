# tests/unit/models/game_prediction/test_elo_prediction_execution.py
"""Tests for live Elo prediction execution with immutable input evidence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionArtifactKind,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
)
from gridiron_edge.models.base import GameModel
from gridiron_edge.models.elo.model import WinProbEloModel
from gridiron_edge.models.game_prediction.prediction_execution import (
    ELO_FORMULA_ID,
)

COMMIT = "a" * 40
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _revision() -> SourceRevision:
    return SourceRevision(
        commit=COMMIT,
        tracked_worktree_clean=True,
    )


def _sources() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_A,
            size_bytes=100,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.metadata.json",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_B,
            size_bytes=200,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest="c" * 64,
            size_bytes=300,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_wk_by_wk_cleaned.csv",
            state=PredictionSourceState.PRESENT,
            content_digest="d" * 64,
            size_bytes=400,
        ),
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["game-2", "game-1"],
            "YEAR": ["2026-2027", "2026-2027"],
            "WEEK_NUM": [2, 2],
            "AWAY_TEAM": ["Away B", "Away A"],
            "HOME_TEAM": ["Home B", "Home A"],
        }
    )


def _elo_state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NFL_TEAM": ["Away A", "Home A", "Away B", "Home B"],
            "NFL_YEAR": ["2026-2027"] * 4,
            "NFL_WEEK": [2] * 4,
            "ELO": [1500.0, 1510.0, 1520.0, 1480.0],
        }
    )


class TestLegacyUpcomingEloPrediction:
    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    def test_legacy_path_preserves_existing_display_schema(
        self,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        load_elo_state.return_value = _elo_state()

        result = WinProbEloModel().predict_upcoming(
            _schedule(),
            repo=tmp_path,
        )

        assert "YEAR" not in result.columns
        assert result["GAME_ID"].tolist() == ["game-2", "game-1"]
        assert {
            "AWAY_TEAM_ELO",
            "HOME_TEAM_ELO",
            "AWAY_WIN_PROB",
            "HOME_WIN_PROB",
            "AWAY_TEAM_WIN_PROB",
            "HOME_TEAM_WIN_PROB",
            "model_name",
            "model_type",
        } <= set(result.columns)
        assert (result["model_name"] == "win_prob").all()
        assert (result["model_type"] == "elo").all()
        load_elo_state.assert_called_once_with(tmp_path)

    def test_model_still_satisfies_root_game_model_protocol(self) -> None:
        assert isinstance(WinProbEloModel(), GameModel)


class TestUpcomingEloPredictionWithEvidence:
    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_returns_predictions_and_computations_from_one_execution(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        verify_lineage.return_value = True
        load_elo_state.return_value = _elo_state()
        schedule = _schedule()
        original = schedule.copy(deep=True)

        execution = WinProbEloModel().predict_upcoming_with_evidence(
            schedule,
            source_revision=_revision(),
            source_artifacts=_sources(),
            repo=tmp_path,
        )

        assert "YEAR" in execution.predictions.columns
        assert execution.predictions["GAME_ID"].tolist() == ["game-2", "game-1"]
        assert tuple(value.game_id for value in execution.computations) == (
            "game-1",
            "game-2",
        )
        assert execution.source_revision == _revision()
        assert execution.source_artifacts == _sources()
        assert tuple(value.kind for value in execution.binary_artifacts) == (
            PredictionArtifactKind.ELO_LINEAGE,
            PredictionArtifactKind.ELO_STATE,
        )
        assert {value.formula_id for value in execution.computations} == {ELO_FORMULA_ID}
        assert {value.divisor for value in execution.computations} == {480.0}
        pd.testing.assert_frame_equal(schedule, original)
        verify_lineage.assert_called_once_with(repo=tmp_path)
        load_elo_state.assert_called_once_with(tmp_path)

    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_computations_match_exact_prediction_rows(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        verify_lineage.return_value = True
        load_elo_state.return_value = _elo_state()

        execution = WinProbEloModel().predict_upcoming_with_evidence(
            _schedule(),
            source_revision=_revision(),
            source_artifacts=_sources(),
            repo=tmp_path,
        )
        rows = execution.predictions.set_index("GAME_ID")

        for computation in execution.computations:
            row = rows.loc[computation.game_id]
            assert row["YEAR"] == computation.season
            assert int(row["WEEK_NUM"]) == computation.week
            assert row["AWAY_TEAM"] == computation.away_team
            assert row["HOME_TEAM"] == computation.home_team
            assert row["AWAY_TEAM_ELO"] == pytest.approx(computation.away_elo)
            assert row["HOME_TEAM_ELO"] == pytest.approx(computation.home_elo)
            assert row["AWAY_WIN_PROB"] == pytest.approx(computation.away_win_probability)
            assert row["HOME_WIN_PROB"] == pytest.approx(computation.home_win_probability)

    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_legacy_and_evidence_paths_produce_same_probabilities(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        verify_lineage.return_value = True
        load_elo_state.return_value = _elo_state()
        model = WinProbEloModel()

        legacy = model.predict_upcoming(_schedule(), repo=tmp_path)
        execution = model.predict_upcoming_with_evidence(
            _schedule(),
            source_revision=_revision(),
            source_artifacts=_sources(),
            repo=tmp_path,
        )

        probability_columns = [
            "GAME_ID",
            "AWAY_TEAM_ELO",
            "HOME_TEAM_ELO",
            "AWAY_WIN_PROB",
            "HOME_WIN_PROB",
        ]
        pd.testing.assert_frame_equal(
            legacy.loc[:, probability_columns].reset_index(drop=True),
            execution.predictions.loc[:, probability_columns].reset_index(drop=True),
        )
        assert load_elo_state.call_count == 2

    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_empty_rating_coverage_fails_closed_for_evidence(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        verify_lineage.return_value = True
        load_elo_state.return_value = _elo_state().assign(NFL_WEEK=1)

        with pytest.raises(ValueError, match="must not be empty"):
            WinProbEloModel().predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )


class TestEloLineageFailureBoundary:
    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_unavailable_lineage_blocks_before_state_loading(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        verify_lineage.return_value = False

        with pytest.raises(
            ValueError,
            match="requires current authenticated schema-1 Elo lineage",
        ):
            WinProbEloModel().predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        verify_lineage.assert_called_once_with(repo=tmp_path)
        load_elo_state.assert_not_called()

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (ValueError("Elo lineage contains malformed JSON"), "malformed JSON"),
            (ValueError("Unsupported Elo lineage schema_version: 999"), "schema_version"),
        ],
    )
    @patch("gridiron_edge.datasets.loaders.load_elo_state")
    @patch("gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage")
    def test_malformed_lineage_errors_propagate_before_state_loading(
        self,
        verify_lineage: MagicMock,
        load_elo_state: MagicMock,
        error: ValueError,
        message: str,
        tmp_path: Path,
    ) -> None:
        verify_lineage.side_effect = error

        with pytest.raises(ValueError, match=message):
            WinProbEloModel().predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        load_elo_state.assert_not_called()

    def test_lineage_verification_precedes_state_loading(self, tmp_path: Path) -> None:
        order: list[str] = []

        def verify(*, repo: Path) -> bool:
            assert repo == tmp_path
            order.append("verify-lineage")
            return True

        def load(repo: Path) -> pd.DataFrame:
            assert repo == tmp_path
            order.append("load-elo-state")
            return _elo_state()

        with (
            patch(
                "gridiron_edge.ratings.elo.lineage.verify_current_elo_lineage",
                side_effect=verify,
            ) as verify_lineage,
            patch(
                "gridiron_edge.datasets.loaders.load_elo_state",
                side_effect=load,
            ) as load_elo_state,
        ):
            WinProbEloModel().predict_upcoming_with_evidence(
                _schedule(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                repo=tmp_path,
            )

        assert order == ["verify-lineage", "load-elo-state"]
        assert verify_lineage.mock_calls == [call(repo=tmp_path)]
        assert load_elo_state.mock_calls == [call(tmp_path)]
