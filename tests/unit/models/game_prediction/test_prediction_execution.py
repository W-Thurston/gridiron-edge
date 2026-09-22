# tests/unit/models/game_prediction/test_prediction_execution.py
"""Tests for evidence-preserving live game prediction execution results."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    ELO_FORMULA_ID,
    EloPredictionComputation,
    build_elo_prediction_execution,
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
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["game-2", "game-1"],
            "YEAR": ["2026-2027", "2026-2027"],
            "WEEK_NUM": [2, 2],
            "AWAY_TEAM": ["Away B", "Away A"],
            "HOME_TEAM": ["Home B", "Home A"],
            "AWAY_TEAM_ELO": [1520.0, 1500.0],
            "HOME_TEAM_ELO": [1480.0, 1510.0],
            "AWAY_WIN_PROB": [0.55, 0.48],
            "HOME_WIN_PROB": [0.45, 0.52],
            "model_name": ["win_prob", "win_prob"],
            "model_type": ["elo", "elo"],
        }
    )


def _execution():
    return build_elo_prediction_execution(
        _predictions(),
        source_revision=_revision(),
        source_artifacts=_sources(),
        divisor=480.0,
    )


class TestEloPredictionExecution:
    def test_preserves_exact_computations_in_deterministic_game_order(self) -> None:
        execution = _execution()

        assert execution.computations == (
            EloPredictionComputation(
                game_id="game-1",
                season="2026-2027",
                week=2,
                away_team="Away A",
                home_team="Home A",
                away_elo=1500.0,
                home_elo=1510.0,
                formula_id=ELO_FORMULA_ID,
                divisor=480.0,
                away_win_probability=0.48,
                home_win_probability=0.52,
            ),
            EloPredictionComputation(
                game_id="game-2",
                season="2026-2027",
                week=2,
                away_team="Away B",
                home_team="Home B",
                away_elo=1520.0,
                home_elo=1480.0,
                formula_id=ELO_FORMULA_ID,
                divisor=480.0,
                away_win_probability=0.55,
                home_win_probability=0.45,
            ),
        )

    def test_returns_defensive_prediction_copy(self) -> None:
        predictions = _predictions()
        execution = build_elo_prediction_execution(
            predictions,
            source_revision=_revision(),
            source_artifacts=_sources(),
            divisor=480.0,
        )

        predictions.loc[0, "AWAY_TEAM_ELO"] = 9999.0

        assert execution.predictions.loc[0, "AWAY_TEAM_ELO"] == pytest.approx(1520.0)

    def test_preserves_complete_source_inventory_and_revision(self) -> None:
        execution = _execution()

        assert execution.source_revision == _revision()
        assert execution.source_artifacts == _sources()

    def test_derives_present_elo_binary_references_from_sources(self) -> None:
        execution = _execution()

        assert tuple(reference.kind for reference in execution.binary_artifacts) == (
            PredictionArtifactKind.ELO_LINEAGE,
            PredictionArtifactKind.ELO_STATE,
        )
        assert all(
            reference.state is PredictionArtifactState.PRESENT
            for reference in execution.binary_artifacts
        )
        lineage, state = execution.binary_artifacts
        assert lineage.source_relative_path == ("data/cleaned/NFL_Team_Elo.metadata.json")
        assert lineage.content_digest == DIGEST_B
        assert lineage.size_bytes == 200
        assert state.source_relative_path == "data/cleaned/NFL_Team_Elo.csv"
        assert state.content_digest == DIGEST_A
        assert state.size_bytes == 100

    def test_custom_formula_identity_is_preserved(self) -> None:
        execution = build_elo_prediction_execution(
            _predictions(),
            source_revision=_revision(),
            source_artifacts=_sources(),
            divisor=480.0,
            formula_id="custom_formula_v2",
        )

        assert {computation.formula_id for computation in execution.computations} == {
            "custom_formula_v2"
        }


class TestExecutionScopeValidation:
    @pytest.mark.parametrize(
        "column",
        [
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "AWAY_TEAM",
            "HOME_TEAM",
            "AWAY_TEAM_ELO",
            "HOME_TEAM_ELO",
            "AWAY_WIN_PROB",
            "HOME_WIN_PROB",
        ],
    )
    def test_missing_execution_column_is_rejected(self, column: str) -> None:
        predictions = _predictions().drop(columns=[column])

        with pytest.raises(
            ValueError,
            match="missing execution-evidence columns",
        ):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_empty_execution_is_rejected(self) -> None:
        predictions = _predictions().iloc[0:0].copy()

        with pytest.raises(ValueError, match="must not be empty"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_duplicate_game_identity_is_rejected(self) -> None:
        predictions = _predictions()
        predictions.loc[1, "GAME_ID"] = "game-2"

        with pytest.raises(ValueError, match="duplicate game IDs"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_same_away_and_home_team_is_rejected(self) -> None:
        predictions = _predictions()
        predictions.loc[0, "HOME_TEAM"] = "Away B"

        with pytest.raises(ValueError, match="Away and Home teams must differ"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    @pytest.mark.parametrize("week", [0, -1, 1.5, float("nan"), True])
    def test_invalid_week_is_rejected(self, week: object) -> None:
        predictions = _predictions()
        predictions["WEEK_NUM"] = predictions["WEEK_NUM"].astype(object)
        predictions.loc[0, "WEEK_NUM"] = week

        with pytest.raises(ValueError, match="positive integer"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )


class TestEloNumericValidation:
    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("AWAY_TEAM_ELO", float("nan")),
            ("HOME_TEAM_ELO", float("inf")),
            ("AWAY_TEAM_ELO", "not-numeric"),
        ],
    )
    def test_nonfinite_or_malformed_rating_is_rejected(
        self,
        column: str,
        value: object,
    ) -> None:
        predictions = _predictions()
        predictions[column] = predictions[column].astype(object)
        predictions.loc[0, column] = value

        with pytest.raises(ValueError, match="finite numeric data"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    @pytest.mark.parametrize("divisor", [0.0, -480.0, float("nan"), float("inf")])
    def test_invalid_divisor_is_rejected(self, divisor: float) -> None:
        expected = "must be positive" if divisor <= 0 else "finite numeric data"

        with pytest.raises(ValueError, match=expected):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=divisor,
            )

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("AWAY_WIN_PROB", -0.01),
            ("AWAY_WIN_PROB", 1.01),
            ("HOME_WIN_PROB", float("nan")),
        ],
    )
    def test_invalid_probability_is_rejected(
        self,
        column: str,
        value: float,
    ) -> None:
        predictions = _predictions()
        predictions.loc[0, column] = value

        with pytest.raises(
            ValueError,
            match=r"between 0 and 1|finite numeric data",
        ):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_noncomplementary_probabilities_are_rejected(self) -> None:
        predictions = _predictions()
        predictions.loc[0, "HOME_WIN_PROB"] = 0.40

        with pytest.raises(ValueError, match="must be complementary"):
            build_elo_prediction_execution(
                predictions,
                source_revision=_revision(),
                source_artifacts=_sources(),
                divisor=480.0,
            )


class TestExecutionProvenanceValidation:
    def test_dirty_revision_is_rejected(self) -> None:
        revision = replace(_revision(), tracked_worktree_clean=False)

        with pytest.raises(ValueError, match="clean tracked source revision"):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=revision,
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_invalid_commit_is_rejected(self) -> None:
        revision = replace(_revision(), commit="invalid")

        with pytest.raises(ValueError, match="lowercase Git commit SHA"):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=revision,
                source_artifacts=_sources(),
                divisor=480.0,
            )

    def test_unsorted_source_inventory_is_rejected(self) -> None:
        sources = tuple(reversed(_sources()))

        with pytest.raises(ValueError, match="sorted and unique"):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=_revision(),
                source_artifacts=sources,
                divisor=480.0,
            )

    @pytest.mark.parametrize(
        "missing_path",
        [
            "data/cleaned/NFL_Team_Elo.csv",
            "data/cleaned/NFL_Team_Elo.metadata.json",
        ],
    )
    def test_missing_required_elo_source_is_rejected(self, missing_path: str) -> None:
        sources = tuple(
            reference for reference in _sources() if reference.relative_path != missing_path
        )

        with pytest.raises(ValueError, match="source inventory is missing"):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=_revision(),
                source_artifacts=sources,
                divisor=480.0,
            )

    @pytest.mark.parametrize(
        "absent_path",
        [
            "data/cleaned/NFL_Team_Elo.csv",
            "data/cleaned/NFL_Team_Elo.metadata.json",
        ],
    )
    def test_absent_required_elo_source_is_rejected(self, absent_path: str) -> None:
        sources = tuple(
            replace(
                reference,
                state=PredictionSourceState.ABSENT,
                content_digest=None,
                size_bytes=None,
            )
            if reference.relative_path == absent_path
            else reference
            for reference in _sources()
        )

        with pytest.raises(ValueError, match="requires present source artifact"):
            build_elo_prediction_execution(
                _predictions(),
                source_revision=_revision(),
                source_artifacts=sources,
                divisor=480.0,
            )
