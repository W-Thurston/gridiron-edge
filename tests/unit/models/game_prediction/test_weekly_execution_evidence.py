# tests/unit/models/game_prediction/test_weekly_execution_evidence.py
"""Tests for weekly Elo forecast UUID binding and input evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from tests.unit.models.game_prediction.test_weekly_execution import (
    _total_execution,
)

from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionExecutionKind,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    authenticate_prediction_input_evidence,
)
from gridiron_edge.models.elo.model import WinProbEloModel
from gridiron_edge.models.game_prediction.prediction_execution import (
    ELO_FORMULA_ID,
    EloPredictionComputation,
    EloPredictionExecution,
)
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    resolve_prediction_policy,
)
from gridiron_edge.models.game_prediction.weekly_execution import (
    execute_weekly_prediction_policy,
)

SEASON = "2026-2027"
WEEK = 1
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)
COMMIT = "a" * 40
ELO_DIGEST = "b" * 64
LINEAGE_DIGEST = "c" * 64


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [WEEK, WEEK],
            "game_id": ["g1", "g2"],
            "game_day_of_week": ["Sun", "Sun"],
            "game_date": ["2026-09-13", "2026-09-13"],
            "game_time": ["13:00", "16:25"],
            "away_team": ["Away A", "Away B"],
            "home_team": ["Home A", "Home B"],
            "neutral_site": [0, 0],
        }
    )


def _availability() -> PredictionAvailability:
    return PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=True,
        total_xgboost_features_available=False,
    )


def _policy():
    return resolve_prediction_policy(
        _availability(),
        win_champion=ModelProvenance(
            model_name="win_prob",
            model_type="logistic",
            source=PredictionModelSource.CHAMPION,
        ),
        total_champion=ModelProvenance(
            model_name="total",
            model_type="random_forest",
            source=PredictionModelSource.CHAMPION,
        ),
    )


def _revision() -> SourceRevision:
    return SourceRevision(commit=COMMIT, tracked_worktree_clean=True)


def _sources() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionSourceState.PRESENT,
            content_digest=ELO_DIGEST,
            size_bytes=100,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.metadata.json",
            state=PredictionSourceState.PRESENT,
            content_digest=LINEAGE_DIGEST,
            size_bytes=200,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest="d" * 64,
            size_bytes=300,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_wk_by_wk_cleaned.csv",
            state=PredictionSourceState.PRESENT,
            content_digest="e" * 64,
            size_bytes=400,
        ),
    )


def _elo_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "YEAR": [SEASON, SEASON],
            "WEEK_NUM": [WEEK, WEEK],
            "AWAY_TEAM": ["Away A", "Away B"],
            "HOME_TEAM": ["Home A", "Home B"],
            "AWAY_TEAM_ELO": [1500.0, 1490.0],
            "HOME_TEAM_ELO": [1510.0, 1520.0],
            "AWAY_WIN_PROB": [0.45, 0.40],
            "HOME_WIN_PROB": [0.55, 0.60],
            "AWAY_TEAM_WIN_PROB": ["45.0 %", "40.0 %"],
            "HOME_TEAM_WIN_PROB": ["55.0 %", "60.0 %"],
            "model_name": ["win_prob", "win_prob"],
            "model_type": ["elo", "elo"],
        }
    )


def _computations() -> tuple[EloPredictionComputation, ...]:
    return (
        EloPredictionComputation(
            game_id="g1",
            season=SEASON,
            week=WEEK,
            away_team="Away A",
            home_team="Home A",
            away_elo=1500.0,
            home_elo=1510.0,
            formula_id=ELO_FORMULA_ID,
            divisor=480.0,
            away_win_probability=0.45,
            home_win_probability=0.55,
        ),
        EloPredictionComputation(
            game_id="g2",
            season=SEASON,
            week=WEEK,
            away_team="Away B",
            home_team="Home B",
            away_elo=1490.0,
            home_elo=1520.0,
            formula_id=ELO_FORMULA_ID,
            divisor=480.0,
            away_win_probability=0.40,
            home_win_probability=0.60,
        ),
    )


def _elo_execution(
    computations: tuple[EloPredictionComputation, ...] | None = None,
) -> EloPredictionExecution:
    return EloPredictionExecution(
        predictions=_elo_predictions(),
        computations=computations or _computations(),
        source_revision=_revision(),
        source_artifacts=_sources(),
        binary_artifacts=(
            BinaryArtifactReference(
                kind=PredictionArtifactKind.ELO_LINEAGE,
                source_relative_path="data/cleaned/NFL_Team_Elo.metadata.json",
                state=PredictionArtifactState.PRESENT,
                content_digest=LINEAGE_DIGEST,
                size_bytes=200,
            ),
            BinaryArtifactReference(
                kind=PredictionArtifactKind.ELO_STATE,
                source_relative_path="data/cleaned/NFL_Team_Elo.csv",
                state=PredictionArtifactState.PRESENT,
                content_digest=ELO_DIGEST,
                size_bytes=100,
            ),
        ),
    )


def _total_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "AWAY_TEAM": ["Away A", "Away B"],
            "HOME_TEAM": ["Home A", "Home B"],
            "WEEK_NUM": [WEEK, WEEK],
            "model_total": [44.5, 47.0],
        }
    )


def _execute(
    tmp_path: Path,
    *,
    elo_execution: EloPredictionExecution | None = None,
    source_revision: SourceRevision | None = None,
    source_artifacts: tuple[SourceArtifactReference, ...] | None = None,
):
    elo_model = WinProbEloModel()
    elo_model.predict_upcoming_with_evidence = MagicMock(  # type: ignore[method-assign]
        return_value=elo_execution or _elo_execution()
    )
    elo_model.predict_upcoming = MagicMock()  # type: ignore[method-assign]
    total_model = MagicMock()
    total_model.predict_upcoming_with_evidence.return_value = _total_execution()
    total_model.predict_upcoming = MagicMock()

    def registry_get(key: str):
        return {
            "win_prob_elo": lambda: elo_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
    ):
        result = execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="run-1",
            generated_at=GENERATED_AT,
            source_revision=source_revision,
            source_artifacts=source_artifacts,
        )
    return result, elo_model, total_model


class TestEloWeeklyEvidence:
    def test_selected_elo_requires_source_revision(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires source_revision"):
            _execute(
                tmp_path,
                source_revision=None,
                source_artifacts=_sources(),
            )

    def test_selected_elo_requires_source_artifacts(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires source_artifacts"):
            _execute(
                tmp_path,
                source_revision=_revision(),
                source_artifacts=None,
            )

    def test_uses_evidence_method_and_passes_exact_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        result, elo_model, total_model = _execute(
            tmp_path,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )

        elo_model.predict_upcoming.assert_not_called()
        evidence_call = elo_model.predict_upcoming_with_evidence.call_args
        assert evidence_call.kwargs["source_revision"] == _revision()
        assert evidence_call.kwargs["source_artifacts"] == _sources()
        assert evidence_call.kwargs["repo"] == tmp_path
        assert len(result.events) == 4
        assert len(result.input_evidence) == 2
        total_model.predict_upcoming.assert_not_called()
        total_model.predict_upcoming_with_evidence.assert_called_once()

        total_call = total_model.predict_upcoming_with_evidence.call_args
        assert total_call.kwargs["source_revision"] == _revision()
        assert total_call.kwargs["source_artifacts"] == _sources()
        assert total_call.kwargs["repo"] == tmp_path

    def test_generated_event_ids_exactly_match_evidence(self, tmp_path: Path) -> None:
        result, _, _ = _execute(
            tmp_path,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )
        evidence = result.input_evidence[0]
        elo_events = result.events.loc[
            (result.events["model_name"] == "win_prob") & (result.events["model_type"] == "elo"),
            :,
        ]

        assert evidence.execution_kind is PredictionExecutionKind.ELO_FORMULA
        assert tuple(sorted(event.event_id for event in evidence.elo_events)) == tuple(
            sorted(elo_events["event_id"].astype(str).tolist())
        )
        assert tuple(event.game_id for event in evidence.elo_events) == ("g1", "g2")
        assert len({event.event_id for event in evidence.elo_events}) == 2
        authenticate_prediction_input_evidence(
            evidence,
            forecast_events=elo_events,
        )

    def test_final_outputs_match_generated_forecast_events(self, tmp_path: Path) -> None:
        result, _, _ = _execute(
            tmp_path,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )
        evidence = result.input_evidence[0]
        rows = result.events.loc[
            result.events["model_type"] == "elo",
            :,
        ].set_index("game_id")

        for event in evidence.elo_events:
            row = rows.loc[event.game_id]
            outputs = dict(event.final_outputs)
            assert outputs == {
                "away_elo": row["away_elo"],
                "away_win_prob": row["away_win_prob"],
                "home_elo": row["home_elo"],
                "home_win_prob": row["home_win_prob"],
            }

    def test_mixed_elo_win_and_statistical_total_returns_two_evidence(
        self,
        tmp_path: Path,
    ) -> None:
        result, _, _ = _execute(
            tmp_path,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )

        assert len(result.input_evidence) == 2
        assert [
            (evidence.model_name, evidence.model_type) for evidence in result.input_evidence
        ] == [
            ("win_prob", "elo"),
            ("total", "random_forest"),
        ]

        for evidence in result.input_evidence:
            family_events = result.events.loc[
                (result.events["model_name"] == evidence.model_name)
                & (result.events["model_type"] == evidence.model_type),
                :,
            ]

            authenticate_prediction_input_evidence(
                evidence,
                forecast_events=family_events,
            )

        elo_event_ids = {event.event_id for event in result.input_evidence[0].elo_events}
        total_event_ids = {event.event_id for event in result.input_evidence[1].statistical_events}

        assert elo_event_ids
        assert total_event_ids
        assert elo_event_ids.isdisjoint(total_event_ids)

    def test_mismatched_computation_games_fail_explicitly(self, tmp_path: Path) -> None:
        mismatched = (
            _computations()[0],
            replace(_computations()[1], game_id="unrelated-game"),
        )

        with pytest.raises(
            ValueError,
            match="computations do not match generated forecast-event games",
        ):
            _execute(
                tmp_path,
                elo_execution=_elo_execution(mismatched),
                source_revision=_revision(),
                source_artifacts=_sources(),
            )

    def test_weekly_layer_performs_no_persistence(self, tmp_path: Path) -> None:
        with (
            patch(
                "gridiron_edge.evaluation.prediction_input_evidence_store.write_binary_snapshot"
            ) as write_snapshot,
            patch(
                "gridiron_edge.evaluation.prediction_input_evidence_store."
                "write_prediction_input_evidence"
            ) as write_evidence,
        ):
            _execute(
                tmp_path,
                source_revision=_revision(),
                source_artifacts=_sources(),
            )

        write_snapshot.assert_not_called()
        write_evidence.assert_not_called()
