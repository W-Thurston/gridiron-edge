# tests/unit/evaluation/test_weekly_recap_starters.py

"""Tests for the weekly recap's starting-QB section (``DECISIONS.md`` D77, D86)."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.weekly_recap import (
    BUILT_AT,
    CAPTURE_DIGEST,
    CARRIED_RUN,
    GAME_A,
    GAME_B,
    GAME_C,
    GAME_D,
    PROJECTED_AS_OF,
    SEASON,
    WEEK,
    RecapWeek,
    capture_sidecar,
    recap_week,
)

from gridiron_edge.evaluation import weekly_recap_build
from gridiron_edge.evaluation.prediction_input_evidence import PredictionSourceState
from gridiron_edge.evaluation.weekly_recap import (
    BuiltWeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
)
from gridiron_edge.evaluation.weekly_recap_starters import (
    PredictorAccuracy,
    StarterSummary,
    runs_without_starter_record,
    week_passers,
)

pytestmark = pytest.mark.unit


def _build(week: RecapWeek | None = None, **overrides: object) -> BuiltWeeklyRecap:
    week = week or recap_week()
    arguments: dict[str, object] = {
        "season": SEASON,
        "week": WEEK,
        "schedule": week.schedule,
        "product": week.product,
        "forecast_events": week.events,
        "games": week.games,
        "player_game_logs": week.logs,
        "sidecar_captures": week.captures,
        "built_at": BUILT_AT,
    }
    return build_weekly_recap(**(arguments | overrides))  # type: ignore[arg-type]


def _plain(value: object) -> object:
    """Return a numpy scalar as its Python value, so ``is True`` checks work."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _side(built: BuiltWeeklyRecap, game_id: str, side: str) -> dict[str, object]:
    row = built.games.set_index("game_id").loc[game_id]
    prefix = f"{side}_"
    return {
        name.removeprefix(prefix): (None if pd.isna(value) else _plain(value))
        for name, value in row.items()
        if str(name).startswith(prefix)
        and str(name).removeprefix(prefix)
        in {
            "starter_status",
            "starter_record_source",
            "assumed_qb_id",
            "assumed_qb_name",
            "assumed_qb_source",
            "assumed_qb_as_of",
            "nflverse_qb_id",
            "nflverse_qb_name",
            "carried_forward_qb_id",
            "carried_forward_qb_name",
            "actual_qb_id",
            "actual_qb_name",
            "actual_qb_source",
            "assumed_correct",
            "nflverse_correct",
            "carry_forward_correct",
        }
    }


class TestSides:
    def test_a_mid_week_change_the_assumption_caught(self) -> None:
        away = _side(_build(), GAME_A, "away")

        assert away == {
            "starter_status": "available",
            "starter_record_source": "product_record",
            "assumed_qb_id": "qa1",
            "assumed_qb_name": "Away Starter",
            "assumed_qb_source": "depth_chart_report",
            "assumed_qb_as_of": PROJECTED_AS_OF,
            "nflverse_qb_id": "qa1",
            "nflverse_qb_name": "Away Starter",
            "carried_forward_qb_id": "qa0",
            "carried_forward_qb_name": "Away Veteran",
            # The starter threw 6 passes before leaving; the backup's 25 do not replace him.
            "actual_qb_id": "qa1",
            "actual_qb_name": "Away Starter",
            "actual_qb_source": "nflverse_schedule",
            "assumed_correct": True,
            "nflverse_correct": True,
            "carry_forward_correct": False,
        }

    def test_a_wrong_assumption(self) -> None:
        home = _side(_build(), GAME_A, "home")

        assert home["assumed_qb_id"] == "qb2"
        assert home["actual_qb_id"] == "qb1"
        assert (home["assumed_correct"], home["nflverse_correct"]) == (False, True)
        assert home["carry_forward_correct"] is True

    def test_a_carried_row_is_scored_from_its_runs_capture(self) -> None:
        built = _build()
        away = _side(built, GAME_B, "away")
        home = _side(built, GAME_B, "home")

        assert away["starter_record_source"] == "sidecar_capture"
        assert (away["assumed_qb_id"], away["assumed_qb_source"]) == ("qc1", "depth_chart_report")
        assert away["assumed_qb_as_of"] == PROJECTED_AS_OF
        assert (away["nflverse_qb_id"], away["nflverse_qb_name"]) == ("qc0", "Carried Veteran")
        assert away["carried_forward_qb_id"] == "qc0"
        # nflverse kept a stale id with no attempt, so the week's top passer started.
        assert (away["actual_qb_id"], away["actual_qb_source"]) == ("qc1", "game_log_top_passer")
        assert (away["assumed_correct"], away["nflverse_correct"]) == (True, False)
        assert away["carry_forward_correct"] is False
        assert home["starter_status"] == "available"
        assert home["assumed_qb_source"] == "nflverse_schedule"
        assert (home["assumed_correct"], home["nflverse_correct"]) == (True, True)
        assert built.recap.starter_capture_digests == (CAPTURE_DIGEST,)

    def test_a_game_not_in_the_product_still_names_who_started(self) -> None:
        away = _side(_build(), GAME_C, "away")

        assert away["starter_status"] == "not_in_product"
        assert away["starter_record_source"] is None
        assert away["assumed_qb_id"] is None
        assert (away["actual_qb_id"], away["actual_qb_name"]) == ("qe1", "Unforecast Away")
        assert away["assumed_correct"] is None

    def test_an_unresolved_side_and_a_missing_actual_are_explicit(self) -> None:
        built = _build()
        away = _side(built, GAME_D, "away")
        home = _side(built, GAME_D, "home")

        assert away["starter_status"] == "actual_missing"
        assert away["assumed_qb_id"] == "qg1"
        assert away["actual_qb_id"] is None
        assert away["assumed_correct"] is None
        assert home["starter_status"] == "starter_unresolved"
        assert home["actual_qb_id"] == "qh1"
        assert home["assumed_correct"] is None

    def test_without_a_capture_the_carried_row_is_not_recorded(self) -> None:
        built = _build(sidecar_captures={})

        for side in ("away", "home"):
            values = _side(built, GAME_B, side)
            assert values["starter_status"] == "not_recorded"
            assert values["assumed_qb_id"] is None
            assert values["actual_qb_id"] is not None
        assert built.recap.starter_capture_digests == ()
        assert built.recap.starters.not_recorded == 2
        assert built.recap.starters.from_capture == 0

    def test_a_product_without_the_record_is_not_recorded_throughout(self) -> None:
        week = recap_week()
        bare = week.product.drop(
            columns=[c for c in week.product.columns if "qb_" in c or c == "starting_qb_status"]
        )

        built = _build(product=bare, sidecar_captures={})

        assert runs_without_starter_record(bare) == tuple(sorted({CARRIED_RUN, "run-product"}))
        assert built.recap.starters.not_recorded == 6
        assert built.recap.starters.scored == 0
        assert built.recap.starters.assumed == PredictorAccuracy(0, 0, None)


def test_the_week_summary_counts_every_slot_and_scores_each_predictor() -> None:
    assert _build().recap.starters == StarterSummary(
        slots=8,
        scored=4,
        from_capture=2,
        not_recorded=0,
        starter_unresolved=1,
        not_in_product=2,
        actual_missing=1,
        actual_from_game_logs=1,
        assumed=PredictorAccuracy(4, 3, 0.75),
        nflverse=PredictorAccuracy(4, 3, 0.75),
        carry_forward=PredictorAccuracy(4, 2, 0.5),
    )


def test_only_the_carried_run_lacks_a_record() -> None:
    assert runs_without_starter_record(recap_week().product) == (CARRIED_RUN,)


def test_week_passers_keep_regular_season_attempts_of_the_week() -> None:
    passers = week_passers(recap_week().logs, season_int=2026, week=WEEK)

    assert "qa7" not in set(passers["player_id"])  # preseason
    assert "qa8" not in set(passers["player_id"])  # no attempt
    assert "qa0" not in set(passers["player_id"])  # another week
    assert len(passers) == 8


def test_a_changed_passer_line_changes_the_recap_identity() -> None:
    week = recap_week()
    logs = week.logs.copy()
    logs.loc[logs["player_id"] == "qb1", "attempts"] = 31.0

    assert _build().recap.recap_id != _build(player_game_logs=logs).recap.recap_id
    assert _build().recap.recap_id == _build().recap.recap_id


class TestLoadSidecarCaptures:
    """Captures are found through the run's evidence and verified by digest and size."""

    @pytest.fixture
    def repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        self.references: dict[str, list[tuple[str, int]]] = {}

        def evidence(run_id: str, *, repo: Path) -> tuple[SimpleNamespace, ...]:
            return tuple(
                SimpleNamespace(
                    source_artifacts=(
                        SimpleNamespace(
                            relative_path="data/cleaned/NFL_qb_coach_sidecar.parquet",
                            content_digest=digest,
                            size_bytes=size,
                            state=PredictionSourceState.PRESENT,
                        ),
                    )
                )
                for digest, size in self.references.get(run_id, [])
            )

        monkeypatch.setattr(weekly_recap_build, "list_prediction_input_evidence_by_run", evidence)
        return tmp_path

    def _capture(self, repo: Path, frame: DataFrame) -> tuple[str, int]:
        directory = repo / "data/output/qb_sidecar_captures"
        directory.mkdir(parents=True, exist_ok=True)
        staging = directory / "staging.parquet"
        frame.to_parquet(staging, index=False)
        content = staging.read_bytes()
        digest = sha256(content).hexdigest()
        staging.rename(directory / f"sidecar_sha256_{digest}.parquet")
        return digest, len(content)

    def test_loads_a_verified_capture(self, repo: Path) -> None:
        digest, size = self._capture(repo, capture_sidecar())
        self.references[CARRIED_RUN] = [(digest, size)]

        captures = weekly_recap_build.load_sidecar_captures(repo, [CARRIED_RUN])

        assert list(captures) == [CARRIED_RUN]
        assert captures[CARRIED_RUN].digest == digest
        assert len(captures[CARRIED_RUN].sidecar) == 2

    def test_skips_a_run_without_a_sidecar_or_without_a_capture(self, repo: Path) -> None:
        self.references["captured-never"] = [("d" * 64, 10)]

        assert (
            weekly_recap_build.load_sidecar_captures(repo, ["no-evidence", "captured-never"]) == {}
        )

    def test_refuses_a_capture_that_does_not_match_its_digest(self, repo: Path) -> None:
        digest, size = self._capture(repo, capture_sidecar())
        path = repo / f"data/output/qb_sidecar_captures/sidecar_sha256_{digest}.parquet"
        path.write_bytes(path.read_bytes()[:-1] + b"\0")
        self.references[CARRIED_RUN] = [(digest, size)]

        with pytest.raises(WeeklyRecapRefusalError, match="does not match"):
            weekly_recap_build.load_sidecar_captures(repo, [CARRIED_RUN])

    def test_refuses_a_capture_of_the_wrong_size(self, repo: Path) -> None:
        digest, size = self._capture(repo, capture_sidecar())
        self.references[CARRIED_RUN] = [(digest, size + 1)]

        with pytest.raises(WeeklyRecapRefusalError, match="does not match"):
            weekly_recap_build.load_sidecar_captures(repo, [CARRIED_RUN])

    def test_refuses_evidence_that_disagrees_about_the_sidecar(self, repo: Path) -> None:
        self.references[CARRIED_RUN] = [("a" * 64, 1), ("b" * 64, 2)]

        with pytest.raises(WeeklyRecapRefusalError, match="disagrees"):
            weekly_recap_build.load_sidecar_captures(repo, [CARRIED_RUN])
