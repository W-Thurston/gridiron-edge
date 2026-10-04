# tests/unit/cli/test_weekly_product_composition.py

"""Tests for the shared, schedule-agnostic weekly-product composition helper.

``test_weekly_predict_product_stage.py`` already proves the live pipeline's
exact call chain and output through this helper. These tests cover what is
new here: composition driven by an arbitrary caller-supplied schedule (not
just the fetch-derived upcoming one), and the empty-scope failure this
introduces for a caller (development regeneration) that can request a scope
the schedule does not cover.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.cli._weekly_product_composition import (
    _carry_rows,
    carried_game_ids,
    compose_and_select_weekly_product,
    require_pregame_live_forecasts,
)
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    resolve_prediction_policy,
)
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    unrecorded_starting_qb_record,
)

SEASON = "2026-2027"
WEEK = 2
RUN_ID = "run-1"
GENERATED_AT = datetime(2026, 9, 20, 12, tzinfo=UTC)


_NO_RUN_COLUMNS = {
    "win_run_id": None,
    "total_run_id": None,
    "win_role": None,
    "total_role": None,
    "win_generated_at": None,
    "total_generated_at": None,
}


def _policy():
    availability = PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=True,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=True,
    )
    return resolve_prediction_policy(
        availability,
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


def test_empty_scoped_schedule_raises_before_any_composition() -> None:
    schedule = DataFrame({"season": [SEASON], "week": [WEEK + 1], "game_id": ["other-week"]})

    with pytest.raises(ValueError, match="Rich schedule has no rows"):
        compose_and_select_weekly_product(
            schedule=schedule,
            events=DataFrame({"event_id": ["event-1"]}),
            policy=_policy(),
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
        )


def test_retained_history_schedule_composes_and_selects(tmp_path: Path) -> None:
    schedule = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": ["2026_02_DET_BUF"],
            "away_team": ["Detroit Lions"],
            "home_team": ["Buffalo Bills"],
            "game_date": ["2026-09-27"],
            "game_time": ["13:00:00"],
        }
    )
    events = DataFrame({"event_id": ["event-1"]})
    win_product = schedule.assign(win_status="available")
    spread_product = win_product.assign(spread_status="available")
    total_product = spread_product.assign(total_status="available")
    final_product = total_product.assign(projected_score_status="available", **_NO_RUN_COLUMNS)
    artifact = tmp_path / "weekly.parquet"
    resolutions = (MagicMock(),)

    with (
        patch(
            "gridiron_edge.cli._weekly_product_composition.resolve_forecast_candidates",
            return_value=resolutions,
        ) as resolve_candidates,
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_win_product",
            return_value=win_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_derived_spreads",
            return_value=spread_product,
        ) as attach_spread,
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_selected_totals",
            return_value=total_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_game_product",
            return_value=final_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.write_weekly_product",
            return_value=artifact,
        ) as write_product,
        patch(
            "gridiron_edge.cli._weekly_product_composition.select_current_weekly_product"
        ) as select_product,
    ):
        composed = compose_and_select_weekly_product(
            schedule=schedule,
            events=events,
            policy=_policy(),
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    assert composed.artifact == artifact
    assert composed.row_count == 1
    assert composed.product_id == f"weekly_{SEASON.replace('-', '_')}_wk{WEEK:02d}_{RUN_ID}"
    assert resolve_candidates.call_count == 2
    attach_spread.assert_called_once_with(win_product, repo=tmp_path)
    write_product.assert_called_once()
    select_product.assert_called_once_with(
        tmp_path,
        composed.product_id,
        season=SEASON,
        week=WEEK,
        selected_at=select_product.call_args.kwargs["selected_at"],
    )


def test_unavailable_family_skips_candidate_resolution(tmp_path: Path) -> None:
    """A policy family with no model_type must not attempt resolution."""
    availability = PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=False,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=False,
        total_xgboost_features_available=False,
    )
    policy = resolve_prediction_policy(availability, win_champion=None, total_champion=None)
    assert policy.win.model_type is None
    assert policy.total.model_type is None

    schedule = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": ["2026_02_DET_BUF"],
            "game_date": ["2026-09-27"],
            "game_time": ["13:00:00"],
        }
    )
    win_product = schedule.assign(win_status="policy_unavailable")
    spread_product = win_product.assign(spread_status="unavailable")
    total_product = spread_product.assign(total_status="policy_unavailable")
    final_product = total_product.assign(
        projected_score_status="total_unavailable", **_NO_RUN_COLUMNS
    )

    with (
        patch(
            "gridiron_edge.cli._weekly_product_composition.resolve_forecast_candidates",
        ) as resolve_candidates,
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_win_product",
            return_value=win_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_derived_spreads",
            return_value=spread_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.load_and_attach_selected_totals",
            return_value=total_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.build_weekly_game_product",
            return_value=final_product,
        ),
        patch(
            "gridiron_edge.cli._weekly_product_composition.write_weekly_product",
            return_value=tmp_path / "weekly.parquet",
        ),
        patch("gridiron_edge.cli._weekly_product_composition.select_current_weekly_product"),
    ):
        compose_and_select_weekly_product(
            schedule=schedule,
            events=DataFrame({"event_id": ["event-1"]}),
            policy=policy,
            run_id=RUN_ID,
            generated_at=GENERATED_AT,
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
        )

    resolve_candidates.assert_not_called()


def _evidence_with_sidecar(digest: str | None):
    from types import SimpleNamespace

    from gridiron_edge.evaluation.prediction_input_evidence import PredictionSourceState

    reference = SimpleNamespace(
        relative_path="data/cleaned/NFL_qb_coach_sidecar.parquet",
        state=PredictionSourceState.PRESENT if digest else PredictionSourceState.ABSENT,
        content_digest=digest,
    )
    return (SimpleNamespace(source_artifacts=(reference,)),)


def _write_sidecar(repo: Path) -> str:
    from hashlib import sha256

    import pandas as pd

    path = repo / "data" / "cleaned" / "NFL_qb_coach_sidecar.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"GAME_ID": ["g"]}).to_parquet(path)
    return sha256(path.read_bytes()).hexdigest()


def test_bound_sidecar_loads_only_when_its_digest_matches_the_evidence(tmp_path: Path) -> None:
    from gridiron_edge.cli._weekly_product_composition import _bound_sidecar

    digest = _write_sidecar(tmp_path)
    target = "gridiron_edge.cli._weekly_product_composition.list_prediction_input_evidence_by_run"

    with patch(target, return_value=_evidence_with_sidecar(digest)):
        assert _bound_sidecar("run", repo=tmp_path) is not None
    with (
        patch(target, return_value=_evidence_with_sidecar("0" * 64)),
        pytest.raises(ValueError, match="sidecar changed after the forecast run"),
    ):
        _bound_sidecar("run", repo=tmp_path)


def test_bound_sidecar_is_absent_without_evidence_for_it(tmp_path: Path) -> None:
    from gridiron_edge.cli._weekly_product_composition import _bound_sidecar

    _write_sidecar(tmp_path)
    target = "gridiron_edge.cli._weekly_product_composition.list_prediction_input_evidence_by_run"

    with patch(target, return_value=()):
        assert _bound_sidecar("run", repo=tmp_path) is None
    with patch(target, return_value=_evidence_with_sidecar(None)):
        assert _bound_sidecar("run", repo=tmp_path) is None


def test_bound_sidecar_rejects_evidence_that_disagrees(tmp_path: Path) -> None:
    from gridiron_edge.cli._weekly_product_composition import _bound_sidecar

    _write_sidecar(tmp_path)
    target = "gridiron_edge.cli._weekly_product_composition.list_prediction_input_evidence_by_run"
    evidence = _evidence_with_sidecar("a" * 64) + _evidence_with_sidecar("b" * 64)

    with patch(target, return_value=evidence), pytest.raises(ValueError, match="disagrees"):
        _bound_sidecar("run", repo=tmp_path)


# ---------------------------------------------------------------------------
# Mid-week refresh: a game moving from upcoming to played keeps its forecast
# ---------------------------------------------------------------------------

THURSDAY = "2026-09-24"  # kickoff 20:15 Eastern = 2026-09-25T00:15Z
SUNDAY = "2026-09-27"  # kickoff 13:00 Eastern = 17:00Z
FRIDAY_NOON = datetime(2026, 9, 25, 12, tzinfo=UTC)
WEDNESDAY = datetime(2026, 9, 23, 12, tzinfo=UTC)


def _forecast_row(
    game_id: str,
    date: str,
    time: str,
    *,
    run: str,
    generated: datetime,
    role: str = "live",
    with_record: bool = True,
) -> DataFrame:
    row = DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": [game_id],
            "game_date": [date],
            "game_time": [time],
            "win_status": ["available"],
            "total_status": ["available"],
            "win_run_id": [run],
            "total_run_id": [run],
            "win_event_id": [f"win-{game_id}-{run}"],
            "total_event_id": [f"total-{game_id}-{run}"],
            "win_role": [role],
            "total_role": [role],
            "win_generated_at": [pd.Timestamp(generated)],
            "total_generated_at": [pd.Timestamp(generated)],
            "home_win_prob": [0.6 if run == "old-run" else 0.7],
        }
    )
    if with_record:
        row = pd.concat([row, unrecorded_starting_qb_record(row.index)], axis=1)
    return row


def _previous() -> DataFrame:
    """The week's selected product from Wednesday: both games forecast by ``old-run``."""
    frame = pd.concat(
        [
            _forecast_row("thu", THURSDAY, "20:15:00", run="old-run", generated=WEDNESDAY),
            _forecast_row("sun", SUNDAY, "13:00:00", run="old-run", generated=WEDNESDAY),
        ],
        ignore_index=True,
    )
    stamped = frame.assign(
        product_schema_version=2,
        product_id="old-product",
        product_run_id="old-run",
        product_generated_at=pd.Timestamp(WEDNESDAY),
    )
    return stamped


def _schedule_of(*games: tuple[str, str, str]) -> DataFrame:
    return DataFrame(
        {
            "season": SEASON,
            "week": WEEK,
            "game_id": [game[0] for game in games],
            "game_date": [game[1] for game in games],
            "game_time": [game[2] for game in games],
        }
    )


def test_a_game_that_started_or_finished_is_carried_and_an_upcoming_one_is_not() -> None:
    both = _schedule_of(("thu", THURSDAY, "20:15:00"), ("sun", SUNDAY, "13:00:00"))
    sunday_only = _schedule_of(("sun", SUNDAY, "13:00:00"))

    # Friday: Thursday's game has kicked off (still on the upcoming schedule).
    assert carried_game_ids(_previous(), both, generated_at=FRIDAY_NOON) == ("thu",)
    # Friday after its result landed: it has left the upcoming schedule.
    assert carried_game_ids(_previous(), sunday_only, generated_at=FRIDAY_NOON) == ("thu",)
    # Wednesday, before any kickoff: everything is forecast again.
    assert carried_game_ids(_previous(), both, generated_at=WEDNESDAY) == ()
    assert carried_game_ids(None, both, generated_at=FRIDAY_NOON) == ()


def test_pregame_guard_rejects_a_live_forecast_made_after_kickoff() -> None:
    on_time = _forecast_row("sun", SUNDAY, "13:00:00", run="r", generated=FRIDAY_NOON)
    late = _forecast_row("thu", THURSDAY, "20:15:00", run="r", generated=FRIDAY_NOON)
    development = _forecast_row(
        "thu", THURSDAY, "20:15:00", run="r", generated=FRIDAY_NOON, role="development"
    )

    require_pregame_live_forecasts(on_time)
    require_pregame_live_forecasts(development)
    with pytest.raises(ValueError, match=r"thu \(total\), thu \(win\)|thu \(win\)"):
        require_pregame_live_forecasts(late)


def test_carried_rows_drop_storage_columns_and_gain_an_unrecorded_starter_record() -> None:
    previous = _previous().drop(
        columns=list(unrecorded_starting_qb_record(pd.RangeIndex(1)).columns)
    )
    columns = _forecast_row("x", SUNDAY, "13:00:00", run="r", generated=WEDNESDAY).columns

    kept = _carry_rows(previous, ("thu",), columns)

    assert list(kept.columns) == list(columns)
    assert kept["game_id"].tolist() == ["thu"]
    assert kept["win_run_id"].tolist() == ["old-run"]
    assert kept["starting_qb_status"].tolist() == ["sidecar_unavailable"]


def _compose_mid_week(
    tmp_path: Path,
    *,
    schedule: DataFrame,
    previous: DataFrame,
    stored_event_ids: list[str],
):
    new_rows = _forecast_row("sun", SUNDAY, "13:00:00", run=RUN_ID, generated=FRIDAY_NOON)
    final_product = new_rows.drop(columns=["starting_qb_status"]).pipe(
        lambda frame: pd.concat([frame, unrecorded_starting_qb_record(frame.index)], axis=1)
    )
    prefix = "gridiron_edge.cli._weekly_product_composition"
    with (
        patch(f"{prefix}.resolve_forecast_candidates", return_value=()) as resolve,
        patch(f"{prefix}.build_weekly_win_product", return_value=new_rows),
        patch(f"{prefix}.load_and_attach_derived_spreads", return_value=new_rows),
        patch(f"{prefix}.load_and_attach_selected_totals", return_value=new_rows),
        patch(f"{prefix}.build_weekly_game_product", return_value=final_product),
        patch(f"{prefix}.write_weekly_product", return_value=tmp_path / "w.parquet") as write,
        patch(f"{prefix}.select_current_weekly_product"),
        patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=DataFrame({"event_id": stored_event_ids}),
        ),
    ):
        composed = compose_and_select_weekly_product(
            schedule=schedule,
            events=DataFrame({"event_id": ["new"]}),
            policy=_policy(),
            run_id=RUN_ID,
            generated_at=FRIDAY_NOON,
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            previous_product=previous,
        )
    return composed, write, resolve


def test_refresh_keeps_the_finished_game_row_and_forecasts_only_the_rest(tmp_path: Path) -> None:
    # Friday: Thursday's game is played and no longer on the upcoming schedule.
    composed, write, resolve = _compose_mid_week(
        tmp_path,
        schedule=_schedule_of(("sun", SUNDAY, "13:00:00")),
        previous=_previous(),
        stored_event_ids=["win-thu-old-run", "total-thu-old-run"],
    )

    written = write.call_args.args[1]
    identity = write.call_args.kwargs["identity"]
    assert written["game_id"].tolist() == ["thu", "sun"]
    thursday = written.loc[written["game_id"] == "thu"].iloc[0]
    assert (thursday["win_run_id"], thursday["win_event_id"]) == ("old-run", "win-thu-old-run")
    assert pd.Timestamp(thursday["win_generated_at"]) == pd.Timestamp(WEDNESDAY)
    assert thursday["home_win_prob"] == 0.6  # the pre-kickoff probability, untouched
    sunday = written.loc[written["game_id"] == "sun"].iloc[0]
    assert sunday["win_run_id"] == RUN_ID
    assert identity.carried_run_ids == ("old-run",)
    assert composed.carried_game_ids == ("thu",)
    assert composed.carried_run_ids == ("old-run",)
    assert [i.game_id for i in resolve.call_args_list[0].args[1]] == ["sun"]


def test_refresh_after_kickoff_keeps_the_started_game_still_on_the_schedule(
    tmp_path: Path,
) -> None:
    both = _schedule_of(("thu", THURSDAY, "20:15:00"), ("sun", SUNDAY, "13:00:00"))

    composed, _, resolve = _compose_mid_week(
        tmp_path,
        schedule=both,
        previous=_previous(),
        stored_event_ids=["win-thu-old-run", "total-thu-old-run"],
    )

    assert composed.carried_game_ids == ("thu",)
    assert [i.game_id for i in resolve.call_args_list[0].args[1]] == ["sun"]


def test_refresh_fails_when_a_carried_forecast_event_is_gone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the forecast store"):
        _compose_mid_week(
            tmp_path,
            schedule=_schedule_of(("sun", SUNDAY, "13:00:00")),
            previous=_previous(),
            stored_event_ids=["win-thu-old-run"],
        )


def test_refresh_with_nothing_left_to_forecast_is_refused(tmp_path: Path) -> None:
    previous = _previous().loc[lambda frame: frame["game_id"] == "thu"]

    with pytest.raises(ValueError, match="nothing to forecast"):
        _compose_mid_week(
            tmp_path,
            schedule=_schedule_of(("thu", THURSDAY, "20:15:00")),
            previous=previous,
            stored_event_ids=[],
        )
