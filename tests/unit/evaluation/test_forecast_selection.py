# tests/unit/evaluation/test_forecast_selection.py

"""Tests for explicit immutable forecast-event selection."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    SelectedForecast,
)
from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateIdentity,
    ForecastCandidateStatus,
    resolve_forecast_candidates,
    select_forecast_events,
    select_forecast_run,
)
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
)


def _event(
    *,
    event_id: str,
    run_id: str,
    role: ForecastRole,
    generated_at: datetime,
    game_id: str = "2026_01_KC_LAC",
    model_name: str = "win_prob",
    model_type: str = "elo",
) -> pd.DataFrame:
    """Create one canonical forecast event."""
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "run_id": [run_id],
            "role": [role.value],
            "generated_at": [generated_at],
            "season": ["2026-2027"],
            "week": [1],
            "game_id": [game_id],
            "model_name": [model_name],
            "model_type": [model_type],
            "game_date": ["2026-09-05"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "away_elo": [1520.0],
            "home_elo": [1480.0],
            "away_win_prob": [0.55],
            "home_win_prob": [0.45],
            "model_spread": [-1.5],
            "model_total": [None],
            "projected_home_score": [None],
            "projected_away_score": [None],
            "margin_std": [13.0],
            "win_prob_lo": [0.47],
            "win_prob_hi": [0.63],
            "confidence_tier": ["Low"],
        }
    )


def _selection(
    *,
    event_id: str,
    game_id: str = "2026_01_KC_LAC",
    model_name: str = "win_prob",
    model_type: str = "elo",
) -> SelectedForecast:
    """Create an explicit selected-forecast reference."""
    return SelectedForecast(
        event_id=event_id,
        game_id=game_id,
        model_name=model_name,
        model_type=model_type,
    )


def _candidate(
    *,
    game_id: str = "2026_01_KC_LAC",
    model_name: str = "win_prob",
    model_type: str = "elo",
) -> ForecastCandidateIdentity:
    """Create a forecast candidate identity."""
    return ForecastCandidateIdentity(
        game_id=game_id,
        model_name=model_name,
        model_type=model_type,
    )


def _events() -> pd.DataFrame:
    """Create older and newer events for one game and model."""
    return pd.concat(
        [
            _event(
                event_id="older-live",
                run_id="run-1",
                role=ForecastRole.LIVE,
                generated_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
            ),
            _event(
                event_id="newer-live",
                run_id="run-2",
                role=ForecastRole.LIVE,
                generated_at=datetime(
                    2026,
                    9,
                    2,
                    12,
                    tzinfo=UTC,
                ),
            ),
            _event(
                event_id="backfilled",
                run_id="run-3",
                role=ForecastRole.BACKFILLED,
                generated_at=datetime(
                    2026,
                    9,
                    3,
                    12,
                    tzinfo=UTC,
                ),
            ),
        ],
        ignore_index=True,
    )


def _multi_family_run_events() -> pd.DataFrame:
    """Create two explicit runs with independent forecast families."""
    return pd.concat(
        [
            _event(
                event_id="run-1-win-kc",
                run_id="run-1",
                role=ForecastRole.LIVE,
                generated_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
                game_id="2026_01_KC_LAC",
                model_name="win_prob",
                model_type="elo",
            ),
            _event(
                event_id="run-1-total-kc",
                run_id="run-1",
                role=ForecastRole.LIVE,
                generated_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
                game_id="2026_01_KC_LAC",
                model_name="total",
                model_type="random_forest",
            ),
            _event(
                event_id="run-1-win-bal",
                run_id="run-1",
                role=ForecastRole.LIVE,
                generated_at=datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=UTC,
                ),
                game_id="2026_01_BAL_BUF",
                model_name="win_prob",
                model_type="elo",
            ),
            _event(
                event_id="run-2-win-kc",
                run_id="run-2",
                role=ForecastRole.BACKFILLED,
                generated_at=datetime(
                    2026,
                    9,
                    2,
                    12,
                    tzinfo=UTC,
                ),
                game_id="2026_01_KC_LAC",
                model_name="win_prob",
                model_type="elo",
            ),
        ],
        ignore_index=True,
    )


def test_selects_exact_event_id() -> None:
    result = select_forecast_events(
        _events(),
        [
            _selection(
                event_id="older-live",
            )
        ],
    )

    assert result.complete
    assert result.missing == ()
    assert result.events["event_id"].tolist() == ["older-live"]


def test_selection_does_not_use_latest_generation_time() -> None:
    result = select_forecast_events(
        _events(),
        [
            _selection(
                event_id="older-live",
            )
        ],
    )

    assert result.events["event_id"].tolist() == ["older-live"]
    assert "newer-live" not in set(result.events["event_id"])


def test_selection_does_not_prefer_role_over_exact_reference() -> None:
    result = select_forecast_events(
        _events(),
        [
            _selection(
                event_id="backfilled",
            )
        ],
    )

    assert result.events["event_id"].tolist() == ["backfilled"]
    assert result.events["role"].tolist() == [ForecastRole.BACKFILLED.value]


def test_preserves_reference_order() -> None:
    result = select_forecast_events(
        _events(),
        [
            _selection(
                event_id="newer-live",
            ),
            _selection(
                event_id="older-live",
            ),
        ],
    )

    assert result.events["event_id"].tolist() == [
        "newer-live",
        "older-live",
    ]


def test_missing_reference_remains_visible() -> None:
    missing = _selection(
        event_id="missing-event",
    )

    result = select_forecast_events(
        _events(),
        [missing],
    )

    assert not result.complete
    assert result.events.empty
    assert result.missing == (missing,)
    assert list(result.events.columns) == FORECAST_EVENT_COLUMNS


def test_returns_selected_and_missing_references_together() -> None:
    missing = _selection(
        event_id="missing-event",
    )

    result = select_forecast_events(
        _events(),
        [
            _selection(
                event_id="older-live",
            ),
            missing,
        ],
    )

    assert not result.complete
    assert result.events["event_id"].tolist() == ["older-live"]
    assert result.missing == (missing,)


@pytest.mark.parametrize(
    ("field_name", "selection"),
    [
        (
            "game_id",
            _selection(
                event_id="older-live",
                game_id="different-game",
            ),
        ),
        (
            "model_name",
            _selection(
                event_id="older-live",
                model_name="total",
            ),
        ),
        (
            "model_type",
            _selection(
                event_id="older-live",
                model_type="random_forest",
            ),
        ),
    ],
)
def test_rejects_reference_identity_conflict(
    field_name: str,
    selection: SelectedForecast,
) -> None:
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        select_forecast_events(
            _events(),
            [selection],
        )


def test_rejects_duplicate_selection_references() -> None:
    selected = _selection(
        event_id="older-live",
    )

    with pytest.raises(
        ValueError,
        match="duplicate event IDs: older-live",
    ):
        select_forecast_events(
            _events(),
            [
                selected,
                selected,
            ],
        )


def test_empty_selection_returns_canonical_empty_result() -> None:
    result = select_forecast_events(
        _events(),
        [],
    )

    assert result.complete
    assert result.missing == ()
    assert result.events.empty
    assert list(result.events.columns) == FORECAST_EVENT_COLUMNS


def test_does_not_mutate_event_frame() -> None:
    events = _events()
    original = events.copy(deep=True)

    select_forecast_events(
        events,
        [
            _selection(
                event_id="older-live",
            )
        ],
    )

    pd.testing.assert_frame_equal(
        events,
        original,
    )


def test_rejects_invalid_event_schema() -> None:
    events = _events().drop(columns=["run_id"])

    with pytest.raises(
        ValueError,
        match="missing required columns: run_id",
    ):
        select_forecast_events(
            events,
            [
                _selection(
                    event_id="older-live",
                )
            ],
        )


class TestSelectForecastRun:
    """Tests for selection by exact forecast-run identity."""

    def test_selects_only_requested_run(self) -> None:
        result = select_forecast_run(
            _multi_family_run_events(),
            run_id="run-1",
        )

        assert result.found
        assert result.run_id == "run-1"
        assert len(result.events) == 3
        assert set(result.events["run_id"]) == {
            "run-1",
        }
        assert "run-2-win-kc" not in set(result.events["event_id"])

    def test_missing_run_remains_visible(self) -> None:
        result = select_forecast_run(
            _multi_family_run_events(),
            run_id="missing-run",
        )

        assert not result.found
        assert result.run_id == "missing-run"
        assert result.events.empty
        assert list(result.events.columns) == FORECAST_EVENT_COLUMNS


class TestResolveForecastCandidates:
    """Tests for explicit candidate eligibility resolution."""

    def test_selects_single_live_event(self) -> None:
        events = _event(
            event_id="live-event",
            run_id="live-run",
            role=ForecastRole.LIVE,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
        )

        resolution = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.SELECTED
        assert resolution.selected is not None
        assert resolution.selected.event_id == "live-event"
        assert resolution.eligible_event_ids == ("live-event",)

    def test_live_event_excludes_backfilled_candidate(
        self,
    ) -> None:
        events = pd.concat(
            [
                _event(
                    event_id="backfilled-event",
                    run_id="backfill-run",
                    role=ForecastRole.BACKFILLED,
                    generated_at=datetime(
                        2026,
                        9,
                        2,
                        12,
                        tzinfo=UTC,
                    ),
                ),
                _event(
                    event_id="live-event",
                    run_id="live-run",
                    role=ForecastRole.LIVE,
                    generated_at=datetime(
                        2026,
                        9,
                        1,
                        12,
                        tzinfo=UTC,
                    ),
                ),
            ],
            ignore_index=True,
        )

        resolution = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.SELECTED
        assert resolution.selected is not None
        assert resolution.selected.event_id == "live-event"
        assert resolution.eligible_event_ids == ("live-event",)

    def test_single_backfilled_event_is_eligible_when_no_live_exists(
        self,
    ) -> None:
        events = _event(
            event_id="backfilled-event",
            run_id="backfill-run",
            role=ForecastRole.BACKFILLED,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
        )

        resolution = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.SELECTED
        assert resolution.selected is not None
        assert resolution.selected.event_id == "backfilled-event"

    def test_multiple_live_runs_remain_ambiguous(
        self,
    ) -> None:
        resolution = resolve_forecast_candidates(
            _events(),
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.AMBIGUOUS
        assert resolution.selected is None
        assert resolution.eligible_event_ids == (
            "newer-live",
            "older-live",
        )

    def test_ambiguity_is_independent_from_input_order(
        self,
    ) -> None:
        events = _events()
        reversed_events = events.iloc[::-1].reset_index(
            drop=True,
        )

        first = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]
        second = resolve_forecast_candidates(
            reversed_events,
            [_candidate()],
        )[0]

        assert first.status is ForecastCandidateStatus.AMBIGUOUS
        assert second.status is ForecastCandidateStatus.AMBIGUOUS
        assert first.eligible_event_ids == second.eligible_event_ids

    def test_newer_backfill_does_not_override_live_event(
        self,
    ) -> None:
        events = pd.concat(
            [
                _event(
                    event_id="live-event",
                    run_id="live-run",
                    role=ForecastRole.LIVE,
                    generated_at=datetime(
                        2026,
                        9,
                        1,
                        12,
                        tzinfo=UTC,
                    ),
                ),
                _event(
                    event_id="newer-backfill",
                    run_id="backfill-run",
                    role=ForecastRole.BACKFILLED,
                    generated_at=datetime(
                        2026,
                        9,
                        3,
                        12,
                        tzinfo=UTC,
                    ),
                ),
            ],
            ignore_index=True,
        )

        resolution = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.SELECTED
        assert resolution.selected is not None
        assert resolution.selected.event_id == "live-event"

    def test_multiple_backfills_are_ambiguous_without_live_event(
        self,
    ) -> None:
        events = pd.concat(
            [
                _event(
                    event_id="backfill-1",
                    run_id="run-1",
                    role=ForecastRole.BACKFILLED,
                    generated_at=datetime(
                        2026,
                        9,
                        1,
                        12,
                        tzinfo=UTC,
                    ),
                ),
                _event(
                    event_id="backfill-2",
                    run_id="run-2",
                    role=ForecastRole.BACKFILLED,
                    generated_at=datetime(
                        2026,
                        9,
                        2,
                        12,
                        tzinfo=UTC,
                    ),
                ),
            ],
            ignore_index=True,
        )

        resolution = resolve_forecast_candidates(
            events,
            [_candidate()],
        )[0]

        assert resolution.status is ForecastCandidateStatus.AMBIGUOUS
        assert resolution.selected is None
        assert resolution.eligible_event_ids == (
            "backfill-1",
            "backfill-2",
        )

    def test_missing_candidate_remains_visible(self) -> None:
        resolution = resolve_forecast_candidates(
            _events(),
            [
                _candidate(
                    game_id="missing-game",
                )
            ],
        )[0]

        assert resolution.status is ForecastCandidateStatus.MISSING
        assert resolution.selected is None
        assert resolution.eligible_event_ids == ()

    def test_win_and_total_candidates_resolve_independently(
        self,
    ) -> None:
        events = _multi_family_run_events()

        resolutions = resolve_forecast_candidates(
            events,
            [
                _candidate(
                    model_name="win_prob",
                    model_type="elo",
                ),
                _candidate(
                    model_name="total",
                    model_type="random_forest",
                ),
            ],
        )

        assert len(resolutions) == 2
        assert resolutions[0].identity.model_name == "win_prob"
        assert resolutions[1].identity.model_name == "total"
        assert resolutions[1].status is ForecastCandidateStatus.SELECTED
        assert resolutions[1].selected is not None
        assert resolutions[1].selected.event_id == "run-1-total-kc"

    def test_preserves_requested_identity_order(self) -> None:
        events = _multi_family_run_events()
        total = _candidate(
            model_name="total",
            model_type="random_forest",
        )
        win = _candidate(
            game_id="2026_01_BAL_BUF",
            model_name="win_prob",
            model_type="elo",
        )

        resolutions = resolve_forecast_candidates(
            events,
            [total, win],
        )

        assert [resolution.identity for resolution in resolutions] == [
            total,
            win,
        ]

    def test_rejects_duplicate_candidate_identity(
        self,
    ) -> None:
        candidate = _candidate()

        with pytest.raises(
            ValueError,
            match=("candidate identities contain duplicates: 2026_01_KC_LAC/win_prob/elo"),
        ):
            resolve_forecast_candidates(
                _events(),
                [
                    candidate,
                    candidate,
                ],
            )

    def test_empty_candidate_list_returns_empty_tuple(
        self,
    ) -> None:
        assert (
            resolve_forecast_candidates(
                _events(),
                [],
            )
            == ()
        )

    def test_does_not_mutate_event_frame(self) -> None:
        events = _events()
        original = events.copy(deep=True)

        resolve_forecast_candidates(
            events,
            [_candidate()],
        )

        pd.testing.assert_frame_equal(
            events,
            original,
        )
