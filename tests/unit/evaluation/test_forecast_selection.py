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
