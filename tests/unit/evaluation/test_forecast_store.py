# tests/unit/evaluation/test_forecast_store.py

"""Tests for immutable forecast-event storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    empty_forecast_events,
    load_forecast_events,
    validate_forecast_events,
    write_forecast_events,
)


def _event(
    *,
    event_id: str = "event-1",
    run_id: str = "run-1",
    role: ForecastRole = ForecastRole.LIVE,
    generated_at: datetime | None = None,
    season: str = "2026-2027",
    week: int = 1,
    game_id: str = "2026_01_KC_LAC",
    model_name: str = "win_prob",
    model_type: str = "elo",
    away_win_prob: float | None = 0.55,
    home_win_prob: float | None = 0.45,
    model_total: float | None = None,
) -> pd.DataFrame:
    """Create one valid forecast event."""
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "run_id": [run_id],
            "role": [role.value],
            "generated_at": [generated_at or datetime(2026, 9, 1, 12, tzinfo=UTC)],
            "season": [season],
            "week": [week],
            "game_id": [game_id],
            "model_name": [model_name],
            "model_type": [model_type],
            "game_date": ["2026-09-05"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "away_elo": [1520.0],
            "home_elo": [1480.0],
            "away_win_prob": [away_win_prob],
            "home_win_prob": [home_win_prob],
            "model_spread": [-1.5],
            "model_total": [model_total],
            "projected_home_score": [None],
            "projected_away_score": [None],
            "margin_std": [13.0],
            "win_prob_lo": [0.47],
            "win_prob_hi": [0.63],
            "confidence_tier": ["Low"],
        }
    )


def test_empty_store_has_canonical_columns() -> None:
    events = empty_forecast_events()

    assert list(events.columns) == FORECAST_EVENT_COLUMNS
    assert events.empty


def test_validate_preserves_canonical_order() -> None:
    source = _event()
    source = source.loc[
        :,
        list(reversed(FORECAST_EVENT_COLUMNS)),
    ]

    validated = validate_forecast_events(source)

    assert list(validated.columns) == FORECAST_EVENT_COLUMNS


def test_validate_rejects_missing_column() -> None:
    events = _event().drop(columns=["run_id"])

    with pytest.raises(
        ValueError,
        match="missing required columns: run_id",
    ):
        validate_forecast_events(events)


def test_validate_rejects_unknown_column() -> None:
    events = _event()
    events["unexpected"] = "value"

    with pytest.raises(
        ValueError,
        match="unknown columns: unexpected",
    ):
        validate_forecast_events(events)


def test_validate_rejects_duplicate_event_ids_in_batch() -> None:
    events = pd.concat(
        [_event(), _event()],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate event IDs",
    ):
        validate_forecast_events(events)


def test_validate_rejects_invalid_role() -> None:
    events = _event()
    events["role"] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid roles",
    ):
        validate_forecast_events(events)


def test_validate_rejects_naive_generated_at() -> None:
    events = _event(
        generated_at=datetime(2026, 9, 1, 12),
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        validate_forecast_events(events)


def test_validate_rejects_non_utc_generated_at() -> None:
    mountain_time = timezone(timedelta(hours=-6))
    events = _event(
        generated_at=datetime(
            2026,
            9,
            1,
            12,
            tzinfo=mountain_time,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use UTC",
    ):
        validate_forecast_events(events)


def test_validate_rejects_invalid_week() -> None:
    with pytest.raises(
        ValueError,
        match="week must be at least 1",
    ):
        validate_forecast_events(_event(week=0))


def test_write_and_load_round_trip(
    tmp_path: Path,
) -> None:
    write_forecast_events(
        _event(),
        repo=tmp_path,
    )

    loaded = load_forecast_events(
        repo=tmp_path,
    )

    assert len(loaded) == 1
    assert list(loaded.columns) == FORECAST_EVENT_COLUMNS
    assert loaded["event_id"].iloc[0] == "event-1"
    assert loaded["generated_at"].dt.tz is not None


def test_multiple_live_events_for_same_game_and_model_coexist(
    tmp_path: Path,
) -> None:
    write_forecast_events(
        _event(
            event_id="event-1",
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
        ),
        repo=tmp_path,
    )
    write_forecast_events(
        _event(
            event_id="event-2",
            generated_at=datetime(
                2026,
                9,
                2,
                12,
                tzinfo=UTC,
            ),
            away_win_prob=0.60,
            home_win_prob=0.40,
        ),
        repo=tmp_path,
    )

    loaded = load_forecast_events(
        repo=tmp_path,
    )

    assert len(loaded) == 2
    assert set(loaded["event_id"]) == {
        "event-1",
        "event-2",
    }


def test_live_and_backfilled_events_coexist(
    tmp_path: Path,
) -> None:
    write_forecast_events(
        _event(
            event_id="live-event",
            role=ForecastRole.LIVE,
        ),
        repo=tmp_path,
    )
    write_forecast_events(
        _event(
            event_id="backfill-event",
            run_id="backfill-run",
            role=ForecastRole.BACKFILLED,
        ),
        repo=tmp_path,
    )

    loaded = load_forecast_events(
        repo=tmp_path,
    )

    assert len(loaded) == 2
    assert set(loaded["role"]) == {
        ForecastRole.LIVE.value,
        ForecastRole.BACKFILLED.value,
    }


def test_identical_event_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    events = _event()

    write_forecast_events(
        events,
        repo=tmp_path,
    )
    write_forecast_events(
        events.copy(),
        repo=tmp_path,
    )

    loaded = load_forecast_events(
        repo=tmp_path,
    )

    assert len(loaded) == 1


def test_event_id_reuse_with_changed_content_is_rejected(
    tmp_path: Path,
) -> None:
    write_forecast_events(
        _event(
            away_win_prob=0.55,
            home_win_prob=0.45,
        ),
        repo=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="cannot be reused with different content",
    ):
        write_forecast_events(
            _event(
                away_win_prob=0.60,
                home_win_prob=0.40,
            ),
            repo=tmp_path,
        )


def test_enrichment_values_round_trip(
    tmp_path: Path,
) -> None:
    write_forecast_events(
        _event(model_total=44.5),
        repo=tmp_path,
    )

    loaded = load_forecast_events(
        repo=tmp_path,
    )

    assert loaded["model_spread"].iloc[0] == pytest.approx(-1.5)
    assert loaded["model_total"].iloc[0] == pytest.approx(44.5)
    assert loaded["margin_std"].iloc[0] == pytest.approx(13.0)
    assert loaded["win_prob_lo"].iloc[0] == pytest.approx(0.47)
    assert loaded["win_prob_hi"].iloc[0] == pytest.approx(0.63)
    assert loaded["confidence_tier"].iloc[0] == "Low"


def test_filters_do_not_select_by_write_order(
    tmp_path: Path,
) -> None:
    events = pd.concat(
        [
            _event(
                event_id="live-event",
                role=ForecastRole.LIVE,
            ),
            _event(
                event_id="backfill-event",
                run_id="backfill-run",
                role=ForecastRole.BACKFILLED,
            ),
        ],
        ignore_index=True,
    )
    write_forecast_events(
        events,
        repo=tmp_path,
    )

    live = load_forecast_events(
        role=ForecastRole.LIVE,
        repo=tmp_path,
    )
    backfilled = load_forecast_events(
        role=ForecastRole.BACKFILLED,
        repo=tmp_path,
    )

    assert live["event_id"].tolist() == ["live-event"]
    assert backfilled["event_id"].tolist() == ["backfill-event"]
