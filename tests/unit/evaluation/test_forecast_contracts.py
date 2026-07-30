# tests/unit/evaluation/test_forecast_contracts.py

"""Tests for immutable forecast event identity contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from gridiron_edge.evaluation.forecast_contracts import (
    ForecastEventIdentity,
    ForecastRole,
    SelectedForecast,
    WeeklyProductIdentity,
    new_forecast_run_id,
)


def _identity(
    *,
    event_id: str = "event-1",
    run_id: str = "run-1",
    role: ForecastRole = ForecastRole.LIVE,
    generated_at: datetime | None = None,
    game_id: str = "2026_01_KC_LAC",
    model_name: str = "win_prob",
    model_type: str = "elo",
) -> ForecastEventIdentity:
    return ForecastEventIdentity(
        event_id=event_id,
        run_id=run_id,
        role=role,
        generated_at=generated_at or datetime(2026, 9, 1, 12, tzinfo=UTC),
        game_id=game_id,
        model_name=model_name,
        model_type=model_type,
    )


def test_forecast_roles_are_distinct() -> None:
    assert ForecastRole.LIVE.value == "live"
    assert ForecastRole.BACKFILLED.value == "backfilled"


def test_invalid_role_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        ForecastRole("invalid")


def test_same_game_and_model_can_have_multiple_events() -> None:
    first = _identity(event_id="event-1")
    second = _identity(event_id="event-2")

    assert first.game_id == second.game_id
    assert first.model_name == second.model_name
    assert first.model_type == second.model_type
    assert first.event_id != second.event_id


def test_one_run_groups_multiple_game_forecasts() -> None:
    first = _identity(
        event_id="event-1",
        run_id="run-1",
        game_id="2026_01_KC_LAC",
    )
    second = _identity(
        event_id="event-2",
        run_id="run-1",
        game_id="2026_01_BAL_BUF",
    )

    assert first.run_id == second.run_id
    assert first.game_id != second.game_id


def test_live_and_backfilled_events_can_share_game_and_model() -> None:
    live = _identity(
        event_id="live-event",
        role=ForecastRole.LIVE,
    )
    backfilled = _identity(
        event_id="backfill-event",
        role=ForecastRole.BACKFILLED,
    )

    assert live.game_id == backfilled.game_id
    assert live.model_name == backfilled.model_name
    assert live.model_type == backfilled.model_type
    assert live.role != backfilled.role
    assert live.event_id != backfilled.event_id


def test_generated_at_requires_timezone_information() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _identity(
            generated_at=datetime(2026, 9, 1, 12),
        )


def test_generated_at_requires_utc() -> None:
    mountain_time = timezone(timedelta(hours=-6))

    with pytest.raises(ValueError, match="must use UTC"):
        _identity(
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=mountain_time,
            ),
        )


def test_empty_identity_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="event_id"):
        _identity(event_id=" ")

    with pytest.raises(ValueError, match="run_id"):
        _identity(run_id=" ")

    with pytest.raises(ValueError, match="game_id"):
        _identity(game_id=" ")

    with pytest.raises(ValueError, match="model_name"):
        _identity(model_name=" ")

    with pytest.raises(ValueError, match="model_type"):
        _identity(model_type=" ")


def test_identity_is_immutable() -> None:
    identity = _identity()

    with pytest.raises(FrozenInstanceError):
        identity.game_id = "2026_01_BAL_BUF"  # type: ignore[misc]


def test_create_assigns_event_id_and_preserves_run_id() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    identity = ForecastEventIdentity.create(
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=generated_at,
        game_id="2026_01_KC_LAC",
        model_name="win_prob",
        model_type="elo",
    )

    assert UUID(identity.event_id)
    assert identity.run_id == "run-1"
    assert identity.generated_at == generated_at


def test_new_forecast_run_id_is_uuid_string() -> None:
    run_id = new_forecast_run_id()

    assert UUID(run_id)


def test_selected_forecast_references_event_identity() -> None:
    event = _identity(
        event_id="event-1",
        game_id="2026_01_KC_LAC",
        model_name="win_prob",
        model_type="elo",
    )

    selected = SelectedForecast.from_event(event)

    assert selected.event_id == event.event_id
    assert selected.game_id == event.game_id
    assert selected.model_name == event.model_name
    assert selected.model_type == event.model_type


def test_selected_forecast_is_independent_from_write_order() -> None:
    first = _identity(
        event_id="event-1",
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    second = _identity(
        event_id="event-2",
        generated_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
    )

    selected = SelectedForecast.from_event(first)

    assert selected.event_id == first.event_id
    assert selected.event_id != second.event_id


def test_selected_forecast_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="event_id"):
        SelectedForecast(
            event_id=" ",
            game_id="2026_01_KC_LAC",
            model_name="win_prob",
            model_type="elo",
        )

    with pytest.raises(ValueError, match="game_id"):
        SelectedForecast(
            event_id="event-1",
            game_id=" ",
            model_name="win_prob",
            model_type="elo",
        )

    with pytest.raises(ValueError, match="model_name"):
        SelectedForecast(
            event_id="event-1",
            game_id="2026_01_KC_LAC",
            model_name=" ",
            model_type="elo",
        )

    with pytest.raises(ValueError, match="model_type"):
        SelectedForecast(
            event_id="event-1",
            game_id="2026_01_KC_LAC",
            model_name="win_prob",
            model_type=" ",
        )


def test_selected_forecast_is_immutable() -> None:
    selected = SelectedForecast(
        event_id="event-1",
        game_id="2026_01_KC_LAC",
        model_name="win_prob",
        model_type="elo",
    )

    with pytest.raises(FrozenInstanceError):
        selected.event_id = "event-2"  # type: ignore[misc]


def test_weekly_product_identity_preserves_scope() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    identity = WeeklyProductIdentity(
        product_id="product-1",
        run_id="run-1",
        season="2026-2027",
        week=1,
        generated_at=generated_at,
    )

    assert identity.product_id == "product-1"
    assert identity.run_id == "run-1"
    assert identity.season == "2026-2027"
    assert identity.week == 1
    assert identity.generated_at == generated_at


def test_weekly_product_create_assigns_product_id() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    identity = WeeklyProductIdentity.create(
        run_id="run-1",
        season="2026-2027",
        week=1,
        generated_at=generated_at,
    )

    assert UUID(identity.product_id)
    assert identity.run_id == "run-1"
    assert identity.generated_at == generated_at


def test_weekly_products_from_same_run_have_distinct_product_ids() -> None:
    first = WeeklyProductIdentity.create(
        run_id="run-1",
        season="2026-2027",
        week=1,
    )
    second = WeeklyProductIdentity.create(
        run_id="run-1",
        season="2026-2027",
        week=1,
    )

    assert first.run_id == second.run_id
    assert first.product_id != second.product_id


def test_weekly_product_rejects_invalid_week() -> None:
    with pytest.raises(ValueError, match="week must be at least 1"):
        WeeklyProductIdentity.create(
            run_id="run-1",
            season="2026-2027",
            week=0,
        )


def test_weekly_product_rejects_empty_scope_fields() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    with pytest.raises(ValueError, match="product_id"):
        WeeklyProductIdentity(
            product_id=" ",
            run_id="run-1",
            season="2026-2027",
            week=1,
            generated_at=generated_at,
        )

    with pytest.raises(ValueError, match="run_id"):
        WeeklyProductIdentity(
            product_id="product-1",
            run_id=" ",
            season="2026-2027",
            week=1,
            generated_at=generated_at,
        )

    with pytest.raises(ValueError, match="season"):
        WeeklyProductIdentity(
            product_id="product-1",
            run_id="run-1",
            season=" ",
            week=1,
            generated_at=generated_at,
        )


def test_weekly_product_requires_timezone_aware_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        WeeklyProductIdentity(
            product_id="product-1",
            run_id="run-1",
            season="2026-2027",
            week=1,
            generated_at=datetime(2026, 9, 1, 12),
        )

    mountain_time = timezone(timedelta(hours=-6))

    with pytest.raises(ValueError, match="must use UTC"):
        WeeklyProductIdentity(
            product_id="product-1",
            run_id="run-1",
            season="2026-2027",
            week=1,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=mountain_time,
            ),
        )


def test_weekly_product_identity_is_immutable() -> None:
    identity = WeeklyProductIdentity.create(
        run_id="run-1",
        season="2026-2027",
        week=1,
    )

    with pytest.raises(FrozenInstanceError):
        identity.week = 2  # type: ignore[misc]
