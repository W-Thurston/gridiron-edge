# tests/unit/evaluation/test_forecast_events.py

"""Tests for composing canonical predictions into forecast events."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS


def _predictions() -> pd.DataFrame:
    """Create canonical prediction rows for two scheduled games."""
    return pd.DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "week": [1, 1],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
            ],
            "game_date": [
                "2026-09-05",
                "2026-09-06",
            ],
            "away_team": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
            ],
            "home_team": [
                "Los Angeles Chargers",
                "Buffalo Bills",
            ],
            "away_elo": [1520.0, 1510.0],
            "home_elo": [1480.0, 1530.0],
            "away_win_prob": [0.55, 0.48],
            "home_win_prob": [0.45, 0.52],
        }
    )


def test_builds_canonical_forecast_events() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    events = build_forecast_events(
        _predictions(),
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=generated_at,
    )

    assert list(events.columns) == FORECAST_EVENT_COLUMNS
    assert len(events) == 2
    assert events["model_name"].tolist() == [
        "win_prob",
        "win_prob",
    ]
    assert events["model_type"].tolist() == [
        "elo",
        "elo",
    ]


def test_assigns_distinct_event_id_per_row() -> None:
    events = build_forecast_events(
        _predictions(),
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    event_ids = events["event_id"].tolist()

    assert len(set(event_ids)) == 2
    assert all(UUID(event_id) for event_id in event_ids)


def test_applies_shared_invocation_identity() -> None:
    generated_at = datetime(2026, 9, 1, 12, tzinfo=UTC)

    events = build_forecast_events(
        _predictions(),
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=generated_at,
    )

    assert events["run_id"].tolist() == [
        "run-1",
        "run-1",
    ]
    assert events["role"].tolist() == [
        ForecastRole.LIVE.value,
        ForecastRole.LIVE.value,
    ]
    assert events["generated_at"].tolist() == [
        pd.Timestamp(generated_at),
        pd.Timestamp(generated_at),
    ]


def test_applies_backfilled_role() -> None:
    events = build_forecast_events(
        _predictions(),
        model_name="win_prob",
        model_type="random_forest",
        run_id="backfill-run",
        role=ForecastRole.BACKFILLED,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    assert (events["role"] == ForecastRole.BACKFILLED.value).all()


def test_preserves_available_prediction_values() -> None:
    predictions = _predictions()
    predictions["model_spread"] = [-1.5, 2.0]
    predictions["model_total"] = [44.5, 47.0]
    predictions["projected_home_score"] = [21.5, 24.5]
    predictions["projected_away_score"] = [23.0, 22.5]
    predictions["margin_std"] = [13.0, 12.5]
    predictions["win_prob_lo"] = [0.47, 0.40]
    predictions["win_prob_hi"] = [0.63, 0.56]
    predictions["confidence_tier"] = ["Low", "Moderate"]

    events = build_forecast_events(
        predictions,
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    assert events["model_spread"].tolist() == pytest.approx([-1.5, 2.0])
    assert events["model_total"].tolist() == pytest.approx([44.5, 47.0])
    assert events["projected_home_score"].tolist() == pytest.approx([21.5, 24.5])
    assert events["projected_away_score"].tolist() == pytest.approx([23.0, 22.5])
    assert events["confidence_tier"].tolist() == [
        "Low",
        "Moderate",
    ]


def test_missing_optional_values_remain_null() -> None:
    predictions = _predictions().drop(
        columns=[
            "game_date",
            "away_elo",
            "home_elo",
            "away_win_prob",
            "home_win_prob",
        ]
    )

    events = build_forecast_events(
        predictions,
        model_name="total",
        model_type="random_forest",
        run_id="run-1",
        role=ForecastRole.BACKFILLED,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    assert events["game_date"].isna().all()
    assert events["away_elo"].isna().all()
    assert events["home_elo"].isna().all()
    assert events["away_win_prob"].isna().all()
    assert events["home_win_prob"].isna().all()
    assert events["model_total"].isna().all()


def test_does_not_mutate_source_predictions() -> None:
    predictions = _predictions()
    original = predictions.copy(deep=True)

    build_forecast_events(
        predictions,
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    pd.testing.assert_frame_equal(
        predictions,
        original,
    )


def test_empty_input_returns_canonical_empty_frame() -> None:
    predictions = _predictions().iloc[0:0]

    events = build_forecast_events(
        predictions,
        model_name="win_prob",
        model_type="elo",
        run_id="run-1",
        role=ForecastRole.LIVE,
        generated_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )

    assert events.empty
    assert list(events.columns) == FORECAST_EVENT_COLUMNS


@pytest.mark.parametrize(
    "missing_column",
    [
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
    ],
)
def test_rejects_missing_required_prediction_column(
    missing_column: str,
) -> None:
    predictions = _predictions().drop(columns=[missing_column])

    with pytest.raises(
        ValueError,
        match=f"missing required columns: {missing_column}",
    ):
        build_forecast_events(
            predictions,
            model_name="win_prob",
            model_type="elo",
            run_id="run-1",
            role=ForecastRole.LIVE,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
        )


def test_rejects_empty_run_id() -> None:
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        build_forecast_events(
            _predictions(),
            model_name="win_prob",
            model_type="elo",
            run_id=" ",
            role=ForecastRole.LIVE,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
                tzinfo=UTC,
            ),
        )


def test_rejects_naive_generated_at() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        build_forecast_events(
            _predictions(),
            model_name="win_prob",
            model_type="elo",
            run_id="run-1",
            role=ForecastRole.LIVE,
            generated_at=datetime(
                2026,
                9,
                1,
                12,
            ),
        )
