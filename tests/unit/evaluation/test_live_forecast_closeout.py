"""Tests for selected live forecast closeout."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from gridiron_edge.evaluation.live_forecast_closeout import close_live_forecasts


def _product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["product-1", "product-1"],
            "product_run_id": ["run-1", "run-1"],
            "season": ["2025-2026", "2025-2026"],
            "week": [1, 1],
            "game_id": ["g1", "g2"],
            "away_team": ["Away One", "Away Two"],
            "home_team": ["Home One", "Home Two"],
            "win_status": ["available", "available"],
            "win_event_id": ["w1", "w2"],
            "win_run_id": ["run-1", "run-1"],
            "win_model_name": ["win_prob", "win_prob"],
            "win_model_type": ["logistic", "logistic"],
            "total_status": ["available", "available"],
            "total_event_id": ["t1", "t2"],
            "total_run_id": ["run-1", "run-1"],
            "total_model_name": ["total", "total"],
            "total_model_type": ["random_forest", "random_forest"],
        }
    )


def _event(
    *,
    event_id: str,
    game_id: str,
    model_name: str,
    model_type: str,
    home_win_prob: float | None = None,
    model_total: float | None = None,
    role: str = "live",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "run_id": "run-1",
        "role": role,
        "season": "2025-2026",
        "week": 1,
        "game_id": game_id,
        "model_name": model_name,
        "model_type": model_type,
        "home_win_prob": home_win_prob,
        "model_total": model_total,
    }


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _event(
                event_id="w1",
                game_id="g1",
                model_name="win_prob",
                model_type="logistic",
                home_win_prob=0.75,
            ),
            _event(
                event_id="w2",
                game_id="g2",
                model_name="win_prob",
                model_type="logistic",
                home_win_prob=0.40,
            ),
            _event(
                event_id="t1",
                game_id="g1",
                model_name="total",
                model_type="random_forest",
                model_total=44.0,
            ),
            _event(
                event_id="t2",
                game_id="g2",
                model_name="total",
                model_type="random_forest",
                model_total=40.0,
            ),
        ]
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "YEAR": ["2025-2026", "2025-2026"],
            "WEEK_NUM": [1, 1],
            "AWAY_SCORE": [20, 17],
            "HOME_SCORE": [27, 17],
        }
    )


def test_complete_closeout_uses_exact_selected_live_events() -> None:
    result = close_live_forecasts(product=_product(), forecast_events=_events(), games=_games())

    assert result.complete
    assert result.scheduled_game_count == 2
    assert result.completed_outcome_count == 2
    assert result.matched_win_event_count == 2
    assert result.matched_total_event_count == 2
    assert result.win.evaluated_count == 1
    assert result.win.brier == pytest.approx(0.0625)
    assert result.win.accuracy == pytest.approx(1.0)
    assert result.total.evaluated_count == 2
    assert result.total.mae == pytest.approx(4.5)
    assert result.total.rmse == pytest.approx(22.5**0.5)
    assert result.total.bias == pytest.approx(1.5)


def test_backfilled_event_cannot_satisfy_selected_live_reference() -> None:
    events = _events()
    events.loc[events["event_id"].eq("w1"), "role"] = "backfilled"

    result = close_live_forecasts(product=_product(), forecast_events=events, games=_games())

    assert not result.complete
    assert result.missing_win_event_game_ids == ("g1",)
    assert result.matched_win_event_count == 1


def test_missing_outcome_remains_visible() -> None:
    games = _games().loc[lambda frame: frame["GAME_ID"].eq("g1"), :]

    result = close_live_forecasts(product=_product(), forecast_events=_events(), games=games)

    assert not result.complete
    assert result.missing_outcome_game_ids == ("g2",)
    missing_outcome = result.reconciliation.loc[
        result.reconciliation["game_id"].eq("g2"),
        "outcome_available",
    ].iloc[0]
    assert not bool(missing_outcome)


def test_unavailable_component_is_not_mislabeled_as_missing_event() -> None:
    product = _product()
    product.loc[product["game_id"].eq("g2"), "total_status"] = "forecast_missing"
    product.loc[
        product["game_id"].eq("g2"),
        [
            "total_event_id",
            "total_run_id",
            "total_model_name",
            "total_model_type",
        ],
    ] = pd.NA

    result = close_live_forecasts(product=product, forecast_events=_events(), games=_games())

    assert not result.complete
    assert result.missing_total_component_game_ids == ("g2",)
    assert result.missing_total_event_game_ids == ()


def test_provenance_mismatch_does_not_match_event() -> None:
    product = _product()
    product.loc[product["game_id"].eq("g1"), "win_run_id"] = "other-run"

    result = close_live_forecasts(product=product, forecast_events=_events(), games=_games())

    assert result.missing_win_event_game_ids == ("g1",)


def test_inputs_are_not_mutated() -> None:
    product = _product()
    events = _events()
    games = _games()
    original = [deepcopy(frame) for frame in (product, events, games)]

    close_live_forecasts(product=product, forecast_events=events, games=games)

    for frame, expected in zip((product, events, games), original, strict=True):
        pd.testing.assert_frame_equal(frame, expected)


def test_required_columns_are_validated() -> None:
    with pytest.raises(ValueError, match="win_event_id"):
        close_live_forecasts(
            product=_product().drop(columns=["win_event_id"]),
            forecast_events=_events(),
            games=_games(),
        )
