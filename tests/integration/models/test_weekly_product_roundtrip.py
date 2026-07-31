# tests/integration/models/test_weekly_product_roundtrip.py

"""Integration tests for persisted weekly game-product round trips."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.loaders import (
    load_current_weekly_product,
    load_weekly_product,
)
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.models.game_prediction.product_validation import (
    validate_weekly_game_product,
)


def _weekly_product() -> DataFrame:
    """Return one complete product with available and blocked game rows."""
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "week": [8, 8],
            "game_id": ["game-1", "game-2"],
            "away_team": ["Away One", "Away Two"],
            "home_team": ["Home One", "Home Two"],
            "neutral_site": [False, True],
            "win_status": ["available", "forecast_missing"],
            "away_win_prob": [0.40, pd.NA],
            "home_win_prob": [0.60, pd.NA],
            "win_model_name": ["win_prob", pd.NA],
            "win_model_type": ["elo", pd.NA],
            "win_event_id": ["win-1", pd.NA],
            "spread_status": ["available", "win_unavailable"],
            "model_spread": [-3.0, pd.NA],
            "spread_uncertainty": [13.5, pd.NA],
            "spread_source_event_id": ["win-1", pd.NA],
            "spread_model_name": ["win_prob", pd.NA],
            "spread_model_type": ["elo", pd.NA],
            "spread_calibration_key": ["win_prob_elo", pd.NA],
            "spread_calibration_updated_at": [
                "2026-07-30T12:00:00+00:00",
                pd.NA,
            ],
            "total_status": ["available", "forecast_missing"],
            "model_total": [44.0, pd.NA],
            "total_uncertainty": [12.8, pd.NA],
            "total_model_name": ["total", pd.NA],
            "total_model_type": ["xgboost", pd.NA],
            "total_event_id": ["total-1", pd.NA],
            "total_uncertainty_trained_at": [
                "2026-07-01T14:20:00",
                pd.NA,
            ],
            "projected_score_status": [
                "available",
                "spread_and_total_unavailable",
            ],
            "projected_home_score": [23.5, pd.NA],
            "projected_away_score": [20.5, pd.NA],
        }
    )


def _identity(
    *,
    product_id: str,
    run_id: str,
    generated_at: datetime,
) -> WeeklyProductIdentity:
    """Create one immutable weekly product identity."""
    return WeeklyProductIdentity(
        product_id=product_id,
        run_id=run_id,
        season="2026-2027",
        week=8,
        generated_at=generated_at,
    )


def _assert_no_compute_artifacts(repo: Path) -> None:
    """Verify static product loading created no unrelated artifacts."""
    forbidden = (
        repo / "data" / "models",
        repo / "data" / "modeling",
        repo / "data" / "output" / "champions",
        repo / "data" / "output" / "calibration",
        repo / "data" / "output" / "predictions" / "forecast_events.parquet",
    )
    assert all(not path.exists() for path in forbidden)


def test_weekly_product_roundtrip_and_explicit_current_selection(
    tmp_path: Path,
) -> None:
    """Persist two weekly runs and change current only by explicit selection."""
    product_a = _weekly_product()
    identity_a = _identity(
        product_id="product-a",
        run_id="run-a",
        generated_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
    )

    write_weekly_product(
        tmp_path,
        product_a,
        identity=identity_a,
    )
    select_current_weekly_product(
        tmp_path,
        identity_a.product_id,
        season=identity_a.season,
        week=identity_a.week,
        selected_at=datetime(2026, 10, 20, 12, 30, tzinfo=UTC),
    )

    loaded_a = load_current_weekly_product(
        tmp_path,
        season=identity_a.season,
        week=identity_a.week,
    )

    assert set(loaded_a["product_id"]) == {"product-a"}
    assert set(loaded_a["product_run_id"]) == {"run-a"}
    assert loaded_a["product_generated_at"].iloc[0] == pd.Timestamp(identity_a.generated_at)
    assert loaded_a.columns[4:].tolist() == product_a.columns.tolist()
    assert pd.isna(loaded_a.loc[1, "away_win_prob"])
    assert pd.isna(loaded_a.loc[1, "projected_home_score"])
    validate_weekly_game_product(
        loaded_a.drop(
            columns=[
                "product_schema_version",
                "product_id",
                "product_run_id",
                "product_generated_at",
            ]
        )
    )

    product_b = product_a.copy()
    product_b.loc[0, "model_total"] = 45.0
    product_b.loc[0, "projected_home_score"] = 24.0
    product_b.loc[0, "projected_away_score"] = 21.0
    identity_b = _identity(
        product_id="product-b",
        run_id="run-b",
        generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
    )

    write_weekly_product(
        tmp_path,
        product_b,
        identity=identity_b,
    )

    still_current = load_current_weekly_product(
        tmp_path,
        season=identity_a.season,
        week=identity_a.week,
    )
    assert set(still_current["product_id"]) == {"product-a"}

    select_current_weekly_product(
        tmp_path,
        identity_b.product_id,
        season=identity_b.season,
        week=identity_b.week,
        selected_at=datetime(2026, 10, 20, 13, 30, tzinfo=UTC),
    )
    loaded_b = load_current_weekly_product(
        tmp_path,
        season=identity_b.season,
        week=identity_b.week,
    )
    assert set(loaded_b["product_id"]) == {"product-b"}
    assert set(loaded_b["product_run_id"]) == {"run-b"}

    exact_a = load_weekly_product(tmp_path, "product-a")
    exact_b = load_weekly_product(tmp_path, "product-b")
    assert exact_a.loc[0, "model_total"] == 44.0
    assert exact_a.loc[0, "projected_home_score"] == 23.5
    assert exact_a.loc[0, "projected_away_score"] == 20.5
    assert exact_b.loc[0, "model_total"] == 45.0
    assert exact_b.loc[0, "projected_home_score"] == 24.0
    assert exact_b.loc[0, "projected_away_score"] == 21.0

    weekly_root = tmp_path / "data" / "output" / "weekly_products"
    assert (weekly_root / "index.json").is_file()
    assert (weekly_root / "current.json").is_file()
    assert (weekly_root / "products" / "product-a.parquet").is_file()
    assert (weekly_root / "products" / "product-b.parquet").is_file()
    _assert_no_compute_artifacts(tmp_path)
