# tests/unit/models/game_prediction/test_weekly_product_store.py

"""Tests for immutable weekly game-product storage."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.models.game_prediction.weekly_product_store import (
    WEEKLY_PRODUCT_SCHEMA_VERSION,
    get_current_weekly_product_selection,
    list_weekly_products,
    load_current_weekly_product,
    load_weekly_product,
    select_current_weekly_product,
    weekly_product_artifact_path,
    weekly_product_root,
    write_weekly_product,
)


def _product(
    *,
    run_id: str = "run-1",
) -> DataFrame:
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "week": [8, 8],
            "game_id": ["game-1", "game-2"],
            "away_team": ["Away One", "Away Two"],
            "home_team": ["Home One", "Home Two"],
            "neutral_site": [False, True],
            "win_status": ["available", "forecast_missing"],
            "win_selection_status": ["selected", "missing"],
            "away_win_prob": [0.40, pd.NA],
            "home_win_prob": [0.60, pd.NA],
            "win_model_name": ["win_prob", pd.NA],
            "win_model_type": ["elo", pd.NA],
            "win_event_id": ["win-1", pd.NA],
            "win_run_id": [run_id, pd.NA],
            "win_generated_at": ["2026-10-20T12:00:00+00:00", pd.NaT],
            "win_role": ["live", pd.NA],
            "spread_status": ["available", "win_unavailable"],
            "model_spread": [-3.0, pd.NA],
            "spread_uncertainty": [13.5, pd.NA],
            "spread_source_event_id": ["win-1", pd.NA],
            "spread_model_name": ["win_prob", pd.NA],
            "spread_model_type": ["elo", pd.NA],
            "spread_calibration_key": ["win_prob_elo", pd.NA],
            "spread_calibration_updated_at": ["2026-07-30T12:00:00+00:00", pd.NA],
            "total_status": ["available", "forecast_missing"],
            "model_total": [44.0, pd.NA],
            "total_uncertainty": [12.8, pd.NA],
            "total_model_name": ["total", pd.NA],
            "total_model_type": ["xgboost", pd.NA],
            "total_event_id": ["total-1", pd.NA],
            "total_run_id": [run_id, pd.NA],
            "total_generated_at": ["2026-10-20T12:00:00+00:00", pd.NaT],
            "total_role": ["live", pd.NA],
            "total_selection_status": ["selected", "missing"],
            "total_uncertainty_trained_at": ["2026-07-01T14:20:00", pd.NA],
            "projected_score_status": ["available", "spread_and_total_unavailable"],
            "projected_home_score": [23.5, pd.NA],
            "projected_away_score": [20.5, pd.NA],
        }
    )


def _identity(
    product_id: str = "product-1",
    *,
    run_id: str = "run-1",
    generated_at: datetime | None = None,
) -> WeeklyProductIdentity:
    return WeeklyProductIdentity(
        product_id=product_id,
        run_id=run_id,
        season="2026-2027",
        week=8,
        generated_at=generated_at or datetime(2026, 10, 20, 12, tzinfo=UTC),
    )


def _index_path(repo: Path) -> Path:
    return weekly_product_root(repo) / "index.json"


def _read_index(repo: Path) -> dict[str, object]:
    return json.loads(_index_path(repo).read_text())


def test_product_round_trips_without_schema_loss(tmp_path: Path) -> None:
    source = _product()
    identity = _identity()

    write_weekly_product(source, identity=identity, repo=tmp_path)
    loaded = load_weekly_product(identity.product_id, repo=tmp_path)

    assert loaded.columns[:4].tolist() == [
        "product_schema_version",
        "product_id",
        "product_run_id",
        "product_generated_at",
    ]
    assert loaded.columns[4:].tolist() == source.columns.tolist()
    assert loaded["game_id"].tolist() == source["game_id"].tolist()
    assert loaded["product_id"].tolist() == [identity.product_id] * len(source)
    assert loaded["product_run_id"].tolist() == [identity.run_id] * len(source)
    assert loaded["product_schema_version"].tolist() == [WEEKLY_PRODUCT_SCHEMA_VERSION] * len(
        source
    )
    assert loaded["product_generated_at"].dt.tz is not None
    assert loaded["product_generated_at"].iloc[0] == pd.Timestamp(identity.generated_at)
    assert pd.isna(loaded.loc[1, "away_win_prob"])


def test_multiple_runs_for_same_week_coexist(tmp_path: Path) -> None:
    source = _product()
    first = _identity("product-1", run_id="run-1")
    second = _identity(
        "product-2",
        run_id="run-2",
        generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
    )

    write_weekly_product(source, identity=first, repo=tmp_path)
    changed = _product(run_id=second.run_id)
    changed.loc[0, "model_total"] = 45.0
    changed.loc[0, "projected_home_score"] = 24.0
    changed.loc[0, "projected_away_score"] = 21.0
    write_weekly_product(changed, identity=second, repo=tmp_path)

    assert load_weekly_product("product-1", repo=tmp_path).loc[0, "model_total"] == 44.0
    assert load_weekly_product("product-2", repo=tmp_path).loc[0, "model_total"] == 45.0
    records = list_weekly_products(season="2026-2027", week=8, repo=tmp_path)
    assert [record.product_id for record in records] == ["product-1", "product-2"]


def test_identical_rewrite_is_idempotent(tmp_path: Path) -> None:
    identity = _identity()
    first_path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    first_bytes = first_path.read_bytes()

    second_path = write_weekly_product(_product(), identity=identity, repo=tmp_path)

    assert second_path == first_path
    assert second_path.read_bytes() == first_bytes
    products = _read_index(tmp_path)["products"]
    assert isinstance(products, dict)
    assert list(products) == [identity.product_id]


@pytest.mark.parametrize(
    ("identity", "mutate"),
    [
        (_identity(run_id="different-run"), False),
        (
            _identity(
                generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
            ),
            False,
        ),
        (_identity(), True),
    ],
)
def test_conflicting_rewrite_is_rejected(
    tmp_path: Path,
    identity: WeeklyProductIdentity,
    mutate: bool,
) -> None:
    write_weekly_product(_product(), identity=_identity(), repo=tmp_path)
    incoming = _product(run_id=identity.run_id)
    if mutate:
        incoming.loc[0, "model_total"] = 45.0
        incoming.loc[0, "projected_home_score"] = 24.0
        incoming.loc[0, "projected_away_score"] = 21.0

    with pytest.raises(ValueError, match="cannot be reused"):
        write_weekly_product(incoming, identity=identity, repo=tmp_path)


def test_win_run_mismatch_fails_before_store_write(tmp_path: Path) -> None:
    product = _product()
    product.loc[0, "win_run_id"] = "different-run"

    with pytest.raises(ValueError, match="Win run_id must match"):
        write_weekly_product(product, identity=_identity(), repo=tmp_path)

    assert not weekly_product_artifact_path("product-1", repo=tmp_path).exists()
    assert not _index_path(tmp_path).exists()


def test_total_run_mismatch_fails_before_store_write(tmp_path: Path) -> None:
    product = _product()
    product.loc[0, "total_run_id"] = "different-run"

    with pytest.raises(ValueError, match="Total run_id must match"):
        write_weekly_product(product, identity=_identity(), repo=tmp_path)

    assert not weekly_product_artifact_path("product-1", repo=tmp_path).exists()
    assert not _index_path(tmp_path).exists()


def test_index_schema_mismatch_fails_clearly(tmp_path: Path) -> None:
    root = weekly_product_root(tmp_path)
    root.mkdir(parents=True)
    _index_path(tmp_path).write_text(json.dumps({"schema_version": 999, "products": {}}))

    with pytest.raises(ValueError, match="Unsupported weekly product index schema"):
        load_weekly_product("product-1", repo=tmp_path)


def test_artifact_column_mismatch_fails_clearly(tmp_path: Path) -> None:
    identity = _identity()
    path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    stored = pd.read_parquet(path).drop(columns=["neutral_site"])
    stored.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="column mismatch"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_index_row_count_mismatch_fails_clearly(tmp_path: Path) -> None:
    identity = _identity()
    write_weekly_product(_product(), identity=identity, repo=tmp_path)
    index = _read_index(tmp_path)
    products = index["products"]
    assert isinstance(products, dict)
    entry = products[identity.product_id]
    assert isinstance(entry, dict)
    entry["row_count"] = 99
    _index_path(tmp_path).write_text(json.dumps(index))

    with pytest.raises(ValueError, match="row-count mismatch"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_artifact_identity_mismatch_fails_clearly(tmp_path: Path) -> None:
    identity = _identity()
    path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    stored = pd.read_parquet(path)
    stored["product_run_id"] = "different-run"
    stored.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="run_id mismatch"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_artifact_season_mismatch_fails_clearly(tmp_path: Path) -> None:
    identity = _identity()
    path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    stored = pd.read_parquet(path)
    stored["season"] = "2025-2026"
    stored.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="season mismatch"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_missing_artifact_fails_clearly(tmp_path: Path) -> None:
    identity = _identity()
    path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    path.unlink()

    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_unindexed_artifact_fails_clearly(tmp_path: Path) -> None:
    path = weekly_product_artifact_path("orphan", repo=tmp_path)
    path.parent.mkdir(parents=True)
    _product().to_parquet(path, index=False)

    with pytest.raises(ValueError, match="without index entry"):
        load_weekly_product("orphan", repo=tmp_path)


def test_load_does_not_import_prediction_or_model_computation(tmp_path: Path) -> None:
    identity = _identity()
    write_weekly_product(_product(), identity=identity, repo=tmp_path)
    module_path = Path("src/gridiron_edge/models/game_prediction/weekly_product_store.py")
    source = module_path.read_text()

    forbidden = (
        "load_prediction_policy",
        "build_weekly_win_product",
        "attach_derived_spreads",
        "attach_selected_totals",
        "build_weekly_game_product",
        "ArtifactStore",
    )
    assert all(name not in source for name in forbidden)
    assert not load_weekly_product(identity.product_id, repo=tmp_path).empty


def _current_path(repo: Path) -> Path:
    return weekly_product_root(repo) / "current.json"


def test_current_product_selection_is_explicit(tmp_path: Path) -> None:
    first = _identity("product-1", run_id="run-1")
    second = _identity(
        "product-2",
        run_id="run-2",
        generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
    )
    write_weekly_product(_product(), identity=first, repo=tmp_path)
    write_weekly_product(_product(run_id=second.run_id), identity=second, repo=tmp_path)

    selected_at = datetime(2026, 10, 20, 14, tzinfo=UTC)
    selection = select_current_weekly_product(
        first.product_id,
        season=first.season,
        week=first.week,
        selected_at=selected_at,
        repo=tmp_path,
    )

    assert selection.product_id == first.product_id
    assert selection.selected_at == selected_at
    current = load_current_weekly_product(
        season=first.season,
        week=first.week,
        repo=tmp_path,
    )
    assert set(current["product_id"]) == {first.product_id}


def test_writing_newer_product_does_not_change_current(tmp_path: Path) -> None:
    first = _identity("product-1", run_id="run-1")
    write_weekly_product(_product(), identity=first, repo=tmp_path)
    select_current_weekly_product(
        first.product_id,
        season=first.season,
        week=first.week,
        selected_at=datetime(2026, 10, 20, 12, 30, tzinfo=UTC),
        repo=tmp_path,
    )

    second = _identity(
        "product-2",
        run_id="run-2",
        generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
    )
    write_weekly_product(_product(run_id=second.run_id), identity=second, repo=tmp_path)

    current = load_current_weekly_product(
        season=first.season,
        week=first.week,
        repo=tmp_path,
    )
    assert set(current["product_id"]) == {first.product_id}


def test_current_selection_can_be_changed_explicitly(tmp_path: Path) -> None:
    first = _identity("product-1", run_id="run-1")
    second = _identity(
        "product-2",
        run_id="run-2",
        generated_at=datetime(2026, 10, 20, 13, tzinfo=UTC),
    )
    write_weekly_product(_product(), identity=first, repo=tmp_path)
    write_weekly_product(_product(run_id=second.run_id), identity=second, repo=tmp_path)
    select_current_weekly_product(
        first.product_id,
        season=first.season,
        week=first.week,
        selected_at=datetime(2026, 10, 20, 14, tzinfo=UTC),
        repo=tmp_path,
    )
    select_current_weekly_product(
        second.product_id,
        season=second.season,
        week=second.week,
        selected_at=datetime(2026, 10, 20, 15, tzinfo=UTC),
        repo=tmp_path,
    )

    selection = get_current_weekly_product_selection(
        season=second.season,
        week=second.week,
        repo=tmp_path,
    )
    assert selection.product_id == second.product_id
    current = load_current_weekly_product(
        season=second.season,
        week=second.week,
        repo=tmp_path,
    )
    assert set(current["product_id"]) == {second.product_id}


def test_missing_current_selection_fails_clearly(tmp_path: Path) -> None:
    write_weekly_product(_product(), identity=_identity(), repo=tmp_path)

    with pytest.raises(FileNotFoundError, match="No current weekly product selected"):
        load_current_weekly_product(
            season="2026-2027",
            week=8,
            repo=tmp_path,
        )


def test_selection_requires_indexed_product(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not indexed"):
        select_current_weekly_product(
            "missing-product",
            season="2026-2027",
            week=8,
            selected_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
            repo=tmp_path,
        )


def test_selection_scope_must_match_product(tmp_path: Path) -> None:
    identity = _identity()
    write_weekly_product(_product(), identity=identity, repo=tmp_path)

    with pytest.raises(ValueError, match="scope does not match"):
        select_current_weekly_product(
            identity.product_id,
            season=identity.season,
            week=9,
            selected_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
            repo=tmp_path,
        )


def test_current_schema_mismatch_fails_clearly(tmp_path: Path) -> None:
    root = weekly_product_root(tmp_path)
    root.mkdir(parents=True)
    _current_path(tmp_path).write_text(json.dumps({"schema_version": 999, "selections": {}}))

    with pytest.raises(ValueError, match="Unsupported weekly product current schema"):
        load_current_weekly_product(
            season="2026-2027",
            week=8,
            repo=tmp_path,
        )


def test_current_selection_target_must_remain_loadable(tmp_path: Path) -> None:
    identity = _identity()
    path = write_weekly_product(_product(), identity=identity, repo=tmp_path)
    select_current_weekly_product(
        identity.product_id,
        season=identity.season,
        week=identity.week,
        selected_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
        repo=tmp_path,
    )
    path.unlink()

    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        load_current_weekly_product(
            season=identity.season,
            week=identity.week,
            repo=tmp_path,
        )
