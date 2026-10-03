# tests/unit/models/game_prediction/test_weekly_starting_qb.py

"""Tests for the assumed starting quarterbacks recorded in a weekly product."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest
from tests.unit.models.game_prediction.test_weekly_product_store import _identity, _product

from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.models.game_prediction.weekly_product_store import (
    WEEKLY_PRODUCT_SCHEMA_VERSION,
    load_weekly_product,
    weekly_product_artifact_path,
    write_weekly_product,
)
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    STARTING_QB_COLUMNS,
    attach_starting_qb_assumptions,
    has_starting_qb_columns,
    validate_starting_qb_record,
)

pytestmark = pytest.mark.unit

AS_OF = "2026-10-03T12:27:18Z"


def _sidecar() -> DataFrame:
    """Team A: played wk1, projected wk2. Team B: nflverse value wk2. Team C: carried wk2."""
    return DataFrame(
        {
            "GAME_ID": ["2026_01_AAA_BBB", "2026_02_AAA_BBB", "2026_02_CCC_DDD"],
            "YEAR": ["2026-2027"] * 3,
            "WEEK_NUM": [1, 2, 2],
            "AWAY_TEAM": ["AAA", "AAA", "CCC"],
            "HOME_TEAM": ["BBB", "BBB", "DDD"],
            "AWAY_QB_ID": ["a1", pd.NA, pd.NA],
            "AWAY_QB_NAME": ["Ann One", pd.NA, pd.NA],
            "HOME_QB_ID": ["b1", "b2", pd.NA],
            "HOME_QB_NAME": ["Bob One", "Bob Two", pd.NA],
            "AWAY_QB_ID_PROJECTED": [pd.NA, "a2", pd.NA],
            "AWAY_QB_PROJECTION_SOURCE": [pd.NA, "depth_chart_report", pd.NA],
            "AWAY_QB_PROJECTION_AS_OF": [pd.NA, AS_OF, pd.NA],
            "HOME_QB_ID_PROJECTED": [pd.NA, pd.NA, pd.NA],
            "HOME_QB_PROJECTION_SOURCE": [pd.NA, pd.NA, pd.NA],
            "HOME_QB_PROJECTION_AS_OF": [pd.NA, pd.NA, pd.NA],
        }
    )


def _games(*game_ids: str) -> DataFrame:
    return DataFrame({"game_id": list(game_ids)})


def test_each_source_is_recorded_with_provenance() -> None:
    names = {"a2": "Ann Two"}
    # Team CCC/DDD have no earlier known starter, so they stay unresolved.
    record = attach_starting_qb_assumptions(_games("2026_02_AAA_BBB"), _sidecar(), names).iloc[0]

    assert record["starting_qb_status"] == "available"
    assert record["away_qb_id"] == "a2"
    assert record["away_qb_name"] == "Ann Two"
    assert record["away_qb_source"] == "depth_chart_report"
    assert record["away_qb_projection_as_of"] == AS_OF
    assert record["away_qb_nflverse_id"] is pd.NA
    assert record["away_qb_carried_forward_id"] == "a1"
    assert bool(record["away_qb_differs_from_carry_forward"]) is True
    assert record["home_qb_id"] == "b2"
    assert record["home_qb_source"] == "nflverse_schedule"
    assert record["home_qb_projection_as_of"] is pd.NA
    assert record["home_qb_nflverse_id"] == "b2"
    assert record["home_qb_carried_forward_id"] == "b1"
    assert bool(record["home_qb_differs_from_carry_forward"]) is True


def test_carry_forward_source_and_no_difference() -> None:
    sidecar = _sidecar().iloc[[0, 1]].copy()
    sidecar.loc[1, ["AWAY_QB_ID_PROJECTED", "AWAY_QB_PROJECTION_SOURCE"]] = pd.NA
    sidecar.loc[1, "HOME_QB_ID"] = pd.NA

    record = attach_starting_qb_assumptions(_games("2026_02_AAA_BBB"), sidecar).iloc[0]

    assert record["away_qb_source"] == "carried_forward"
    assert record["away_qb_id"] == "a1"
    assert bool(record["away_qb_differs_from_carry_forward"]) is False
    assert record["home_qb_source"] == "carried_forward"
    assert record["away_qb_name"] is pd.NA


def test_the_record_uses_the_same_ids_as_the_feature_resolver() -> None:
    from gridiron_edge.features.team.qb_profile import resolve_starting_qb_history

    sidecar = _sidecar()
    resolved = resolve_starting_qb_history(sidecar).set_index(["GAME_ID", "SIDE"])
    record = attach_starting_qb_assumptions(_games("2026_02_AAA_BBB"), sidecar).iloc[0]

    assert record["away_qb_id"] == resolved.loc[("2026_02_AAA_BBB", "AWAY"), "RESOLVED_QB_ID"]
    assert record["home_qb_id"] == resolved.loc[("2026_02_AAA_BBB", "HOME"), "RESOLVED_QB_ID"]


def test_unresolvable_side_is_recorded_as_a_status() -> None:
    record = attach_starting_qb_assumptions(_games("2026_02_CCC_DDD"), _sidecar()).iloc[0]

    assert record["starting_qb_status"] == "starter_unresolved"
    assert record["away_qb_id"] is pd.NA
    assert record["home_qb_source"] is pd.NA


def test_game_missing_from_sidecar_and_missing_sidecar_are_statuses() -> None:
    absent = attach_starting_qb_assumptions(_games("2026_09_XXX_YYY"), _sidecar())
    none = attach_starting_qb_assumptions(_games("g"), None)

    assert absent["starting_qb_status"].tolist() == ["starter_unresolved"]
    assert none["starting_qb_status"].tolist() == ["sidecar_unavailable"]
    assert set(STARTING_QB_COLUMNS) <= set(none.columns)


def test_validation_rejects_inconsistent_records() -> None:
    good = attach_starting_qb_assumptions(_games("2026_02_AAA_BBB"), _sidecar())
    validate_starting_qb_record(good)

    unsourced = good.assign(away_qb_source="hunch")
    with pytest.raises(ValueError, match="unsupported away source"):
        validate_starting_qb_record(unsourced)

    no_time = good.assign(away_qb_projection_as_of=pd.NA)
    with pytest.raises(ValueError, match="projection time"):
        validate_starting_qb_record(no_time)

    empty_available = good.assign(home_qb_id=pd.NA)
    with pytest.raises(ValueError, match=r"record without a starter id|no home starter id"):
        validate_starting_qb_record(empty_available)

    with pytest.raises(ValueError, match="unsupported status"):
        validate_starting_qb_record(good.assign(starting_qb_status="maybe"))


def test_partial_record_is_rejected() -> None:
    partial = _product().assign(starting_qb_status="available")
    with pytest.raises(ValueError, match="partial starting-QB record"):
        has_starting_qb_columns(partial)


def test_recorded_product_is_stored_as_the_current_schema_and_round_trips(tmp_path: Path) -> None:
    product = attach_starting_qb_assumptions(
        _product().assign(game_id=["2026_02_AAA_BBB", "2026_09_XXX_YYY"]),
        _sidecar(),
        {"a2": "Ann Two"},
    )
    identity = _identity()

    write_weekly_product(product, identity=identity, repo=tmp_path)
    loaded = load_weekly_product(identity.product_id, repo=tmp_path)

    assert set(loaded["product_schema_version"]) == {WEEKLY_PRODUCT_SCHEMA_VERSION}
    assert loaded["away_qb_id"].tolist()[0] == "a2"
    assert loaded["starting_qb_status"].tolist() == ["available", "starter_unresolved"]
    assert loaded["away_qb_projection_as_of"].tolist()[0] == AS_OF


def test_product_without_a_record_stays_loadable_as_the_earlier_version(tmp_path: Path) -> None:
    identity: WeeklyProductIdentity = _identity()

    write_weekly_product(_product(), identity=identity, repo=tmp_path)
    loaded = load_weekly_product(identity.product_id, repo=tmp_path)

    assert set(loaded["product_schema_version"]) == {1}
    assert not has_starting_qb_columns(loaded)


def test_artifact_whose_columns_disagree_with_its_version_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    write_weekly_product(
        attach_starting_qb_assumptions(_product(), None), identity=identity, repo=tmp_path
    )
    artifact = weekly_product_artifact_path(identity.product_id, repo=tmp_path)
    stored = pd.read_parquet(artifact)
    stored["product_schema_version"] = 1
    stored.to_parquet(artifact, index=False)

    with pytest.raises(ValueError, match="starting-QB record disagrees"):
        load_weekly_product(identity.product_id, repo=tmp_path)


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    write_weekly_product(_product(), identity=identity, repo=tmp_path)
    artifact = weekly_product_artifact_path(identity.product_id, repo=tmp_path)
    stored = pd.read_parquet(artifact)
    stored["product_schema_version"] = 3
    stored.to_parquet(artifact, index=False)

    with pytest.raises(ValueError, match="schema mismatch"):
        load_weekly_product(identity.product_id, repo=tmp_path)
