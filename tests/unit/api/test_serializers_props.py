# tests/unit/api/test_serializers_props.py

"""Tests for /props serializers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridiron_edge.api.meta import BlockedStatus
from gridiron_edge.api.schemas.props import (
    LineBlock,
    ProjectionBlock,
    PropDetail,
    PropList,
    PropSummary,
)
from gridiron_edge.api.serializers.props import (
    _build_line_block,
    _build_model_key,
    _build_projection_block,
    _build_prop_id,
    _none_if_nan,
    _season_int_to_str,
    serialize_prop_detail,
    serialize_prop_summary,
    serialize_props_list,
)


def _valid_row() -> dict:
    """A canonical valid archive row for reuse across tests."""
    return {
        "predicted_at": "2026-08-01T00:00:00+00:00",
        "is_backfilled": True,
        "season": 2026,
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "player_id": "00-0033873",
        "player_name": "P.Mahomes",
        "position": "QB",
        "team": "KC",
        "stat_type": "qb_pass_yards",
        "model_name": "qb_pass_yards",
        "model_type": "elasticnet",
        "predicted_mean": 265.0,
        "predicted_std": 45.0,
        "lo_90": 190.0,
        "hi_90": 340.0,
        "line": None,
        "p_over": float("nan"),
        "lean": float("nan"),
        "confidence_tier": float("nan"),
    }


class TestNoneIfNan:
    def test_none_returns_none(self) -> None:
        assert _none_if_nan(None) is None

    def test_nan_returns_none(self) -> None:
        assert _none_if_nan(float("nan")) is None
        assert _none_if_nan(np.nan) is None

    def test_value_preserved(self) -> None:
        assert _none_if_nan(265.0) == 265.0
        assert _none_if_nan("Over") == "Over"


class TestSeasonIntToStr:
    def test_int_converts(self) -> None:
        assert _season_int_to_str(2026) == "2026-2027"

    def test_float_converts(self) -> None:
        assert _season_int_to_str(2026.0) == "2026-2027"

    def test_none_returns_none(self) -> None:
        assert _season_int_to_str(None) is None

    def test_nan_returns_none(self) -> None:
        assert _season_int_to_str(float("nan")) is None

    def test_unparseable_returns_none(self) -> None:
        assert _season_int_to_str("not a year") is None


class TestBuildPropId:
    def test_full_composite(self) -> None:
        row = _valid_row()
        assert _build_prop_id(row) == "2026_01_KC_LAC__00-0033873__qb_pass_yards"


class TestBuildModelKey:
    def test_composite_string(self) -> None:
        row = _valid_row()
        assert _build_model_key(row) == "qb_pass_yards_elasticnet"


class TestBuildProjectionBlock:
    def test_full_row_builds_block(self) -> None:
        block = _build_projection_block(_valid_row())
        assert isinstance(block, ProjectionBlock)
        assert block.predicted_mean == 265.0
        assert block.hi_90 == 340.0

    def test_nan_fields_become_none(self) -> None:
        row = _valid_row()
        row["predicted_std"] = float("nan")
        block = _build_projection_block(row)
        assert block.predicted_std is None
        assert block.predicted_mean == 265.0  # unaffected


class TestBuildLineBlock:
    def test_all_null_returns_block_with_nulls(self) -> None:
        block = _build_line_block(_valid_row())
        assert isinstance(block, LineBlock)
        assert block.line is None
        assert block.p_over is None
        assert block.lean is None
        assert block.confidence_tier is None

    def test_populated_row(self) -> None:
        row = _valid_row()
        row.update(
            {
                "line": 273.5,
                "p_over": 0.45,
                "lean": "Under",
                "confidence_tier": "Low",
            }
        )
        block = _build_line_block(row)
        assert block.line == 273.5
        assert block.lean == "Under"


class TestSerializePropSummary:
    def test_full_row(self) -> None:
        summary = serialize_prop_summary(_valid_row())
        assert isinstance(summary, PropSummary)
        assert summary.prop_id == "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        assert summary.model_key == "qb_pass_yards_elasticnet"
        assert summary.season == "2026-2027"
        assert summary.week == 1
        assert summary.projection is not None
        assert summary.projection.predicted_mean == 265.0
        assert summary.line_context is not None
        assert summary.line_context.line is None

    def test_nan_scalar_fields_become_none(self) -> None:
        row = _valid_row()
        row["week"] = float("nan")
        row["season"] = float("nan")
        summary = serialize_prop_summary(row)
        assert summary.week is None
        assert summary.season is None


class TestSerializePropsList:
    def test_empty_dataframe(self) -> None:
        response = serialize_props_list(
            pd.DataFrame(),
            season="2026-2027",
            week=1,
            stat_type=None,
            position=None,
        )
        assert isinstance(response, PropList)
        assert response.items == []
        assert response.total == 0
        assert response.season == "2026-2027"

    def test_dataframe_of_rows_serializes_each(self) -> None:
        df = pd.DataFrame(
            [
                _valid_row(),
                {**_valid_row(), "player_id": "00-0035700", "player_name": "L.Jackson"},
            ]
        )
        response = serialize_props_list(
            df,
            season="2026-2027",
            week=1,
            stat_type="qb_pass_yards",
            position="QB",
        )
        assert len(response.items) == 2
        assert response.total == 2
        assert response.items[0].player_name == "P.Mahomes"
        assert response.items[1].player_name == "L.Jackson"

    def test_field_status_marks_line_context_fields(self) -> None:
        response = serialize_props_list(
            pd.DataFrame([_valid_row()]),
            season="2026-2027",
            week=1,
            stat_type=None,
            position=None,
        )
        assert response.response_meta is not None
        status = response.response_meta.field_status
        assert status["items.line_context.line"] == "pending"
        assert status["items.line_context.p_over"] == "pending"
        assert status["items.line_context.lean"] == "pending"
        assert status["items.line_context.confidence_tier"] == "pending"


class TestSerializePropDetail:
    def test_full_row(self) -> None:
        detail = serialize_prop_detail(_valid_row())
        assert isinstance(detail, PropDetail)
        assert detail.prop_id == "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        assert detail.model_key == "qb_pass_yards_elasticnet"
        assert detail.season == "2026-2027"
        assert detail.projection is not None
        assert detail.projection.predicted_mean == 265.0
        assert detail.line_context is not None
        assert detail.line_context.line is None

    def test_scaffolded_fields_null(self) -> None:
        detail = serialize_prop_detail(_valid_row())
        assert detail.historical_vs_opponent is None
        assert detail.situational_splits is None
        assert detail.recent_form is None
        assert detail.prop_reasoning is None
        assert detail.injury_status is None
        assert detail.multi_book_shopping is None

    def test_field_status_marks_line_context_pending(self) -> None:
        detail = serialize_prop_detail(_valid_row())
        assert detail.response_meta is not None
        status = detail.response_meta.field_status
        assert status["line_context.line"] == "pending"
        assert status["line_context.p_over"] == "pending"

    def test_field_status_marks_scaffolds_pending(self) -> None:
        detail = serialize_prop_detail(_valid_row())
        assert detail.response_meta is not None
        status = detail.response_meta.field_status
        assert status["historical_vs_opponent"] == "pending"
        assert status["situational_splits"] == "pending"
        assert status["recent_form"] == "pending"

    def test_field_status_marks_scaffolds_blocked(self) -> None:
        detail = serialize_prop_detail(_valid_row())
        assert detail.response_meta is not None
        status = detail.response_meta.field_status

        assert isinstance(status["prop_reasoning"], BlockedStatus)
        assert status["prop_reasoning"].blocker == "feature_attribution"

        assert isinstance(status["injury_status"], BlockedStatus)
        assert status["injury_status"].blocker == "injury_data_source"

        assert isinstance(status["multi_book_shopping"], BlockedStatus)
        assert status["multi_book_shopping"].blocker == "multi_book_ingest"


class TestSerializePropDetailSituationalSplits:
    def test_populates_situational_splits(self) -> None:
        from gridiron_edge.api.serializers.props import serialize_prop_detail

        row = _valid_row()  # Existing helper in this file
        splits = {
            "season": {"sample_size": 5, "mean_value": 260.0},
            "home": {"sample_size": 3, "mean_value": 293.3},
        }

        result = serialize_prop_detail(row, situational_splits=splits)

        assert result.situational_splits == splits

    def test_null_splits_leaves_pending_marker(self) -> None:
        from gridiron_edge.api.serializers.props import serialize_prop_detail

        row = _valid_row()
        result = serialize_prop_detail(row, situational_splits=None)

        assert result.situational_splits is None
        assert result.response_meta is not None
        assert result.response_meta.field_status.get("situational_splits") == "pending"

    def test_empty_dict_no_pending_marker(self) -> None:
        """Empty dict (artifact exists but no data for player) should not be pending."""
        from gridiron_edge.api.serializers.props import serialize_prop_detail

        row = _valid_row()
        result = serialize_prop_detail(row, situational_splits={})

        assert result.situational_splits == {}
        assert result.response_meta is not None
        assert "situational_splits" not in result.response_meta.field_status
