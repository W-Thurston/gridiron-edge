# tests/unit/api/test_schemas_props.py

"""Tests for /props response schemas (W8 Tier 2 Step 7b)."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.props import (
    LineBlock,
    ProjectionBlock,
    PropDetail,
    PropList,
    PropSummary,
)


class TestProjectionBlock:
    def test_full_projection(self) -> None:
        block = ProjectionBlock(
            predicted_mean=265.0,
            predicted_std=45.0,
            lo_90=190.0,
            hi_90=340.0,
        )
        assert block.predicted_mean == 265.0
        assert block.hi_90 == 340.0

    def test_all_fields_default_to_none(self) -> None:
        block = ProjectionBlock()
        assert block.predicted_mean is None
        assert block.predicted_std is None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ProjectionBlock(predicted_mean=265.0, mystery=1)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        block = ProjectionBlock(predicted_mean=265.0)
        with pytest.raises(ValidationError):
            block.predicted_mean = 300.0  # type: ignore[misc]


class TestLineBlock:
    def test_full_line(self) -> None:
        block = LineBlock(
            line=273.5,
            p_over=0.52,
            lean="Over",
            confidence_tier="Moderate",
        )
        assert block.line == 273.5
        assert block.lean == "Over"

    def test_all_fields_default_to_none(self) -> None:
        block = LineBlock()
        assert block.line is None
        assert block.p_over is None

    def test_frozen(self) -> None:
        block = LineBlock(line=273.5)
        with pytest.raises(ValidationError):
            block.line = 280.0  # type: ignore[misc]


class TestPropSummary:
    def test_minimum_shape(self) -> None:
        summary = PropSummary(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
        )
        assert summary.prop_id == "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        assert summary.projection is None
        assert summary.line_context is None

    def test_with_projection_and_line(self) -> None:
        summary = PropSummary(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            season="2026-2027",
            week=1,
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
            projection=ProjectionBlock(
                predicted_mean=265.0,
                predicted_std=45.0,
                lo_90=190.0,
                hi_90=340.0,
            ),
            line_context=LineBlock(
                line=273.5,
                p_over=0.45,
                lean="Under",
                confidence_tier="Low",
            ),
        )
        assert summary.projection is not None
        assert summary.projection.predicted_mean == 265.0
        assert summary.line_context is not None
        assert summary.line_context.lean == "Under"

    def test_rejects_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            PropSummary(prop_id="x")  # type: ignore[call-arg]


class TestPropList:
    def test_empty_list(self) -> None:
        response = PropList()
        assert response.items == []
        assert response.total is None
        assert response.season is None
        assert response.week is None
        assert response.stat_type is None
        assert response.position is None

    def test_with_summaries_and_filters(self) -> None:
        summary = PropSummary(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
        )
        response = PropList(
            items=[summary],
            season="2026-2027",
            week=1,
            stat_type="qb_pass_yards",
            position="QB",
        )
        assert len(response.items) == 1
        assert response.season == "2026-2027"
        assert response.stat_type == "qb_pass_yards"


class TestPropDetail:
    def test_minimum_shape(self) -> None:
        detail = PropDetail(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
        )
        assert detail.projection is None
        assert detail.line_context is None
        # All scaffolded fields default to None.
        assert detail.historical_vs_opponent is None
        assert detail.situational_splits is None
        assert detail.prop_reasoning is None
        assert detail.injury_status is None
        assert detail.recent_form is None
        assert detail.multi_book_shopping is None

    def test_with_all_populated_blocks(self) -> None:
        detail = PropDetail(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            season="2026-2027",
            week=1,
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
            projection=ProjectionBlock(
                predicted_mean=265.0,
                predicted_std=45.0,
                lo_90=190.0,
                hi_90=340.0,
            ),
            line_context=LineBlock(line=273.5),
        )
        assert detail.projection is not None
        assert detail.projection.predicted_mean == 265.0
        assert detail.line_context is not None
        assert detail.line_context.line == 273.5

    def test_scaffolded_fields_accept_shapes(self) -> None:
        detail = PropDetail(
            prop_id="2026_01_KC_LAC__00-0033873__qb_pass_yards",
            game_id="2026_01_KC_LAC",
            player_id="00-0033873",
            player_name="P.Mahomes",
            position="QB",
            team="KC",
            stat_type="qb_pass_yards",
            model_key="qb_pass_yards_elasticnet",
            historical_vs_opponent=[{"season": "2025-2026", "value": 240.0}],
            situational_splits={"home_avg": 275.0, "away_avg": 258.0},
            prop_reasoning={"top_features": ["opp_pass_def_epa"]},
            injury_status={"status": "questionable"},
            recent_form=[{"week": 17, "actual": 302}],
            multi_book_shopping={"draftkings": 273.5, "fanduel": 274.0},
        )
        assert detail.historical_vs_opponent == [{"season": "2025-2026", "value": 240.0}]
        assert detail.multi_book_shopping == {"draftkings": 273.5, "fanduel": 274.0}

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            PropDetail(
                prop_id="x",
                game_id="y",
                player_id="p",
                player_name="X",
                position="QB",
                team="KC",
                stat_type="qb_pass_yards",
                model_key="qb_pass_yards_elasticnet",
                mystery="oops",  # type: ignore[call-arg]
            )
