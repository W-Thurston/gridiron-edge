"""Tests for weekly readiness domain contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from gridiron_edge.evaluation.weekly_readiness import (
    WeeklyReadiness,
    WeeklyReadinessBlocker,
    evaluate_weekly_readiness,
)


def _readiness(
    **overrides: object,
) -> WeeklyReadiness:
    """Create one complete weekly readiness result."""
    values: dict[str, object] = {
        "season": "2026-2027",
        "week": 1,
        "scheduled_game_count": 16,
        "selected_win_prediction_count": 16,
        "spread_value_count": 16,
        "total_prediction_count": 16,
        "projected_score_count": 16,
        "complete_provenance_count": 16,
        "market_game_count": 16,
        "prediction_market_match_count": 16,
        "eligible_market_count": 48,
        "positive_edge_count": 4,
        "prediction_generated_at": datetime(
            2026,
            9,
            1,
            12,
            tzinfo=UTC,
        ),
        "market_fetched_at": datetime(
            2026,
            9,
            1,
            13,
            tzinfo=UTC,
        ),
        "market_source": "draftkings",
        "blockers": (),
    }
    values.update(overrides)

    return WeeklyReadiness(**values)  # type: ignore[arg-type]


def _schedule(
    count: int = 2,
) -> pd.DataFrame:
    """Create canonical upcoming schedule rows."""
    return pd.DataFrame(
        {
            "YEAR": ["2026-2027"] * count,
            "WEEK_NUM": [1] * count,
            "GAME_ID": [f"2026_01_GAME_{index}" for index in range(count)],
        }
    )


def _predictions(
    count: int = 2,
) -> pd.DataFrame:
    """Create complete canonical weekly prediction rows."""
    generated_at = datetime(
        2026,
        9,
        1,
        12,
        tzinfo=UTC,
    )

    return pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(count)],
            "run_id": ["run-1"] * count,
            "generated_at": [generated_at] * count,
            "season": ["2026-2027"] * count,
            "week": [1] * count,
            "game_id": [f"2026_01_GAME_{index}" for index in range(count)],
            "model_name": ["win_prob"] * count,
            "model_type": ["elo"] * count,
            "home_win_prob": [0.55] * count,
            "model_spread": [-1.5] * count,
            "model_total": [44.5] * count,
            "projected_home_score": [23.0] * count,
            "projected_away_score": [21.5] * count,
        }
    )


def _markets(
    count: int = 2,
) -> pd.DataFrame:
    """Create complete long-format markets."""
    fetched_at = datetime(
        2026,
        9,
        1,
        13,
        tzinfo=UTC,
    )
    rows: list[dict[str, object]] = []

    for index in range(count):
        game_id = f"2026_01_GAME_{index}"
        base = {
            "fetched_at": fetched_at,
            "sportsbook": "draftkings",
            "season": "2026-2027",
            "week": 1,
            "game_id": game_id,
        }
        rows.extend(
            [
                {
                    **base,
                    "market": "moneyline",
                    "side": "home",
                    "odds": -110.0,
                    "line": None,
                },
                {
                    **base,
                    "market": "moneyline",
                    "side": "away",
                    "odds": -110.0,
                    "line": None,
                },
                {
                    **base,
                    "market": "spread",
                    "side": "home",
                    "odds": -110.0,
                    "line": -1.5,
                },
                {
                    **base,
                    "market": "spread",
                    "side": "away",
                    "odds": -110.0,
                    "line": 1.5,
                },
                {
                    **base,
                    "market": "total",
                    "side": "over",
                    "odds": -110.0,
                    "line": 44.5,
                },
                {
                    **base,
                    "market": "total",
                    "side": "under",
                    "odds": -110.0,
                    "line": 44.5,
                },
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "fetched_at",
            "sportsbook",
            "season",
            "week",
            "game_id",
            "market",
            "side",
            "odds",
            "line",
        ],
    )


def test_complete_result_is_ready() -> None:
    result = _readiness()

    assert result.ready
    assert result.has_positive_edges


def test_zero_positive_edges_is_valid_ready_result() -> None:
    result = _readiness(
        positive_edge_count=0,
    )

    assert result.ready
    assert not result.has_positive_edges
    assert result.blockers == ()


def test_blocker_makes_result_not_ready() -> None:
    result = _readiness(
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        market_fetched_at=None,
        market_source=None,
        blockers=(WeeklyReadinessBlocker.MISSING_MARKET_DATA,),
    )

    assert not result.ready
    assert result.blockers == (WeeklyReadinessBlocker.MISSING_MARKET_DATA,)


def test_partial_prediction_coverage_is_representable() -> None:
    result = _readiness(
        selected_win_prediction_count=15,
        blockers=(WeeklyReadinessBlocker.PARTIAL_WIN_PREDICTION_COVERAGE,),
    )

    assert result.scheduled_game_count == 16
    assert result.selected_win_prediction_count == 15
    assert not result.ready


def test_no_predictions_is_distinct_from_no_market_data() -> None:
    no_predictions = _readiness(
        selected_win_prediction_count=0,
        spread_value_count=0,
        total_prediction_count=0,
        projected_score_count=0,
        complete_provenance_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        prediction_generated_at=None,
        blockers=(WeeklyReadinessBlocker.MISSING_WIN_PREDICTIONS,),
    )
    no_market = _readiness(
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        market_fetched_at=None,
        market_source=None,
        blockers=(WeeklyReadinessBlocker.MISSING_MARKET_DATA,),
    )

    assert no_predictions.blockers != no_market.blockers


def test_zero_matches_is_distinct_from_incomplete_markets() -> None:
    zero_matches = _readiness(
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        blockers=(WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES,),
    )
    incomplete = _readiness(
        eligible_market_count=0,
        positive_edge_count=0,
        blockers=(WeeklyReadinessBlocker.INCOMPLETE_MARKETS,),
    )

    assert zero_matches.blockers != incomplete.blockers


@pytest.mark.parametrize(
    "field_name",
    [
        "scheduled_game_count",
        "selected_win_prediction_count",
        "spread_value_count",
        "total_prediction_count",
        "projected_score_count",
        "complete_provenance_count",
        "market_game_count",
        "prediction_market_match_count",
        "eligible_market_count",
        "positive_edge_count",
    ],
)
def test_rejects_negative_counts(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must not be negative",
    ):
        _readiness(
            **{field_name: -1},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "selected_win_prediction_count",
        "spread_value_count",
        "total_prediction_count",
        "projected_score_count",
        "complete_provenance_count",
        "market_game_count",
        "prediction_market_match_count",
    ],
)
def test_game_coverage_cannot_exceed_schedule(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(f"{field_name} must not exceed scheduled_game_count"),
    ):
        _readiness(
            **{field_name: 17},
        )


def test_eligible_markets_may_exceed_game_count() -> None:
    result = _readiness(
        eligible_market_count=96,
        positive_edge_count=10,
    )

    assert result.eligible_market_count == 96


def test_positive_edges_cannot_exceed_eligible_markets() -> None:
    with pytest.raises(
        ValueError,
        match=("positive_edge_count must not exceed eligible_market_count"),
    ):
        _readiness(
            eligible_market_count=2,
            positive_edge_count=3,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prediction_generated_at",
        "market_fetched_at",
    ],
)
def test_artifact_timestamp_requires_timezone(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be timezone-aware UTC",
    ):
        _readiness(
            **{
                field_name: datetime(
                    2026,
                    9,
                    1,
                    12,
                )
            },
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prediction_generated_at",
        "market_fetched_at",
    ],
)
def test_artifact_timestamp_requires_utc(
    field_name: str,
) -> None:
    mountain_time = timezone(timedelta(hours=-6))

    with pytest.raises(
        ValueError,
        match=f"{field_name} must use UTC",
    ):
        _readiness(
            **{
                field_name: datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=mountain_time,
                )
            },
        )


def test_nullable_provenance_is_allowed() -> None:
    result = _readiness(
        prediction_generated_at=None,
        market_fetched_at=None,
        market_source=None,
    )

    assert result.prediction_generated_at is None
    assert result.market_fetched_at is None
    assert result.market_source is None


def test_rejects_empty_market_source() -> None:
    with pytest.raises(
        ValueError,
        match="market_source must not be empty",
    ):
        _readiness(
            market_source=" ",
        )


def test_rejects_duplicate_blockers() -> None:
    blocker = WeeklyReadinessBlocker.MISSING_MARKET_DATA

    with pytest.raises(
        ValueError,
        match="blockers must not contain duplicate values",
    ):
        _readiness(
            blockers=(
                blocker,
                blocker,
            ),
        )


def test_rejects_invalid_scope() -> None:
    with pytest.raises(
        ValueError,
        match="season must not be empty",
    ):
        _readiness(
            season=" ",
        )

    with pytest.raises(
        ValueError,
        match="week must be at least 1",
    ):
        _readiness(
            week=0,
        )


def test_result_is_immutable() -> None:
    result = _readiness()

    with pytest.raises(FrozenInstanceError):
        result.week = 2  # type: ignore[misc]


class TestEvaluateWeeklyReadiness:
    """Tests for pure weekly readiness calculation."""

    def test_complete_week_counts_all_inputs(self) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=_markets(),
            edges=pd.DataFrame(
                {
                    "ev": [
                        0.05,
                        -0.01,
                        0.0,
                    ]
                }
            ),
        )

        assert result.ready
        assert result.scheduled_game_count == 2
        assert result.selected_win_prediction_count == 2
        assert result.spread_value_count == 2
        assert result.total_prediction_count == 2
        assert result.projected_score_count == 2
        assert result.complete_provenance_count == 2
        assert result.market_game_count == 2
        assert result.prediction_market_match_count == 2
        assert result.eligible_market_count == 6
        assert result.positive_edge_count == 1
        assert result.market_source == "draftkings"

    def test_sixteen_scheduled_and_fifteen_predicted(
        self,
    ) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(16),
            predictions=_predictions(15),
            markets=_markets(16),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.scheduled_game_count == 16
        assert result.selected_win_prediction_count == 15
        assert WeeklyReadinessBlocker.PARTIAL_WIN_PREDICTION_COVERAGE in result.blockers

    def test_no_predictions_is_distinct_from_no_market_data(
        self,
    ) -> None:
        no_predictions = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(0),
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )
        no_markets = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=_markets(0),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert WeeklyReadinessBlocker.MISSING_WIN_PREDICTIONS in no_predictions.blockers
        assert WeeklyReadinessBlocker.MISSING_MARKET_DATA in no_markets.blockers

    def test_zero_matches_is_distinct_from_incomplete_markets(
        self,
    ) -> None:
        schedule = pd.DataFrame(
            {
                "YEAR": [
                    "2026-2027",
                    "2026-2027",
                ],
                "WEEK_NUM": [1, 1],
                "GAME_ID": [
                    "2026_01_PREDICTION_GAME",
                    "2026_01_MARKET_GAME",
                ],
            }
        )

        predictions = _predictions(1)
        predictions["game_id"] = "2026_01_PREDICTION_GAME"

        unmatched_markets = _markets(1)
        unmatched_markets["game_id"] = "2026_01_MARKET_GAME"

        zero_matches = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=schedule,
            predictions=predictions,
            markets=unmatched_markets,
            edges=pd.DataFrame(columns=["ev"]),
        )

        incomplete_markets = _markets()
        incomplete_markets = incomplete_markets.loc[
            (incomplete_markets["market"] == "moneyline") & (incomplete_markets["side"] == "home"),
            :,
        ].reset_index(drop=True)

        incomplete_result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=incomplete_markets,
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES in zero_matches.blockers
        assert WeeklyReadinessBlocker.MISSING_MARKET_DATA not in zero_matches.blockers
        assert WeeklyReadinessBlocker.INCOMPLETE_MARKETS not in zero_matches.blockers

        assert incomplete_result.prediction_market_match_count == 2
        assert incomplete_result.eligible_market_count == 0
        assert WeeklyReadinessBlocker.INCOMPLETE_MARKETS in incomplete_result.blockers
        assert (
            WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES not in incomplete_result.blockers
        )

    def test_no_positive_edges_is_ready_when_inputs_complete(
        self,
    ) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=_markets(),
            edges=pd.DataFrame(
                {
                    "ev": [
                        0.0,
                        -0.02,
                    ]
                }
            ),
        )

        assert result.ready
        assert result.positive_edge_count == 0
        assert not result.has_positive_edges

    def test_partial_market_coverage_is_quantitative(
        self,
    ) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(2),
            predictions=_predictions(2),
            markets=_markets(1),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.market_game_count == 1
        assert result.prediction_market_match_count == 1
        assert WeeklyReadinessBlocker.PARTIAL_MARKET_COVERAGE in result.blockers

    def test_market_provenance_is_preserved(self) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.market_source == "draftkings"
        assert result.market_fetched_at == datetime(
            2026,
            9,
            1,
            13,
            tzinfo=UTC,
        )

    def test_mixed_market_sources_are_ambiguous(
        self,
    ) -> None:
        markets = _markets()
        markets.loc[
            markets.index[0],
            "sportsbook",
        ] = "other_source"

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=markets,
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.market_source is None
        assert WeeklyReadinessBlocker.AMBIGUOUS_MARKET_PROVENANCE in result.blockers

    def test_does_not_mutate_inputs(self) -> None:
        schedule = _schedule()
        predictions = _predictions()
        markets = _markets()
        edges = pd.DataFrame({"ev": [0.05]})

        originals = [
            frame.copy(deep=True)
            for frame in (
                schedule,
                predictions,
                markets,
                edges,
            )
        ]

        evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=schedule,
            predictions=predictions,
            markets=markets,
            edges=edges,
        )

        for frame, original in zip(
            (
                schedule,
                predictions,
                markets,
                edges,
            ),
            originals,
            strict=True,
        ):
            pd.testing.assert_frame_equal(
                frame,
                original,
            )

    def test_missing_schedule_is_explicit(self) -> None:
        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(0),
            predictions=_predictions(),
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.scheduled_game_count == 0
        assert result.blockers == (WeeklyReadinessBlocker.MISSING_SCHEDULE,)

    @pytest.mark.parametrize(
        ("column", "expected_blocker"),
        [
            (
                "model_spread",
                WeeklyReadinessBlocker.PARTIAL_SPREAD_COVERAGE,
            ),
            (
                "model_total",
                WeeklyReadinessBlocker.PARTIAL_TOTAL_PREDICTION_COVERAGE,
            ),
            (
                "projected_home_score",
                WeeklyReadinessBlocker.PARTIAL_PROJECTED_SCORE_COVERAGE,
            ),
            (
                "model_name",
                WeeklyReadinessBlocker.PARTIAL_MODEL_PROVENANCE,
            ),
        ],
    )
    def test_partial_component_coverage_is_reported(
        self,
        column: str,
        expected_blocker: WeeklyReadinessBlocker,
    ) -> None:
        predictions = _predictions()
        predictions.loc[
            predictions.index[0],
            column,
        ] = None

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=predictions,
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert expected_blocker in result.blockers

    def test_missing_prediction_artifact_provenance_is_explicit(
        self,
    ) -> None:
        predictions = _predictions().drop(columns=["generated_at"])

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=predictions,
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.prediction_generated_at is None
        assert WeeklyReadinessBlocker.MISSING_PREDICTION_PROVENANCE in result.blockers
        assert WeeklyReadinessBlocker.MISSING_MODEL_PROVENANCE in result.blockers

    def test_multiple_prediction_timestamps_are_not_collapsed(
        self,
    ) -> None:
        predictions = _predictions()
        predictions.loc[
            predictions.index[1],
            "generated_at",
        ] = datetime(
            2026,
            9,
            1,
            12,
            30,
            tzinfo=UTC,
        )

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=predictions,
            markets=_markets(),
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.prediction_generated_at is None
        assert WeeklyReadinessBlocker.MISSING_PREDICTION_PROVENANCE in result.blockers

    def test_missing_market_provenance_is_explicit(
        self,
    ) -> None:
        markets = _markets().drop(
            columns=[
                "fetched_at",
                "sportsbook",
            ]
        )

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=markets,
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.market_fetched_at is None
        assert result.market_source is None
        assert WeeklyReadinessBlocker.MISSING_MARKET_PROVENANCE in result.blockers

    def test_mixed_market_timestamps_are_ambiguous(
        self,
    ) -> None:
        markets = _markets()
        markets.loc[
            markets.index[0],
            "fetched_at",
        ] = datetime(
            2026,
            9,
            1,
            13,
            30,
            tzinfo=UTC,
        )

        result = evaluate_weekly_readiness(
            season="2026-2027",
            week=1,
            schedule=_schedule(),
            predictions=_predictions(),
            markets=markets,
            edges=pd.DataFrame(columns=["ev"]),
        )

        assert result.market_fetched_at is None
        assert WeeklyReadinessBlocker.AMBIGUOUS_MARKET_PROVENANCE in result.blockers

    def test_rejects_duplicate_schedule_game_ids(
        self,
    ) -> None:
        schedule = pd.concat(
            [
                _schedule(1),
                _schedule(1),
            ],
            ignore_index=True,
        )

        with pytest.raises(
            ValueError,
            match=("Schedule contains duplicate game IDs: 2026_01_GAME_0"),
        ):
            evaluate_weekly_readiness(
                season="2026-2027",
                week=1,
                schedule=schedule,
                predictions=_predictions(1),
                markets=_markets(1),
                edges=pd.DataFrame(columns=["ev"]),
            )

    def test_rejects_duplicate_prediction_game_ids(
        self,
    ) -> None:
        predictions = pd.concat(
            [
                _predictions(1),
                _predictions(1),
            ],
            ignore_index=True,
        )

        with pytest.raises(
            ValueError,
            match=("Predictions contain duplicate game IDs: 2026_01_GAME_0"),
        ):
            evaluate_weekly_readiness(
                season="2026-2027",
                week=1,
                schedule=_schedule(1),
                predictions=predictions,
                markets=_markets(1),
                edges=pd.DataFrame(columns=["ev"]),
            )

    @pytest.mark.parametrize(
        ("input_name", "missing_column", "message"),
        [
            (
                "schedule",
                "GAME_ID",
                "Schedule is missing required columns: GAME_ID",
            ),
            (
                "predictions",
                "game_id",
                "Predictions is missing required columns: game_id",
            ),
            (
                "markets",
                "market",
                "Markets is missing required columns: market",
            ),
        ],
    )
    def test_rejects_missing_input_columns(
        self,
        input_name: str,
        missing_column: str,
        message: str,
    ) -> None:
        inputs = {
            "schedule": _schedule(),
            "predictions": _predictions(),
            "markets": _markets(),
        }
        inputs[input_name] = inputs[input_name].drop(columns=[missing_column])

        with pytest.raises(
            ValueError,
            match=message,
        ):
            evaluate_weekly_readiness(
                season="2026-2027",
                week=1,
                schedule=inputs["schedule"],
                predictions=inputs["predictions"],
                markets=inputs["markets"],
                edges=pd.DataFrame(columns=["ev"]),
            )

    def test_non_empty_edges_require_ev_column(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="Edges is missing required columns: ev",
        ):
            evaluate_weekly_readiness(
                season="2026-2027",
                week=1,
                schedule=_schedule(),
                predictions=_predictions(),
                markets=_markets(),
                edges=pd.DataFrame(
                    {
                        "game_id": [
                            "2026_01_GAME_0",
                        ]
                    }
                ),
            )
