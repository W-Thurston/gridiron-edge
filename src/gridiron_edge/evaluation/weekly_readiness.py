"""Domain contracts and pure evaluation for weekly game-prediction readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

import pandas as pd
from pandas import DataFrame


class WeeklyReadinessBlocker(StrEnum):
    """Machine-readable reason a weekly product is not fully ready."""

    MISSING_SCHEDULE = "missing_schedule"
    MISSING_WIN_PREDICTIONS = "missing_win_predictions"
    PARTIAL_WIN_PREDICTION_COVERAGE = "partial_win_prediction_coverage"
    MISSING_SPREAD_VALUES = "missing_spread_values"
    PARTIAL_SPREAD_COVERAGE = "partial_spread_coverage"
    MISSING_TOTAL_PREDICTIONS = "missing_total_predictions"
    PARTIAL_TOTAL_PREDICTION_COVERAGE = "partial_total_prediction_coverage"
    MISSING_PROJECTED_SCORES = "missing_projected_scores"
    PARTIAL_PROJECTED_SCORE_COVERAGE = "partial_projected_score_coverage"
    MISSING_MODEL_PROVENANCE = "missing_model_provenance"
    PARTIAL_MODEL_PROVENANCE = "partial_model_provenance"
    MISSING_MARKET_DATA = "missing_market_data"
    PARTIAL_MARKET_COVERAGE = "partial_market_coverage"
    ZERO_PREDICTION_MARKET_MATCHES = "zero_prediction_market_matches"
    INCOMPLETE_MARKETS = "incomplete_markets"
    MISSING_PREDICTION_PROVENANCE = "missing_prediction_provenance"
    MISSING_MARKET_PROVENANCE = "missing_market_provenance"
    AMBIGUOUS_MARKET_PROVENANCE = "ambiguous_market_provenance"


_GAME_COVERAGE_FIELDS: Final[tuple[str, ...]] = (
    "selected_win_prediction_count",
    "spread_value_count",
    "total_prediction_count",
    "projected_score_count",
    "complete_provenance_count",
    "market_game_count",
    "prediction_market_match_count",
)

_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "YEAR",
    "WEEK_NUM",
    "GAME_ID",
)

_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
)

_MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "market",
    "side",
    "odds",
    "line",
)

_PREDICTION_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "run_id",
    "model_name",
    "model_type",
    "generated_at",
)


@dataclass(frozen=True)
class WeeklyReadiness:
    """Quantitative readiness state for one season and week."""

    season: str
    week: int

    scheduled_game_count: int
    selected_win_prediction_count: int
    spread_value_count: int
    total_prediction_count: int
    projected_score_count: int
    complete_provenance_count: int

    market_game_count: int
    prediction_market_match_count: int
    eligible_market_count: int
    positive_edge_count: int

    prediction_generated_at: datetime | None = None
    market_fetched_at: datetime | None = None
    market_source: str | None = None

    blockers: tuple[WeeklyReadinessBlocker, ...] = ()

    def __post_init__(self) -> None:
        """Validate weekly readiness invariants."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")

        if self.week < 1:
            raise ValueError("week must be at least 1.")

        count_fields = (
            "scheduled_game_count",
            *_GAME_COVERAGE_FIELDS,
            "eligible_market_count",
            "positive_edge_count",
        )

        for field_name in count_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")

        for field_name in _GAME_COVERAGE_FIELDS:
            value = getattr(self, field_name)
            if value > self.scheduled_game_count:
                raise ValueError(f"{field_name} must not exceed scheduled_game_count.")

        if self.positive_edge_count > self.eligible_market_count:
            raise ValueError("positive_edge_count must not exceed eligible_market_count.")

        self._validate_utc_timestamp(
            "prediction_generated_at",
            self.prediction_generated_at,
        )
        self._validate_utc_timestamp(
            "market_fetched_at",
            self.market_fetched_at,
        )

        if self.market_source is not None and not self.market_source.strip():
            raise ValueError("market_source must not be empty when provided.")

        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("blockers must not contain duplicate values.")

    @staticmethod
    def _validate_utc_timestamp(
        field_name: str,
        value: datetime | None,
    ) -> None:
        """Validate an optional timezone-aware UTC timestamp."""
        if value is None:
            return

        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware UTC.")

        if value.utcoffset() != timedelta(0):
            raise ValueError(f"{field_name} must use UTC.")

    @property
    def ready(self) -> bool:
        """Return whether no operational blockers are present."""
        return not self.blockers

    @property
    def has_positive_edges(self) -> bool:
        """Return whether analysis produced any positive edges."""
        return self.positive_edge_count > 0


def _require_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Raise when a readiness input lacks required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _scope_schedule(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Return distinct scheduled games in the requested scope."""
    _require_columns(schedule, _SCHEDULE_COLUMNS, label="Schedule")

    scoped = schedule.loc[
        (schedule["YEAR"].astype(str) == season) & (schedule["WEEK_NUM"] == week),
        :,
    ].copy()

    if scoped["GAME_ID"].isna().any():
        raise ValueError("Schedule game IDs must not contain nulls.")

    empty_game_ids = scoped["GAME_ID"].astype(str).str.strip().eq("")
    if empty_game_ids.any():
        raise ValueError("Schedule game IDs must not contain empty values.")

    if scoped["GAME_ID"].duplicated().any():
        duplicate_ids = sorted(
            scoped.loc[
                scoped["GAME_ID"].duplicated(keep=False),
                "GAME_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError("Schedule contains duplicate game IDs: " + ", ".join(duplicate_ids))

    return scoped.reset_index(drop=True)


def _scope_predictions(
    predictions: DataFrame,
    *,
    season: str,
    week: int,
    scheduled_game_ids: set[str],
) -> DataFrame:
    """Return canonical predictions for scheduled games in scope."""
    _require_columns(predictions, _PREDICTION_COLUMNS, label="Predictions")

    scoped = predictions.loc[
        (predictions["season"].astype(str) == season)
        & (predictions["week"] == week)
        & predictions["game_id"].astype(str).isin(scheduled_game_ids),
        :,
    ].copy()

    if scoped["game_id"].duplicated().any():
        duplicate_ids = sorted(
            scoped.loc[
                scoped["game_id"].duplicated(keep=False),
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError("Predictions contain duplicate game IDs: " + ", ".join(duplicate_ids))

    return scoped.reset_index(drop=True)


def _scope_markets(
    markets: DataFrame,
    *,
    season: str,
    week: int,
    scheduled_game_ids: set[str],
) -> DataFrame:
    """Return long-format market rows for scheduled games in scope."""
    _require_columns(markets, _MARKET_COLUMNS, label="Markets")

    return (
        markets.loc[
            (markets["season"].astype(str) == season)
            & (markets["week"] == week)
            & markets["game_id"].astype(str).isin(scheduled_game_ids),
            :,
        ]
        .copy()
        .reset_index(drop=True)
    )


def _count_non_null_games(
    predictions: DataFrame,
    columns: tuple[str, ...],
) -> int:
    """Count prediction games where all requested values exist."""
    if predictions.empty:
        return 0

    if any(column not in predictions.columns for column in columns):
        return 0

    complete = predictions.loc[:, list(columns)].notna().all(axis=1)
    return int(predictions.loc[complete, "game_id"].nunique())


def _count_complete_provenance(predictions: DataFrame) -> int:
    """Count games with complete immutable model provenance."""
    if predictions.empty:
        return 0

    if any(column not in predictions.columns for column in _PREDICTION_PROVENANCE_COLUMNS):
        return 0

    complete = (
        predictions.loc[
            :,
            list(_PREDICTION_PROVENANCE_COLUMNS),
        ]
        .notna()
        .all(axis=1)
    )

    for column in ("event_id", "run_id", "model_name", "model_type"):
        complete &= predictions[column].astype(str).str.strip().ne("")

    return int(predictions.loc[complete, "game_id"].nunique())


def _market_side_present(
    game_markets: DataFrame,
    *,
    market: str,
    side: str,
    require_line: bool,
) -> bool:
    """Return whether one required market side is complete."""
    rows = game_markets.loc[
        (game_markets["market"] == market) & (game_markets["side"] == side),
        :,
    ]

    if rows.empty:
        return False

    complete = rows["odds"].notna()
    if require_line:
        complete &= rows["line"].notna()

    return bool(complete.any())


def _count_eligible_markets(
    predictions: DataFrame,
    markets: DataFrame,
) -> int:
    """Count complete, calculable game-market pairs."""
    if predictions.empty or markets.empty:
        return 0

    predictions_by_game = predictions.set_index("game_id", drop=False)
    eligible = 0

    for game_id, game_markets in markets.groupby("game_id", sort=False):
        if game_id not in predictions_by_game.index:
            continue

        prediction = predictions_by_game.loc[game_id]
        if isinstance(prediction, DataFrame):
            raise ValueError(f"Prediction game ID is not unique: {game_id}")

        moneyline_complete = (
            pd.notna(prediction.get("home_win_prob"))
            and _market_side_present(
                game_markets,
                market="moneyline",
                side="home",
                require_line=False,
            )
            and _market_side_present(
                game_markets,
                market="moneyline",
                side="away",
                require_line=False,
            )
        )
        if moneyline_complete:
            eligible += 1

        spread_complete = (
            pd.notna(prediction.get("model_spread"))
            and _market_side_present(
                game_markets,
                market="spread",
                side="home",
                require_line=True,
            )
            and _market_side_present(
                game_markets,
                market="spread",
                side="away",
                require_line=False,
            )
        )
        if spread_complete:
            eligible += 1

        total_complete = (
            pd.notna(prediction.get("model_total"))
            and _market_side_present(
                game_markets,
                market="total",
                side="over",
                require_line=True,
            )
            and _market_side_present(
                game_markets,
                market="total",
                side="under",
                require_line=False,
            )
        )
        if total_complete:
            eligible += 1

    return eligible


def _unique_utc_timestamp(
    frame: DataFrame,
    column: str,
) -> datetime | None:
    """Return one unique UTC timestamp, otherwise None."""
    if frame.empty or column not in frame.columns:
        return None

    # pyrefly: ignore [missing-attribute]
    values = pd.to_datetime(
        frame[column],
        utc=True,
        errors="coerce",
    ).dropna()
    unique = values.drop_duplicates()

    if len(unique) != 1:
        return None

    return unique.iloc[0].to_pydatetime()


def _market_source(markets: DataFrame) -> str | None:
    """Return one explicit market source, otherwise None."""
    if markets.empty or "sportsbook" not in markets.columns:
        return None

    sources = sorted(
        {str(value).strip() for value in markets["sportsbook"].dropna() if str(value).strip()}
    )

    return sources[0] if len(sources) == 1 else None


def _market_provenance_is_ambiguous(markets: DataFrame) -> bool:
    """Return whether market rows contain mixed provenance."""
    if markets.empty:
        return False

    if "sportsbook" in markets.columns:
        sources = {
            str(value).strip() for value in markets["sportsbook"].dropna() if str(value).strip()
        }
        if len(sources) > 1:
            return True

    if "fetched_at" in markets.columns:
        # pyrefly: ignore [missing-attribute]
        timestamps = pd.to_datetime(
            markets["fetched_at"],
            utc=True,
            errors="coerce",
        ).dropna()
        if timestamps.nunique() > 1:
            return True

    return False


def _coverage_blocker(
    *,
    count: int,
    scheduled_count: int,
    missing: WeeklyReadinessBlocker,
    partial: WeeklyReadinessBlocker,
) -> WeeklyReadinessBlocker | None:
    """Return the missing or partial blocker for one coverage count."""
    if scheduled_count == 0:
        return None
    if count == 0:
        return missing
    if count < scheduled_count:
        return partial
    return None


def _append_blocker(
    blockers: list[WeeklyReadinessBlocker],
    blocker: WeeklyReadinessBlocker | None,
) -> None:
    """Append one non-null blocker."""
    if blocker is not None:
        blockers.append(blocker)


def _prediction_coverage_blockers(
    *,
    scheduled_count: int,
    selected_win_count: int,
    spread_count: int,
    total_count: int,
    projected_score_count: int,
    complete_provenance_count: int,
) -> list[WeeklyReadinessBlocker]:
    """Derive schedule and prediction-component coverage blockers."""
    if scheduled_count == 0:
        return [
            WeeklyReadinessBlocker.MISSING_SCHEDULE,
        ]

    blockers: list[WeeklyReadinessBlocker] = []

    coverage_checks = (
        (
            selected_win_count,
            WeeklyReadinessBlocker.MISSING_WIN_PREDICTIONS,
            WeeklyReadinessBlocker.PARTIAL_WIN_PREDICTION_COVERAGE,
        ),
        (
            spread_count,
            WeeklyReadinessBlocker.MISSING_SPREAD_VALUES,
            WeeklyReadinessBlocker.PARTIAL_SPREAD_COVERAGE,
        ),
        (
            total_count,
            WeeklyReadinessBlocker.MISSING_TOTAL_PREDICTIONS,
            WeeklyReadinessBlocker.PARTIAL_TOTAL_PREDICTION_COVERAGE,
        ),
        (
            projected_score_count,
            WeeklyReadinessBlocker.MISSING_PROJECTED_SCORES,
            WeeklyReadinessBlocker.PARTIAL_PROJECTED_SCORE_COVERAGE,
        ),
        (
            complete_provenance_count,
            WeeklyReadinessBlocker.MISSING_MODEL_PROVENANCE,
            WeeklyReadinessBlocker.PARTIAL_MODEL_PROVENANCE,
        ),
    )

    for count, missing, partial in coverage_checks:
        _append_blocker(
            blockers,
            _coverage_blocker(
                count=count,
                scheduled_count=scheduled_count,
                missing=missing,
                partial=partial,
            ),
        )

    return blockers


def _market_coverage_blockers(
    *,
    scheduled_count: int,
    selected_win_count: int,
    market_game_count: int,
    prediction_market_match_count: int,
    eligible_market_count: int,
) -> list:
    """Derive market coverage, join, and completeness blockers."""
    blockers: list[WeeklyReadinessBlocker] = []

    if scheduled_count > 0:
        if market_game_count == 0:
            blockers.append(WeeklyReadinessBlocker.MISSING_MARKET_DATA)
        elif market_game_count < scheduled_count:
            blockers.append(WeeklyReadinessBlocker.PARTIAL_MARKET_COVERAGE)

    if selected_win_count > 0 and market_game_count > 0 and prediction_market_match_count == 0:
        blockers.append(WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES)

    if market_game_count > 0 and prediction_market_match_count > 0 and eligible_market_count == 0:
        blockers.append(WeeklyReadinessBlocker.INCOMPLETE_MARKETS)

    return blockers


def _artifact_provenance_blockers(
    *,
    predictions: DataFrame,
    markets: DataFrame,
    prediction_generated_at: datetime | None,
    market_fetched_at: datetime | None,
    market_source: str | None,
) -> list:
    """Derive prediction and market artifact-provenance blockers."""
    blockers: list[WeeklyReadinessBlocker] = []

    if not predictions.empty and prediction_generated_at is None:
        blockers.append(WeeklyReadinessBlocker.MISSING_PREDICTION_PROVENANCE)

    if markets.empty:
        return blockers

    if _market_provenance_is_ambiguous(markets):
        blockers.append(WeeklyReadinessBlocker.AMBIGUOUS_MARKET_PROVENANCE)
    elif market_fetched_at is None or market_source is None:
        blockers.append(WeeklyReadinessBlocker.MISSING_MARKET_PROVENANCE)

    return blockers


def evaluate_weekly_readiness(
    *,
    season: str,
    week: int,
    schedule: DataFrame,
    predictions: DataFrame,
    markets: DataFrame,
    edges: DataFrame,
) -> WeeklyReadiness:
    """Derive weekly readiness from supplied domain inputs.

    This function performs no file I/O, forecast selection, model inference,
    market ingestion, or edge calculation.
    """
    scoped_schedule = _scope_schedule(schedule, season=season, week=week)
    scheduled_game_ids = set(scoped_schedule["GAME_ID"].astype(str))
    scheduled_count = len(scheduled_game_ids)

    scoped_predictions = _scope_predictions(
        predictions,
        season=season,
        week=week,
        # pyrefly: ignore [bad-argument-type]
        scheduled_game_ids=scheduled_game_ids,
    )
    scoped_markets = _scope_markets(
        markets,
        season=season,
        week=week,
        # pyrefly: ignore [bad-argument-type]
        scheduled_game_ids=scheduled_game_ids,
    )

    prediction_game_ids = set(scoped_predictions["game_id"].astype(str))
    market_game_ids = set(scoped_markets["game_id"].astype(str))

    selected_win_count = _count_non_null_games(
        scoped_predictions,
        ("home_win_prob",),
    )
    spread_count = _count_non_null_games(
        scoped_predictions,
        ("model_spread",),
    )
    total_count = _count_non_null_games(
        scoped_predictions,
        ("model_total",),
    )
    projected_score_count = _count_non_null_games(
        scoped_predictions,
        ("projected_home_score", "projected_away_score"),
    )
    complete_provenance_count = _count_complete_provenance(scoped_predictions)

    prediction_market_match_count = len(prediction_game_ids.intersection(market_game_ids))
    eligible_market_count = _count_eligible_markets(
        scoped_predictions,
        scoped_markets,
    )

    positive_edge_count = 0
    if not edges.empty:
        _require_columns(
            edges,
            ("ev",),
            label="Edges",
        )
        positive_edge_count = int(
            # pyrefly: ignore [missing-attribute]
            (
                pd.to_numeric(
                    edges["ev"],
                    errors="coerce",
                )
                > 0.0
            ).sum()
        )

    prediction_generated_at = _unique_utc_timestamp(
        scoped_predictions,
        "generated_at",
    )
    market_fetched_at = _unique_utc_timestamp(
        scoped_markets,
        "fetched_at",
    )
    market_source = _market_source(
        scoped_markets,
    )

    market_game_count = len(market_game_ids)

    blockers = _prediction_coverage_blockers(
        scheduled_count=scheduled_count,
        selected_win_count=selected_win_count,
        spread_count=spread_count,
        total_count=total_count,
        projected_score_count=projected_score_count,
        complete_provenance_count=complete_provenance_count,
    )
    blockers.extend(
        _market_coverage_blockers(
            scheduled_count=scheduled_count,
            selected_win_count=selected_win_count,
            market_game_count=market_game_count,
            prediction_market_match_count=(prediction_market_match_count),
            eligible_market_count=eligible_market_count,
        )
    )
    blockers.extend(
        _artifact_provenance_blockers(
            predictions=scoped_predictions,
            markets=scoped_markets,
            prediction_generated_at=prediction_generated_at,
            market_fetched_at=market_fetched_at,
            market_source=market_source,
        )
    )

    return WeeklyReadiness(
        season=season,
        week=week,
        scheduled_game_count=scheduled_count,
        selected_win_prediction_count=selected_win_count,
        spread_value_count=spread_count,
        total_prediction_count=total_count,
        projected_score_count=projected_score_count,
        complete_provenance_count=complete_provenance_count,
        market_game_count=market_game_count,
        prediction_market_match_count=(prediction_market_match_count),
        eligible_market_count=eligible_market_count,
        positive_edge_count=positive_edge_count,
        prediction_generated_at=prediction_generated_at,
        market_fetched_at=market_fetched_at,
        market_source=market_source,
        blockers=tuple(blockers),
    )
