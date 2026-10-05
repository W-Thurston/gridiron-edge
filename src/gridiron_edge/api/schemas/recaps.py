# src/gridiron_edge/api/schemas/recaps.py

"""Schemas for /recaps: the selected weekly and season recaps (``DECISIONS.md`` D86).

Every value is serialized from the persisted recap. Each forecast family, market side, game row
and season week carries the recap's own status enum, so a missing value is always explained,
never dropped.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse
from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.weekly_recap import (
    Closer,
    ClosingLineStatus,
    ForecastMarketStatus,
    ForecastStatus,
    ProductRowStatus,
)
from gridiron_edge.evaluation.weekly_recap_season import SeasonWeekStatus
from gridiron_edge.evaluation.weekly_recap_starters import (
    ActualStarterSource,
    StarterRecordSource,
    StarterStatus,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RecapFamilyAvailability(_Frozen):
    """How many scheduled games carry an available forecast, family by family."""

    win: int
    spread: int
    total: int
    projected_score: int


class WeeklyRecapSummary(_Frozen):
    """One selected weekly recap, as listed by ``GET /recaps``."""

    season: str
    week: int
    recap_id: str
    built_at: datetime
    selected_at: datetime
    product_id: str
    forecast_role: ForecastRole | None = Field(
        description="The selected product's forecast role.",
    )
    row_forecast_roles: list[ForecastRole] = Field(
        description="Every distinct forecast role among the product's rows, sorted.",
    )
    scheduled_games: int
    in_product: int
    available: RecapFamilyAvailability


class WeeklyRecapList(BaseListResponse[WeeklyRecapSummary]):
    """Every selected weekly recap, ordered by season and week (never by build time)."""


class RecapConventionsResponse(_Frozen):
    """The conventions every value in the recap is stated in."""

    orientation: str
    closing_market: str
    market_at_forecast: str
    market_win_probability: str
    margin_std: float
    actual_starter: str


class RecapCountsResponse(_Frozen):
    """Row counts of the recap's game frame."""

    scheduled_games: int
    in_product: int
    not_in_product: int
    carried_rows: int
    outcome_ties: int


class RecapProbabilityMetrics(_Frozen):
    """Win-probability scores over ``count`` decided games."""

    count: int
    brier: float | None
    log_loss: float | None
    accuracy: float | None


class RecapErrorMetrics(_Frozen):
    """Point-forecast errors (prediction minus result) over ``count`` values."""

    count: int
    mae: float | None
    rmse: float | None
    bias: float | None


class RecapCloserCounts(_Frozen):
    """How often each side landed closer, over games both sides forecast."""

    model: int
    market: int
    tie: int


class RecapWinComparison(_Frozen):
    """Model against closing-market win probability."""

    model: RecapProbabilityMetrics
    market: RecapProbabilityMetrics
    closer: RecapCloserCounts


class RecapPointComparison(_Frozen):
    """Model against closing-market point forecast."""

    model: RecapErrorMetrics
    market: RecapErrorMetrics
    closer: RecapCloserCounts


class RecapWeekMetrics(_Frozen):
    """The week's model and market metrics; projected-score errors are per team score."""

    win: RecapWinComparison
    spread: RecapPointComparison
    total: RecapPointComparison
    projected_score: RecapPointComparison


class RecapGameForecast(_Frozen):
    """The product row's forecast provenance; null fields when the game is not in the product."""

    role: ForecastRole | None
    run_id: str | None
    generated_at: datetime | None
    carried: bool | None = Field(
        description="Whether the row was carried from an earlier run than the product's.",
    )


class RecapGameResult(_Frozen):
    """The final score; ``home_win`` is null on a tie."""

    away_score: float
    home_score: float
    home_margin: float
    total: float
    home_win: bool | None


class RecapMarketAtForecast(_Frozen):
    """The market lines the product was composed against."""

    status: ForecastMarketStatus
    home_margin: float | None
    total: float | None
    ingested_at: datetime | None


class RecapClosingMarket(_Frozen):
    """The closing lines (``VEGAS_LINE`` as home margin, ``OVER_UNDER``) and what they imply."""

    spread_status: ClosingLineStatus
    home_margin: float | None
    home_win_prob: float | None
    total_status: ClosingLineStatus
    total: float | None
    implied_away_score: float | None
    implied_home_score: float | None


class RecapWinForecast(_Frozen):
    """The home-win probability against the result and the closing market."""

    status: ForecastStatus
    home_win_prob: float | None
    model_brier: float | None
    market_brier: float | None
    closer: Closer | None


class RecapPointForecast(_Frozen):
    """A point forecast (home margin or total) against the result and the closing market."""

    status: ForecastStatus
    value: float | None
    model_error: float | None = Field(description="Model forecast minus the result.")
    market_error: float | None = Field(description="Closing line minus the result.")
    closer: Closer | None


class RecapProjectedScoreForecast(_Frozen):
    """Projected team scores against the result and the market-implied team scores."""

    status: ForecastStatus
    away_score: float | None
    home_score: float | None
    model_abs_error: float | None = Field(description="Mean absolute error per team score.")
    market_abs_error: float | None = Field(description="Mean absolute error per team score.")
    closer: Closer | None


class RecapStarterSide(_Frozen):
    """One side's assumed, nflverse and carried-forward starter against who started.

    The ``*_correct`` flags are null unless ``status`` is ``available`` and the predictor made a
    pick. The actual starter is shown whenever it is known, whatever the status.
    """

    status: StarterStatus
    record_source: StarterRecordSource | None = Field(
        description="The product's own record, or the capture of the sidecar its run used.",
    )
    assumed_id: str | None
    assumed_name: str | None
    assumed_source: str | None = Field(description="How the assumed starter was resolved.")
    assumed_as_of: str | None = Field(description="The projection snapshot time, if projected.")
    nflverse_id: str | None = Field(description="nflverse's schedule value at composition.")
    nflverse_name: str | None
    carried_forward_id: str | None
    carried_forward_name: str | None
    actual_id: str | None
    actual_name: str | None
    actual_source: ActualStarterSource | None
    assumed_correct: bool | None
    nflverse_correct: bool | None
    carry_forward_correct: bool | None


class RecapGameStarters(_Frozen):
    """Both sides' starters."""

    away: RecapStarterSide
    home: RecapStarterSide


class RecapGame(_Frozen):
    """One scheduled game of the week, whether or not the product holds it."""

    game_id: str
    away_team: str
    home_team: str
    kickoff: datetime
    product_row_status: ProductRowStatus
    forecast: RecapGameForecast
    result: RecapGameResult
    market_at_forecast: RecapMarketAtForecast
    closing_market: RecapClosingMarket
    win: RecapWinForecast
    spread: RecapPointForecast
    total: RecapPointForecast
    projected_score: RecapProjectedScoreForecast
    starting_qbs: RecapGameStarters


class RecapPredictorAccuracy(_Frozen):
    """How often one predictor named the actual starter, over the slots it made a pick for."""

    count: int
    correct: int
    accuracy: float | None


class RecapStarterCounts(_Frozen):
    """Starter slots (two per game) by status, and each predictor's accuracy."""

    slots: int
    scored: int
    from_capture: int = Field(description="Scored slots whose record came from a sidecar capture.")
    not_recorded: int
    starter_unresolved: int
    not_in_product: int
    actual_missing: int
    actual_from_game_logs: int = Field(
        description="Slots whose actual starter is the top passer, not nflverse's value.",
    )
    assumed: RecapPredictorAccuracy
    nflverse: RecapPredictorAccuracy
    carry_forward: RecapPredictorAccuracy


class RecapStartingQbs(RecapStarterCounts):
    """The week's starter slots by status, each predictor's accuracy, and the captures read."""

    capture_digests: list[str] = Field(description="The sidecar captures the section read.")


class WeeklyRecapResponse(BaseResponse):
    """The selected recap of one week.

    ``edges`` and ``wagers`` are null and marked pending in ``_meta.field_status`` until the
    recap carries those sections. The season so far is served by ``GET /recaps/{season}/summary``.
    """

    season: str
    week: int
    recap_id: str
    built_at: datetime
    selected_at: datetime
    product_id: str
    product_run_id: str
    carried_run_ids: list[str]
    product_schema_version: int
    forecast_role: ForecastRole | None
    conventions: RecapConventionsResponse
    counts: RecapCountsResponse
    metrics: RecapWeekMetrics
    games: list[RecapGame]
    starting_qbs: RecapStartingQbs
    edges: None = None
    wagers: None = None


class RecapPairedDifference(_Frozen):
    """Mean per-game model-minus-market difference over games both sides forecast.

    Negative means the model's error was lower. The interval is mean +/- ``interval_z`` x the
    standard error: approximate, by the normal approximation, and not a verdict.
    """

    count: int
    mean: float | None
    standard_error: float | None = Field(description="Sample standard deviation over sqrt(n).")
    interval_low: float | None
    interval_high: float | None


class RecapPairedDifferences(_Frozen):
    """Win Brier and absolute spread, total and per-team-score error, model minus market."""

    win: RecapPairedDifference
    spread: RecapPairedDifference
    total: RecapPairedDifference
    projected_score: RecapPairedDifference


class RecapReliabilityBin(_Frozen):
    """One fixed home-win probability bin; the means are null when it holds no game."""

    lower: float
    upper: float
    count: int
    mean_forecast: float | None
    observed_rate: float | None = Field(description="Share of the bin's games the home team won.")


class RecapSideCalibration(_Frozen):
    """One side's reliability over every bin, and its ECE over ``count`` decided games."""

    count: int
    ece: float | None
    bins: list[RecapReliabilityBin]


class RecapCalibration(_Frozen):
    """Model and closing-market home win probability on the same bins."""

    model: RecapSideCalibration
    market: RecapSideCalibration


class SeasonRecapScope(_Frozen):
    """The season so far over one scope: every recapped row, or the live rows."""

    weeks: int = Field(description="Weeks with at least one row in the scope.")
    games: int
    metrics: RecapWeekMetrics
    paired: RecapPairedDifferences
    calibration: RecapCalibration
    starting_qbs: RecapStarterCounts


class SeasonRecapWeek(_Frozen):
    """One finished week: its own values and the season to date through it.

    Every metric is null when ``status`` is ``not_built``; a live metric scores no game when the
    week has no live row.
    """

    week: int
    status: SeasonWeekStatus
    recap_id: str | None
    forecast_role: ForecastRole | None
    games: int
    live_games: int
    metrics: RecapWeekMetrics | None
    live_metrics: RecapWeekMetrics | None
    to_date: RecapWeekMetrics | None
    live_to_date: RecapWeekMetrics | None


class SeasonRecapConventions(_Frozen):
    """The conventions the season's aggregates are stated in."""

    recap: RecapConventionsResponse
    season_to_date: str
    live_only: str
    calibration_bins: str
    paired_difference: str
    interval: str
    interval_z: float


class SeasonRecapResponse(BaseResponse):
    """The selected season recap, rolled up from the season's selected weekly recaps."""

    season: str
    season_recap_id: str
    built_at: datetime
    selected_at: datetime
    weekly_recap_ids: list[str]
    conventions: SeasonRecapConventions
    weeks: list[SeasonRecapWeek]
    all_games: SeasonRecapScope
    live_games: SeasonRecapScope
