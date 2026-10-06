# src/gridiron_edge/evaluation/weekly_recap_build.py

"""Build, store and select a finished week's recap and its season's recap (``DECISIONS.md`` D86).

Shared by ``gridiron evaluate weekly-recap`` and the closeout's ``build-weekly-recap`` stage, so a
recap built by hand and one built at closeout pass the same refusals and follow the same selection
rules. The season recap is rolled up from the season's selected weekly recaps only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from gridiron_edge.betting.ledger import load_bets
from gridiron_edge.datasets import loaders
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.evaluation import forecast_store
from gridiron_edge.evaluation.prediction_input_evidence import PredictionSourceState
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    list_prediction_input_evidence_by_run,
)
from gridiron_edge.evaluation.week_schedule import finished_weeks, scheduled_games
from gridiron_edge.evaluation.weekly_recap import (
    WEEKLY_RECAP_SCHEMA_VERSION,
    BuiltWeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
    unplayed_game_ids,
)
from gridiron_edge.evaluation.weekly_recap_markets import MarketInputs
from gridiron_edge.evaluation.weekly_recap_season import SeasonRecap, build_season_recap
from gridiron_edge.evaluation.weekly_recap_starters import (
    SidecarCapture,
    runs_without_starter_record,
)
from gridiron_edge.evaluation.weekly_recap_store import (
    RecapWriteOutcome,
    get_current_season_recap_selection,
    get_current_weekly_recap_selection,
    list_current_weekly_recap_selections,
    load_weekly_recap,
    select_current_season_recap,
    select_current_weekly_recap,
    write_season_recap,
    write_weekly_recap,
)
from gridiron_edge.ingest.odds.store import load_odds_ledger
from gridiron_edge.market.candidate_issuance_store import candidate_issuances_for_product
from gridiron_edge.market.recommended_bet_result_store import (
    load_latest_recommended_bet_results,
)
from gridiron_edge.ratings.history import season_start
from gridiron_edge.weekly_cycle.receipts import read_receipts


class RecapSelectionOutcome(StrEnum):
    """What storing and selecting a built recap did."""

    NEW = "new"
    REPLAYED = "replayed"
    RESELECTED = "reselected"


@dataclass(frozen=True, slots=True)
class RecapSelectionResult:
    """The recap now selected for its week, the one it replaced, and how it got there."""

    recap_id: str
    previous_recap_id: str | None
    outcome: RecapSelectionOutcome


def build_weekly_recap_from_repo(
    *,
    repo: Path,
    season: str,
    week: int,
    built_at: datetime,
) -> BuiltWeeklyRecap:
    """Build one week's recap from the raw schedule, the selected product and the cleaned games.

    The starting-QB section also reads the player game logs and the sidecar captures of the runs
    whose rows carry no starter record; the market sections read the product's candidate
    issuance, the week's recommendation results and weekly-cycle receipts, the bet ledger's rows
    for the week's games, and the week's quote-ledger partition.

    Raises:
        WeeklyRecapRefusalError: No game is scheduled, a game has no result yet, no product is
            selected, a sidecar capture fails verification, or the builder refuses the persisted
            state (it names the reason).
    """
    raw = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    schedule = scheduled_games(raw, season_int=season_start(season), week=week)
    if schedule.empty:
        raise WeeklyRecapRefusalError("no regular-season games are scheduled.")
    unplayed = unplayed_game_ids(schedule)
    if unplayed:
        raise WeeklyRecapRefusalError(
            f"{len(unplayed)} of {len(schedule)} games have no result yet: {', '.join(unplayed)}"
        )
    try:
        product = loaders.load_current_weekly_product(repo, season=season, week=week)
    except FileNotFoundError as error:
        raise WeeklyRecapRefusalError("no weekly product is selected.") from error
    return build_weekly_recap(
        season=season,
        week=week,
        schedule=schedule,
        product=product,
        forecast_events=forecast_store.load_forecast_events(repo=repo, season=season, week=week),
        games=loaders.load_games(repo),
        player_game_logs=pd.read_parquet(dataset_path(repo, "player_game_logs")),
        sidecar_captures=load_sidecar_captures(repo, runs_without_starter_record(product)),
        markets=load_market_inputs(repo, season=season, week=week, product=product),
        built_at=built_at,
    )


def load_market_inputs(
    repo: Path, *, season: str, week: int, product: pd.DataFrame
) -> MarketInputs:
    """Load the persisted edge and wager evidence of one week's selected product."""
    product_id = str(product["product_id"].iloc[0])
    product_run_id = str(product["product_run_id"].iloc[0])
    return MarketInputs(
        issuances=candidate_issuances_for_product(
            season=season,
            week=week,
            product_id=product_id,
            product_run_id=product_run_id,
            repo=repo,
        ),
        recommendations=load_latest_recommended_bet_results(season=season, week=week, repo=repo),
        receipts=tuple(read_receipts(repo, season=season, week=week)),
        bets=load_bets(season=season, week=week, repo=repo),
        quotes=load_odds_ledger(season=season, week=week, repo=repo),
    )


def load_sidecar_captures(repo: Path, run_ids: Iterable[str]) -> dict[str, SidecarCapture]:
    """Load the stored capture of the QB/coach sidecar each run was bound to, where one exists.

    A run's prediction-input evidence names its sidecar by digest and size; the capture is the
    file ``sidecar_sha256_{digest}.parquet``. A run whose evidence names no sidecar, or whose
    sidecar was never captured, is left out (its rows read ``not_recorded``).

    Raises:
        WeeklyRecapRefusalError: A run's evidence disagrees about its sidecar, or a capture's
            bytes or size do not match the digest it is named by.
    """
    relative = dataset_path(repo, "qb_coach_sidecar").relative_to(repo).as_posix()
    captures: dict[str, SidecarCapture] = {}
    for run_id in sorted(set(run_ids)):
        references = {
            (reference.content_digest, reference.size_bytes)
            for evidence in list_prediction_input_evidence_by_run(run_id, repo=repo)
            for reference in evidence.source_artifacts
            if reference.relative_path == relative
            and reference.state is PredictionSourceState.PRESENT
            and reference.content_digest is not None
        }
        if not references:
            continue
        if len(references) != 1:
            raise WeeklyRecapRefusalError(
                f"Prediction-input evidence of run {run_id} disagrees about its QB/coach sidecar."
            )
        digest, size = references.pop()
        path = dataset_path(repo, "qb_sidecar_captures") / f"sidecar_sha256_{digest}.parquet"
        if not path.is_file():
            continue
        content = path.read_bytes()
        if len(content) != size or sha256(content).hexdigest() != digest:
            raise WeeklyRecapRefusalError(
                f"Sidecar capture {path.name} does not match the digest and size it is named by."
            )
        captures[run_id] = SidecarCapture(digest, pd.read_parquet(BytesIO(content)))
    return captures


def store_and_select_weekly_recap(
    built: BuiltWeeklyRecap,
    *,
    selected_at: datetime,
    repo: Path,
) -> RecapSelectionResult:
    """Write the recap (or accept an exact replay) and select it for its week.

    A recap that is already the week's selection is a replay and the selection is left untouched;
    a stored recap that is not selected is reselected.
    """
    recap = built.recap
    previous = get_current_weekly_recap_selection(season=recap.season, week=recap.week, repo=repo)
    previous_id = previous.recap_id if previous is not None else None
    written = write_weekly_recap(built, repo=repo)
    if previous_id == recap.recap_id:
        return RecapSelectionResult(recap.recap_id, previous_id, RecapSelectionOutcome.REPLAYED)
    select_current_weekly_recap(recap.recap_id, selected_at=selected_at, repo=repo)
    outcome = (
        RecapSelectionOutcome.NEW
        if written is RecapWriteOutcome.WRITTEN
        else RecapSelectionOutcome.RESELECTED
    )
    return RecapSelectionResult(recap.recap_id, previous_id, outcome)


def build_season_recap_from_repo(*, repo: Path, season: str, built_at: datetime) -> SeasonRecap:
    """Roll the season's selected weekly recaps up; the raw games file names the finished weeks.

    The bet ledger supplies the season's bets without a game.

    Raises:
        WeeklyRecapRefusalError: No weekly recap is selected for the season, or a selected recap
            is not stored at the current weekly recap schema (that week needs rebuilding first).
        ValueError: A selected recap fails verification or does not match its game frame.
    """
    recaps = []
    for selection in list_current_weekly_recap_selections(repo=repo):
        if selection.season != season:
            continue
        try:
            recaps.append(load_weekly_recap(selection.recap_id, repo=repo))
        except FileNotFoundError as error:
            raise WeeklyRecapRefusalError(
                f"week {selection.week}'s selected recap {selection.recap_id[:12]} is not stored "
                f"at weekly recap schema {WEEKLY_RECAP_SCHEMA_VERSION}; rebuild that week first."
            ) from error
    raw = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    return build_season_recap(
        season=season,
        recaps=recaps,
        finished_weeks=finished_weeks(raw, season_int=season_start(season)),
        bets=load_bets(repo=repo),
        built_at=built_at,
    )


def store_and_select_season_recap(
    recap: SeasonRecap,
    *,
    selected_at: datetime,
    repo: Path,
) -> RecapSelectionResult:
    """Write the season recap (or accept an exact replay) and select it for its season."""
    previous = get_current_season_recap_selection(season=recap.season, repo=repo)
    previous_id = previous.season_recap_id if previous is not None else None
    written = write_season_recap(recap, repo=repo)
    if previous_id == recap.season_recap_id:
        return RecapSelectionResult(
            recap.season_recap_id, previous_id, RecapSelectionOutcome.REPLAYED
        )
    select_current_season_recap(recap.season_recap_id, selected_at=selected_at, repo=repo)
    outcome = (
        RecapSelectionOutcome.NEW
        if written is RecapWriteOutcome.WRITTEN
        else RecapSelectionOutcome.RESELECTED
    )
    return RecapSelectionResult(recap.season_recap_id, previous_id, outcome)


def rebuild_season_recap(*, repo: Path, season: str, now: datetime) -> RecapSelectionResult:
    """Roll the season up from its selected weekly recaps, then store and select it."""
    recap = build_season_recap_from_repo(repo=repo, season=season, built_at=now)
    return store_and_select_season_recap(recap, selected_at=now, repo=repo)
