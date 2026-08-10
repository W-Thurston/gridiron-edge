"""Pure coverage diagnostics for canonical historical quote observations.

Coverage describes observed source and temporal depth only. It does not select
or infer opening, closing, movement, CLV, backtest, or recommendation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pandas import DataFrame

from gridiron_edge.ingest.odds.store import validate_quote_rows

HISTORY_IDENTITY_COLUMNS: tuple[str, ...] = (
    "provider",
    "provider_event_id",
    "sportsbook",
    "game_id",
    "market",
    "side",
)


@dataclass(frozen=True, slots=True)
class QuoteHistoryCoverage:
    """Coverage and temporal depth of canonical quote observations."""

    row_count: int
    provider_count: int
    sportsbook_count: int
    game_count: int
    market_identity_count: int
    fetch_count: int
    earliest_fetched_at: datetime | None
    latest_fetched_at: datetime | None
    identities_with_multiple_observations: int
    identities_with_multiple_fetches: int
    maximum_observations_per_identity: int
    maximum_fetches_per_identity: int
    pregame_observation_count: int
    live_observation_count: int
    missing_commence_time_count: int
    repeated_observation_evidence_available: bool


def evaluate_quote_history_coverage(
    observations: DataFrame,
) -> QuoteHistoryCoverage:
    """Describe quote coverage without interpreting historical market state."""
    rows = validate_quote_rows(observations)
    if rows.empty:
        return QuoteHistoryCoverage(
            row_count=0,
            provider_count=0,
            sportsbook_count=0,
            game_count=0,
            market_identity_count=0,
            fetch_count=0,
            earliest_fetched_at=None,
            latest_fetched_at=None,
            identities_with_multiple_observations=0,
            identities_with_multiple_fetches=0,
            maximum_observations_per_identity=0,
            maximum_fetches_per_identity=0,
            pregame_observation_count=0,
            live_observation_count=0,
            missing_commence_time_count=0,
            repeated_observation_evidence_available=False,
        )

    identities = rows.groupby(
        list(HISTORY_IDENTITY_COLUMNS),
        dropna=False,
        sort=True,
    )
    observation_counts = identities.size()
    fetch_counts = identities["fetched_at"].nunique()
    repeated_fetches = int(fetch_counts.gt(1).sum())
    fetched = rows["fetched_at"]
    live_count = int(rows["is_live"].sum())

    return QuoteHistoryCoverage(
        row_count=len(rows),
        provider_count=int(rows["provider"].nunique(dropna=True)),
        sportsbook_count=int(rows["sportsbook"].nunique(dropna=True)),
        game_count=int(rows["game_id"].nunique(dropna=True)),
        market_identity_count=len(observation_counts),
        fetch_count=int(fetched.nunique()),
        earliest_fetched_at=fetched.min().to_pydatetime(),
        latest_fetched_at=fetched.max().to_pydatetime(),
        identities_with_multiple_observations=int(observation_counts.gt(1).sum()),
        identities_with_multiple_fetches=repeated_fetches,
        maximum_observations_per_identity=int(observation_counts.max()),
        maximum_fetches_per_identity=int(fetch_counts.max()),
        pregame_observation_count=len(rows) - live_count,
        live_observation_count=live_count,
        missing_commence_time_count=int(rows["commence_time"].isna().sum()),
        repeated_observation_evidence_available=repeated_fetches > 0,
    )
