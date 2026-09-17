# src/gridiron_edge/betting/recording.py
"""Rollback-safe orchestration for recording one wager locally."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

from gridiron_edge.betting._artifact_transaction import (
    restore_artifacts,
    snapshot_artifact,
)
from gridiron_edge.betting.bankroll import _txn_path, record_bet_placed
from gridiron_edge.betting.ledger import _LEDGER_LOCK, _bet_ledger_path, log_bet


@dataclass(frozen=True, slots=True)
class RecommendationRecordingEvidence:
    """Trusted persisted recommendation evidence attached to a wager."""

    result_id: str
    evaluation_id: str
    candidate_reference_id: str
    policy_id: str
    game_id: str
    market_type: str
    side: str
    provider: str
    provider_event_id: str | None
    sportsbook: str | None
    fetched_at: datetime
    sportsbook_updated_at: datetime | None
    commence_time: datetime | None
    american_odds: int | None
    line: float | None
    model_name: str | None
    model_type: str | None
    model_probability: float | None
    expected_value: float | None


@dataclass(frozen=True, slots=True)
class RecordWagerCommand:
    """Validated recorded-wager terms and optional trusted provenance."""

    game_id: str
    market_type: str
    side: str
    odds: int
    stake: float
    book: str
    line: float | None = None
    model_name: str | None = None
    model_type: str | None = None
    model_probability: float | None = None
    expected_value: float | None = None
    edge_strength: str | None = None
    confidence_tier: str | None = None
    recommendation: RecommendationRecordingEvidence | None = None


@dataclass(frozen=True, slots=True)
class RecordedWager:
    """Identities created by one successful recorded-wager operation."""

    bet_id: str
    bankroll_transaction_id: str


def _validate(command: RecordWagerCommand) -> None:
    """Validate one live single-game wager command."""
    if not command.game_id.strip():
        raise ValueError("game_id must be a nonempty string.")
    if command.market_type not in {"moneyline", "spread", "total"}:
        raise ValueError("market_type must be moneyline, spread, or total.")
    valid_sides = {"over", "under"} if command.market_type == "total" else {"home", "away"}
    if command.side not in valid_sides:
        raise ValueError("side is not valid for market_type.")
    if isinstance(command.odds, bool) or not isinstance(command.odds, int) or command.odds == 0:
        raise ValueError("odds must be a nonzero integer.")
    if not isfinite(command.stake) or command.stake <= 0:
        raise ValueError("stake must be finite and positive.")
    if not command.book.strip():
        raise ValueError("book must be a nonempty string.")
    if command.market_type == "moneyline" and command.line is not None:
        raise ValueError("moneyline wagers must not provide line.")
    if command.market_type != "moneyline" and (command.line is None or not isfinite(command.line)):
        raise ValueError("spread and total wagers require a finite line.")
    evidence = command.recommendation
    if evidence is not None and (
        evidence.game_id != command.game_id
        or evidence.market_type != command.market_type
        or evidence.side != command.side
    ):
        raise ValueError("Recommendation evidence does not match recorded wager identity.")


def record_wager(
    command: RecordWagerCommand,
    *,
    repo: Path,
    placed_at: datetime | None = None,
) -> RecordedWager:
    """Record wager and cash placement as one compensating-atomic action.

    An explicit placement timestamp is normalized once and supplied to both
    persisted ledgers. When omitted, one UTC timestamp is captured for the
    complete operation rather than allowing each ledger to observe a different
    wall-clock instant.
    """
    _validate(command)
    evidence = command.recommendation
    recorded_at = placed_at or datetime.now(UTC)

    with _LEDGER_LOCK:
        ledger_snapshot = snapshot_artifact(_bet_ledger_path(repo))
        bankroll_snapshot = snapshot_artifact(_txn_path(repo))
        try:
            bet_id = log_bet(
                command.game_id,
                market_type=command.market_type,
                side=command.side,
                odds=command.odds,
                stake=command.stake,
                book=command.book,
                line=command.line,
                model_name=(evidence.model_name if evidence else command.model_name),
                model_type=(evidence.model_type if evidence else command.model_type),
                model_prob=(evidence.model_probability if evidence else command.model_probability),
                model_ev=(evidence.expected_value if evidence else command.expected_value),
                edge_strength=command.edge_strength,
                confidence_tier=command.confidence_tier,
                reference_provider=evidence.provider if evidence else None,
                reference_provider_event_id=(evidence.provider_event_id if evidence else None),
                reference_sportsbook=(evidence.sportsbook if evidence else None),
                reference_market_fetched_at=(evidence.fetched_at if evidence else None),
                reference_sportsbook_updated_at=(
                    evidence.sportsbook_updated_at if evidence else None
                ),
                reference_commence_time=(evidence.commence_time if evidence else None),
                reference_american_odds=(evidence.american_odds if evidence else None),
                reference_line=evidence.line if evidence else None,
                recommended_bet_result_id=(evidence.result_id if evidence else None),
                recommendation_evaluation_id=(evidence.evaluation_id if evidence else None),
                candidate_reference_id=(evidence.candidate_reference_id if evidence else None),
                recommendation_policy_id=(evidence.policy_id if evidence else None),
                placed_at=recorded_at,
                repo=repo,
            )
            transaction_id = record_bet_placed(
                command.stake,
                bet_id=bet_id,
                placed_at=recorded_at,
                repo=repo,
            )
        except Exception:
            restore_artifacts(
                (ledger_snapshot, bankroll_snapshot),
                failure_message=(
                    "Recorded-wager write failed and artifact restoration was incomplete."
                ),
            )
            raise

    return RecordedWager(bet_id, transaction_id)
