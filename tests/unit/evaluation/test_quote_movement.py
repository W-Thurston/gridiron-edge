# tests/unit/evaluation/test_quote_movement.py

"""Tests for quote drift by age and the market age limit derived from it."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.quote_movement import (
    MIN_PAIRS_PER_BIN,
    LagBin,
    age_limit,
    build_report,
    conservative_limit,
    movement_tables,
)
from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS

pytestmark = pytest.mark.unit

KICKOFF = pd.Timestamp("2026-10-10T17:00:00Z")


def _quotes(
    fetched_at: str,
    home_odds: list[float],
    *,
    game: str = "g1",
    week: int = 4,
) -> DataFrame:
    rows = []
    for index, odds in enumerate(home_odds):
        away = -odds if odds > 0 else abs(odds)
        for side, price in (("home", odds), ("away", away)):
            rows.append(
                {
                    "fetched_at": pd.Timestamp(fetched_at, tz="UTC"),
                    "provider": "the_odds_api",
                    "provider_event_id": game,
                    "sportsbook": f"book{index}",
                    "sportsbook_updated_at": pd.Timestamp(fetched_at, tz="UTC"),
                    "commence_time": KICKOFF,
                    "is_live": False,
                    "season": "2026-2027",
                    "week": week,
                    "game_id": game,
                    "game_date": "2026-10-10",
                    "away_team": "A",
                    "home_team": "B",
                    "market": "moneyline",
                    "side": side,
                    "odds": price,
                    "line": float("nan"),
                }
            )
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def _ledger(snapshots: list[tuple[str, list[float]]]) -> DataFrame:
    return pd.concat([_quotes(at, odds) for at, odds in snapshots], ignore_index=True)


def _bin(lower: float, upper: float, pairs: int, median: float) -> LagBin:
    return LagBin(lower, upper, pairs, median, median * 2)


def test_disagreement_is_how_far_one_book_sits_from_the_consensus() -> None:
    # Three books at -110/-110 style quotes around one consensus.
    ledger = _ledger([("2026-10-08T12:00:00", [-120.0, -110.0, -100.0])])

    movement = movement_tables(ledger)

    assert movement.games == 1
    assert movement.snapshots_by_week == {4: 1}
    assert movement.disagreement_median_points > 0
    assert movement.cross_book_range_median_points > movement.disagreement_median_points


def test_drift_is_the_consensus_movement_between_snapshots_by_age() -> None:
    # Two snapshots 10 hours apart where every book moves the same amount.
    ledger = _ledger(
        [
            ("2026-10-08T12:00:00", [-120.0, -120.0, -120.0]),
            ("2026-10-08T22:00:00", [-130.0, -130.0, -130.0]),
        ]
    )

    movement = movement_tables(ledger)

    drifted = [item for item in movement.bins if item.pairs]
    assert [(item.lower_hours, item.upper_hours, item.pairs) for item in drifted] == [
        (6.0, 12.0, 1)
    ]
    assert drifted[0].median_points > 0


def test_pairs_after_kickoff_are_ignored() -> None:
    ledger = _ledger(
        [
            ("2026-10-08T12:00:00", [-120.0]),
            ("2026-10-10T18:00:00", [-200.0]),
        ]
    )

    movement = movement_tables(ledger)

    assert sum(item.pairs for item in movement.bins) == 0


def test_a_ledger_without_complete_moneylines_is_rejected() -> None:
    ledger = _quotes("2026-10-08T12:00:00", [-110.0])
    ledger = ledger.loc[ledger["side"] == "home"]

    with pytest.raises(ValueError, match="no complete pregame moneyline"):
        movement_tables(ledger)


def test_the_limit_is_the_lower_edge_of_the_first_bin_that_drifts_past_the_books() -> None:
    bins = (
        _bin(0, 1, 40, 0.0),
        _bin(1, 3, 40, 0.1),
        _bin(3, 6, 40, 0.2),
        _bin(6, np.inf, 40, 0.9),
    )

    limit, reason = age_limit(bins, 0.3)

    assert limit == 6
    assert "exceeds" in reason


def test_no_limit_when_drift_never_passes_the_books() -> None:
    limit, reason = age_limit((_bin(0, 1, 40, 0.1), _bin(1, np.inf, 40, 0.2)), 0.3)

    assert limit is None
    assert "never exceeded" in reason


def test_no_limit_when_a_bin_has_too_few_pairs() -> None:
    limit, reason = age_limit(
        (_bin(0, 1, 40, 0.1), _bin(1, np.inf, MIN_PAIRS_PER_BIN - 1, 0.9)), 0.3
    )

    assert limit is None
    assert "fewer than" in reason


def test_a_borderline_limit_takes_the_lower_bin_edge() -> None:
    bins = (
        _bin(0, 12, 40, 0.1),
        _bin(12, 24, 40, 0.28),
        _bin(24, 48, 40, 0.34),
        _bin(48, np.inf, 40, 0.97),
    )

    assert age_limit(bins, 0.34)[0] == 48
    assert conservative_limit(bins, 48, 0.34) == 24
    # A clearly lower previous bin leaves the derived limit alone.
    clear = (*bins[:2], _bin(24, 48, 40, 0.1), bins[3])
    assert conservative_limit(clear, 48, 0.34) == 48
    assert conservative_limit(bins, None, 0.34) is None


def test_the_report_states_the_rule_and_whether_the_configuration_matches() -> None:
    ledger = _ledger(
        [
            ("2026-10-08T12:00:00", [-120.0, -110.0]),
            ("2026-10-09T12:00:00", [-125.0, -115.0]),
        ]
    )
    movement = movement_tables(ledger)

    text = build_report(movement, configured_hours=24, generated_on=date(2026, 10, 4))

    assert "# Quote movement and the market age limit" in text
    assert "Generated 2026-10-04" in text
    assert "post-hoc criterion" in text
    assert "| Age between snapshots (h) | Pairs |" in text
    assert "**Configured (`MAX_MARKET_AGE`):** 24 hours" in text
