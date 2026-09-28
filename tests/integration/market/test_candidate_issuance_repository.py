"""Real-repository candidate issuance assessment without persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gridiron_edge.datasets.loaders import load_current_weekly_product
from gridiron_edge.evaluation.forecast_store import load_forecast_events
from gridiron_edge.ingest.odds.store import load_odds_ledger
from gridiron_edge.market.candidate_issuance import issue_pregame_candidates


def test_current_week_one_history_can_issue_candidates() -> None:
    repo = Path(__file__).resolve().parents[3]
    product = load_current_weekly_product(repo, season="2026-2027", week=1)
    run_ids = product["product_run_id"].dropna().astype(str).unique().tolist()
    assert len(run_ids) == 1
    events = load_forecast_events(
        season="2026-2027",
        week=1,
        run_id=run_ids[0],
        repo=repo,
    )
    quotes = load_odds_ledger(season="2026-2027", week=1, repo=repo)
    issuance = issue_pregame_candidates(
        product=product,
        forecast_events=events,
        quotes=quotes,
        evaluated_at=datetime(2026, 8, 18, 14, 45, tzinfo=UTC),
    )
    assert issuance.product_id == product["product_id"].iloc[0]
    assert issuance.product_run_id == run_ids[0]
    # The canonical Week 1 quote ledger has grown as the real collector
    # worker executed its full 34-poll plan (pulled onto this workspace via
    # `gridiron ops pull-collector-evidence`); it no longer stops at the
    # original 1,680-row rehearsal snapshot. Row count must simply match the
    # ledger it was issued from, not a frozen historical value.
    assert len(issuance.rows) == len(quotes) > 1680
    assert {row.market for row in issuance.rows} == {"moneyline", "spread", "total"}
