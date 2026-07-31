# tests/integration/test_odds_join.py

"""Integration: adapt and persist nflverse schedule markets by canonical game ID."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.ingest.odds.nflverse_schedule import (
    NFLVERSE_SCHEDULE_SOURCE,
    adapt_nflverse_schedule_markets,
)
from gridiron_edge.ingest.odds.store import (
    load_current_odds,
    write_current_odds_snapshot,
)
from gridiron_edge.market.edge_diagnostics import EdgeResultState
from gridiron_edge.market.weekly_edge_service import build_weekly_edge_result


def _rich_schedule() -> DataFrame:
    """Return schedule truth with complete, incomplete, and unmatched games."""
    timestamp = datetime(2026, 7, 30, 18, tzinfo=UTC)
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [1, 1, 1],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
                "2026_01_GB_CHI",
            ],
            "game_date": ["2026-09-10", "2026-09-13", "2026-09-13"],
            "away_team": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
                "Green Bay Packers",
            ],
            "home_team": [
                "Los Angeles Chargers",
                "Buffalo Bills",
                "Chicago Bears",
            ],
            "away_moneyline": [-120.0, None, None],
            "home_moneyline": [105.0, None, None],
            "spread_line": [-2.5, None, None],
            "away_spread_odds": [-110.0, None, None],
            "home_spread_odds": [-110.0, None, None],
            "total_line": [45.5, None, None],
            "over_odds": [-110.0, None, None],
            "under_odds": [-110.0, None, None],
            "source": ["nflverse", "nflverse", "nflverse"],
            "ingested_at": [timestamp, timestamp, timestamp],
        }
    )


def test_nflverse_markets_roundtrip_and_join_by_schedule_game_id(
    tmp_path: Path,
) -> None:
    """Persist adapted markets and retain complete and incomplete joins."""
    schedule = _rich_schedule()
    adapted = adapt_nflverse_schedule_markets(
        schedule.iloc[:2].copy(),
        season="2026-2027",
        week=1,
    )

    snapshot_path = write_current_odds_snapshot(
        adapted,
        repo=tmp_path,
    )
    loaded = load_current_odds(repo=tmp_path)

    assert loaded is not None
    assert snapshot_path.name == "odds_current.parquet"
    assert set(loaded["sportsbook"]) == {NFLVERSE_SCHEDULE_SOURCE}
    assert "draftkings" not in set(loaded["sportsbook"])
    assert loaded["fetched_at"].nunique() == 1
    assert loaded["fetched_at"].iloc[0] == pd.Timestamp("2026-07-30T18:00:00Z")

    market_coverage = (
        loaded.groupby("game_id", sort=False)
        .agg(
            market_rows=("market", "size"),
            populated_odds=("odds", "count"),
        )
        .reset_index()
    )
    joined = schedule.merge(
        market_coverage,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    assert len(joined) == len(schedule)
    assert joined["game_id"].tolist() == schedule["game_id"].tolist()

    complete = joined.loc[joined["game_id"] == "2026_01_KC_LAC"].iloc[0]
    incomplete = joined.loc[joined["game_id"] == "2026_01_BAL_BUF"].iloc[0]
    unmatched = joined.loc[joined["game_id"] == "2026_01_GB_CHI"].iloc[0]

    assert complete["market_rows"] == 6
    assert complete["populated_odds"] == 6
    assert incomplete["market_rows"] == 6
    assert incomplete["populated_odds"] == 0
    assert pd.isna(unmatched["market_rows"])
    assert pd.isna(unmatched["populated_odds"])

    assert not (tmp_path / "data" / "odds" / "dk_odds_current.parquet").exists()


def test_loaded_spread_and_total_sides_preserve_normalized_values(
    tmp_path: Path,
) -> None:
    """Round-trip normalized spread orientation and total-side values."""
    adapted = adapt_nflverse_schedule_markets(
        _rich_schedule().iloc[:1].copy(),
        season="2026-2027",
        week=1,
    )
    write_current_odds_snapshot(adapted, repo=tmp_path)
    loaded = load_current_odds(repo=tmp_path)

    assert loaded is not None
    spread = loaded.loc[loaded["market"] == "spread"].set_index("side")
    total = loaded.loc[loaded["market"] == "total"].set_index("side")

    assert spread.loc["away", "line"] == 2.5
    assert spread.loc["home", "line"] == -2.5
    assert spread.loc["away", "odds"] == -110.0
    assert spread.loc["home", "odds"] == -110.0
    assert total.loc["over", "line"] == 45.5
    assert total.loc["under", "line"] == 45.5
    assert total.loc["over", "odds"] == -110.0
    assert total.loc["under", "odds"] == -110.0


# ---------------------------------------------------------------------------
# Persisted weekly product + market snapshot -> shared edge result
# ---------------------------------------------------------------------------


def _weekly_product_for_edges() -> DataFrame:
    """Return one persisted product with independent win and Total models."""
    return DataFrame(
        {
            "season": ["2026-2027"],
            "week": [1],
            "game_id": ["2026_01_KC_LAC"],
            "game_date": ["2026-09-10"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "neutral_site": [False],
            "win_status": ["available"],
            "away_win_prob": [0.30],
            "home_win_prob": [0.70],
            "win_model_name": ["win_prob"],
            "win_model_type": ["elo"],
            "win_event_id": ["win-event-1"],
            "win_run_id": ["win-run-1"],
            "spread_status": ["available"],
            "model_spread": [-7.0],
            "spread_uncertainty": [13.5],
            "spread_source_event_id": ["win-event-1"],
            "spread_model_name": ["win_prob"],
            "spread_model_type": ["elo"],
            "spread_calibration_key": ["win_prob_elo"],
            "spread_calibration_updated_at": ["2026-07-30T12:00:00+00:00"],
            "total_status": ["available"],
            "model_total": [52.0],
            "total_uncertainty": [12.8],
            "total_model_name": ["total"],
            "total_model_type": ["xgboost"],
            "total_event_id": ["total-event-1"],
            "total_run_id": ["total-run-1"],
            "total_uncertainty_trained_at": ["2026-07-01T14:20:00"],
            "projected_score_status": ["available"],
            "projected_home_score": [29.5],
            "projected_away_score": [22.5],
        }
    )


def _persist_selected_product(repo: Path) -> None:
    """Write and explicitly select one immutable weekly product."""
    identity = WeeklyProductIdentity(
        product_id="weekly-edge-product",
        run_id="weekly-edge-run",
        season="2026-2027",
        week=1,
        generated_at=datetime(2026, 9, 9, 12, tzinfo=UTC),
    )
    write_weekly_product(
        repo,
        _weekly_product_for_edges(),
        identity=identity,
    )
    select_current_weekly_product(
        repo,
        identity.product_id,
        season=identity.season,
        week=identity.week,
        selected_at=datetime(2026, 9, 9, 13, tzinfo=UTC),
    )


def _persist_edge_markets(repo: Path) -> None:
    """Adapt and persist the matching nflverse market snapshot."""
    adapted = adapt_nflverse_schedule_markets(
        _rich_schedule().iloc[:1].copy(),
        season="2026-2027",
        week=1,
    )
    write_current_odds_snapshot(adapted, repo=repo)


def test_weekly_edge_service_roundtrip_uses_persisted_product_and_markets(
    tmp_path: Path,
) -> None:
    """Build one shared edge result from both persisted boundaries."""
    _persist_selected_product(tmp_path)
    _persist_edge_markets(tmp_path)

    result = build_weekly_edge_result(
        season="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert result.diagnostics.state is EdgeResultState.POSITIVE_EDGES
    assert result.diagnostics.prediction_game_count == 1
    assert result.diagnostics.market_game_count == 1
    assert result.diagnostics.matched_game_count == 1
    assert result.diagnostics.eligible_market_count == 3
    assert result.diagnostics.filtered_edge_count == len(result.rows)
    assert set(result.rows["game_id"]) == {"2026_01_KC_LAC"}
    assert set(result.rows["market_type"]) == {
        "moneyline",
        "spread",
        "total",
    }
    assert result.rows["kelly_stake"].isna().all()

    provenance = result.diagnostics.provenance
    assert provenance.win_event_ids == ("win-event-1",)
    assert provenance.win_run_ids == ("win-run-1",)
    assert provenance.win_model_types == ("elo",)
    assert provenance.total_event_ids == ("total-event-1",)
    assert provenance.total_run_ids == ("total-run-1",)
    assert provenance.total_model_types == ("xgboost",)
    assert provenance.product_ids == ("weekly-edge-product",)
    assert provenance.product_run_ids == ("weekly-edge-run",)
    assert provenance.market_sources == (NFLVERSE_SCHEDULE_SOURCE,)


def test_weekly_edge_service_explicit_bankroll_populates_stakes(
    tmp_path: Path,
) -> None:
    """Use the same persisted inputs while enabling dollar stake output."""
    _persist_selected_product(tmp_path)
    _persist_edge_markets(tmp_path)

    result = build_weekly_edge_result(
        season="2026-2027",
        week=1,
        bankroll=2500.0,
        kelly_multiplier=0.10,
        repo=tmp_path,
    )

    assert not result.rows.empty
    assert result.rows["kelly_stake"].notna().all()
    assert (result.rows["kelly_stake"] >= 0.0).all()
    assert (result.rows["kelly_stake"] <= 250.0).all()


def test_weekly_edge_service_threshold_keeps_pre_filter_diagnostics(
    tmp_path: Path,
) -> None:
    """An empty display table retains calculated and positive row counts."""
    _persist_selected_product(tmp_path)
    _persist_edge_markets(tmp_path)

    baseline = build_weekly_edge_result(
        season="2026-2027",
        week=1,
        repo=tmp_path,
    )
    filtered = build_weekly_edge_result(
        season="2026-2027",
        week=1,
        min_ev=1.0,
        repo=tmp_path,
    )

    assert baseline.diagnostics.positive_edge_count > 0
    assert filtered.rows.empty
    assert filtered.diagnostics.state is EdgeResultState.POSITIVE_EDGES
    assert filtered.diagnostics.calculated_edge_count == baseline.diagnostics.calculated_edge_count
    assert filtered.diagnostics.positive_edge_count == baseline.diagnostics.positive_edge_count
    assert filtered.diagnostics.filtered_edge_count == 0
