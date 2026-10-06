# tests/fixtures/weekly_recap.py

"""A synthetic finished week for weekly-recap tests.

Four scheduled games, chosen so every status appears once:

* ``A`` - every family available, product run, closing lines present; the hand case.
* ``B`` - carried from an earlier run, the game ended tied 20-20.
* ``C`` - scheduled but not in the product.
* ``D`` - total forecast missing (so no projected score), closing lines missing.

Starting QBs (actual starter: nflverse's id when he threw a pass for the team, else the top passer):

* ``A`` - recorded in the product. Away: a mid-week change the assumption caught (assumed and
  nflverse right, carry-forward wrong); the starter left mid-game, so nflverse's id stands
  although a backup threw more. Home: the assumption was wrong, nflverse and carry-forward right.
* ``B`` - carried, no record; scored from the capture of its run's sidecar. Away: nflverse kept
  a stale id (no attempt), so the top passer is the actual starter; only the assumption was
  right. Home: all three right.
* ``C`` - not in the product (actual starters still known).
* ``D`` - recorded with an unresolved home starter; the away team has no game log
  (``actual_missing``).

Markets: no candidate issuance; the cycle published edges, then withheld them as unvalidated
before the last kickoff (``D``, 2026-10-06T00:15Z), and a later receipt falls outside the window.
Recorded wagers: a won moneyline on ``A`` ($1 at -150, paid $1.66), a lost moneyline on ``C``,
an open moneyline on ``B`` (``unsettled``) and a bonus-funded same-game parlay on ``D``; one
parlay without a game belongs to the season, not the week.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame

from gridiron_edge.betting.ledger import _BET_COLUMNS
from gridiron_edge.evaluation.weekly_recap_markets import MarketInputs
from gridiron_edge.evaluation.weekly_recap_starters import SidecarCapture
from gridiron_edge.ingest.odds.store import empty_quote_frame
from gridiron_edge.weekly_cycle.receipts import CycleReceipt

SEASON = "2026-2027"
WEEK = 4
PRODUCT_RUN = "run-product"
CARRIED_RUN = "run-earlier"
BUILT_AT = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

# game_id: (away, home, gameday, gametime, away_score, home_score, vegas_line, over_under)
_GAMES: dict[str, tuple[str, str, str, str, float, float, float | None, float | None]] = {
    "2026_04_AAA_BBB": ("AAA", "BBB", "2026-10-04", "13:00", 17.0, 24.0, 2.5, 45.5),
    "2026_04_CCC_DDD": ("CCC", "DDD", "2026-10-01", "20:15", 20.0, 20.0, -3.0, 41.0),
    "2026_04_EEE_FFF": ("EEE", "FFF", "2026-10-04", "16:25", 30.0, 27.0, -1.0, 50.0),
    "2026_04_GGG_HHH": ("GGG", "HHH", "2026-10-05", "20:15", 10.0, 13.0, None, None),
}
GAME_A, GAME_B, GAME_C, GAME_D = _GAMES

CAPTURE_DIGEST = "c" * 64
PROJECTED_AS_OF = "2026-10-03T12:00:00Z"
# nflverse's *_qb_id per game after the game: (away, home); B's away id is stale.
_NFLVERSE_STARTERS: dict[str, tuple[str, str]] = {
    GAME_A: ("qa1", "qb1"),
    GAME_B: ("qc0", "qd1"),
    GAME_C: ("qe1", "qf1"),
    GAME_D: ("qg1", "qh1"),
}
# (team, player, name, attempts) for the week; GGG has no log.
_PASSERS: tuple[tuple[str, str, str, float], ...] = (
    ("AAA", "qa1", "Away Starter", 6.0),
    ("AAA", "qa9", "Away Backup", 25.0),
    ("BBB", "qb1", "Home Veteran", 30.0),
    ("CCC", "qc1", "Carried Projected", 28.0),
    ("DDD", "qd1", "Carried Home", 20.0),
    ("EEE", "qe1", "Unforecast Away", 31.0),
    ("FFF", "qf1", "Unforecast Home", 22.0),
    ("HHH", "qh1", "Unresolved Home", 18.0),
)
# The product's own record of games A and D, per side: (id, name, source, as_of, nflverse, carried).
_RECORDS: dict[tuple[str, str], tuple[str, str | None, str, str | None, str, str]] = {
    (GAME_A, "away"): ("qa1", "Away Starter", "depth_chart_report", PROJECTED_AS_OF, "qa1", "qa0"),
    (GAME_A, "home"): ("qb2", "Home Rookie", "depth_chart_report", PROJECTED_AS_OF, "qb1", "qb1"),
    (GAME_D, "away"): ("qg1", "Unrecorded Away", "nflverse_schedule", None, "qg1", "qg1"),
}


@dataclass(frozen=True)
class RecapWeek:
    """The inputs of one recap build."""

    schedule: DataFrame
    product: DataFrame
    events: DataFrame
    games: DataFrame
    logs: DataFrame
    captures: dict[str, SidecarCapture]
    markets: MarketInputs


def schedule() -> DataFrame:
    """Return the raw-games rows of the week (regular season, all played)."""
    return DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": WEEK,
                "game_type": "REG",
                "away_team": away,
                "home_team": home,
                "gameday": day,
                "gametime": time,
                "away_score": away_score,
                "home_score": home_score,
                "result": home_score - away_score,
                "away_qb_id": _NFLVERSE_STARTERS[game_id][0],
                "home_qb_id": _NFLVERSE_STARTERS[game_id][1],
            }
            for game_id, (away, home, day, time, away_score, home_score, _, _) in _GAMES.items()
        ]
    )


def cleaned_games() -> DataFrame:
    """Return the cleaned games file rows of the week, plus one game of another week."""
    rows = [
        {
            "GAME_ID": game_id,
            "YEAR": SEASON,
            "WEEK_NUM": WEEK,
            "AWAY_SCORE": away_score,
            "HOME_SCORE": home_score,
            "VEGAS_LINE": line,
            "OVER_UNDER": total,
        }
        for game_id, (_, _, _, _, away_score, home_score, line, total) in _GAMES.items()
    ]
    rows.append(
        {
            "GAME_ID": "2026_03_AAA_BBB",
            "YEAR": SEASON,
            "WEEK_NUM": 3,
            "AWAY_SCORE": 1.0,
            "HOME_SCORE": 2.0,
            "VEGAS_LINE": 1.0,
            "OVER_UNDER": 3.0,
        }
    )
    return DataFrame(rows)


def _product_row(
    game_id: str,
    *,
    run_id: str,
    home_win_prob: float,
    model_spread: float,
    model_total: float | None,
    spread_line: float,
    total_line: float,
) -> dict[str, object]:
    away, home = _GAMES[game_id][:2]
    total_available = model_total is not None
    generated = pd.Timestamp("2026-10-01T12:00Z" if run_id == CARRIED_RUN else "2026-10-03T12:00Z")
    projected_home = projected_away = None
    if total_available:
        assert model_total is not None
        projected_home = (model_total - model_spread) / 2
        projected_away = (model_total + model_spread) / 2
    return {
        "product_schema_version": 2,
        "product_id": "weekly_product",
        "product_run_id": PRODUCT_RUN,
        "product_generated_at": pd.Timestamp("2026-10-03T12:00Z"),
        "season": SEASON,
        "week": WEEK,
        "game_id": game_id,
        "away_team": away,
        "home_team": home,
        "spread_line": spread_line,
        "total_line": total_line,
        "ingested_at": pd.Timestamp("2026-10-03T11:00Z"),
        "win_status": "available",
        "home_win_prob": home_win_prob,
        "win_model_name": "win_prob",
        "win_model_type": "logistic",
        "win_event_id": f"win-{game_id}",
        "win_run_id": run_id,
        "win_generated_at": generated,
        "win_role": "live",
        "spread_status": "available",
        "model_spread": model_spread,
        "spread_source_event_id": f"win-{game_id}",
        "spread_model_name": "win_prob",
        "spread_model_type": "logistic",
        "total_status": "available" if total_available else "model_unavailable",
        "model_total": model_total,
        "total_model_name": "total",
        "total_model_type": "xgboost",
        "total_event_id": f"total-{game_id}" if total_available else None,
        "total_run_id": run_id if total_available else None,
        "total_role": "live" if total_available else None,
        "projected_score_status": "available" if total_available else "total_unavailable",
        "projected_home_score": projected_home,
        "projected_away_score": projected_away,
    } | _starter_record(game_id, carried=run_id == CARRIED_RUN)


def _starter_record(game_id: str, *, carried: bool) -> dict[str, object]:
    """Return the product's starting-QB columns: B (carried) has none, D has no home starter."""
    record: dict[str, object] = {}
    resolved = 0
    for side in ("away", "home"):
        values = None if carried else _RECORDS.get((game_id, side))
        resolved += values is not None
        qb_id, name, source, as_of, nflverse, carried_id = values or (None,) * 6
        record |= {
            f"{side}_qb_id": qb_id,
            f"{side}_qb_name": name,
            f"{side}_qb_source": source,
            f"{side}_qb_projection_as_of": as_of,
            f"{side}_qb_nflverse_id": nflverse,
            f"{side}_qb_carried_forward_id": carried_id,
            f"{side}_qb_differs_from_carry_forward": (
                None if qb_id is None else qb_id != carried_id
            ),
        }
    status = (
        "sidecar_unavailable"
        if carried
        else ("available" if resolved == 2 else "starter_unresolved")
    )
    return {"starting_qb_status": status} | record


def product() -> DataFrame:
    """Return the selected product: A, B (carried) and D, not C."""
    return DataFrame(
        [
            _product_row(
                GAME_A,
                run_id=PRODUCT_RUN,
                home_win_prob=0.6,
                model_spread=-3.0,
                model_total=44.0,
                spread_line=3.0,
                total_line=44.5,
            ),
            _product_row(
                GAME_B,
                run_id=CARRIED_RUN,
                home_win_prob=0.55,
                model_spread=1.0,
                model_total=40.0,
                spread_line=-2.5,
                total_line=41.5,
            ),
            _product_row(
                GAME_D,
                run_id=PRODUCT_RUN,
                home_win_prob=0.3,
                model_spread=-2.0,
                model_total=None,
                spread_line=1.0,
                total_line=39.0,
            ),
        ]
    )


def events(product_frame: DataFrame) -> DataFrame:
    """Return the exact forecast events the product cites."""
    rows: list[dict[str, object]] = []
    for row in product_frame.to_dict("records"):
        base = {
            "season": SEASON,
            "week": WEEK,
            "game_id": row["game_id"],
            "role": "live",
            "home_win_prob": row["home_win_prob"],
            "model_spread": row["model_spread"],
            "model_total": row["model_total"],
        }
        rows.append(
            base
            | {
                "event_id": row["win_event_id"],
                "run_id": row["win_run_id"],
                "model_name": "win_prob",
                "model_type": "logistic",
            }
        )
        if row["total_event_id"] is not None:
            rows.append(
                base
                | {
                    "event_id": row["total_event_id"],
                    "run_id": row["total_run_id"],
                    "model_name": "total",
                    "model_type": "xgboost",
                }
            )
    return DataFrame(rows)


def player_game_logs() -> DataFrame:
    """Return the week's passers, an earlier week's line (a name) and a preseason line."""
    rows = [
        {
            "player_id": player,
            "player_display_name": name,
            "team": team,
            "season": 2026,
            "season_type": "REG",
            "week": WEEK,
            "attempts": attempts,
        }
        for team, player, name, attempts in _PASSERS
    ]
    rows.append(rows[0] | {"player_id": "qa0", "player_display_name": "Away Veteran", "week": 3})
    rows.append(rows[0] | {"player_id": "qa7", "season_type": "PRE", "attempts": 40.0})
    rows.append(rows[0] | {"player_id": "qa8", "player_display_name": "Away Runner", "attempts": 0})
    return DataFrame(rows)


def capture_sidecar() -> DataFrame:
    """Return the sidecar the carried run was bound to: week 3 history, then game B projected."""
    base = {"YEAR": SEASON, "AWAY_COACH": None, "HOME_COACH": None}
    none = {
        "AWAY_QB_ID_PROJECTED": None,
        "AWAY_QB_PROJECTION_SOURCE": None,
        "AWAY_QB_PROJECTION_AS_OF": None,
        "HOME_QB_ID_PROJECTED": None,
        "HOME_QB_PROJECTION_SOURCE": None,
        "HOME_QB_PROJECTION_AS_OF": None,
    }
    rows = [
        base
        | none
        | {
            "GAME_ID": "2026_03_CCC_DDD",
            "WEEK_NUM": 3,
            "AWAY_TEAM": "CCC",
            "HOME_TEAM": "DDD",
            "AWAY_QB_ID": "qc0",
            "AWAY_QB_NAME": "Carried Veteran",
            "HOME_QB_ID": "qd1",
            "HOME_QB_NAME": "Carried Home",
        },
        base
        | none
        | {
            "GAME_ID": GAME_B,
            "WEEK_NUM": WEEK,
            "AWAY_TEAM": "CCC",
            "HOME_TEAM": "DDD",
            "AWAY_QB_ID": "qc0",
            "AWAY_QB_NAME": "Carried Veteran",
            "HOME_QB_ID": "qd1",
            "HOME_QB_NAME": "Carried Home",
            "AWAY_QB_ID_PROJECTED": "qc1",
            "AWAY_QB_PROJECTION_SOURCE": "depth_chart_report",
            "AWAY_QB_PROJECTION_AS_OF": PROJECTED_AS_OF,
        },
    ]
    return DataFrame(rows)


def recap_week() -> RecapWeek:
    """Return the inputs of the synthetic week."""
    selected = product()
    return RecapWeek(
        schedule(),
        selected,
        events(selected),
        cleaned_games(),
        player_game_logs(),
        {CARRIED_RUN: SidecarCapture(CAPTURE_DIGEST, capture_sidecar())},
        market_inputs(),
    )


def edge_receipt(decided_at: str, outcome: str, detail: str) -> CycleReceipt:
    """Return one weekly-cycle edge receipt for the synthetic week."""
    inputs = {"edge_inputs": {"market_fetched_at": decided_at, "product_id": "weekly_product"}}
    return CycleReceipt(
        decided_at=decided_at,
        season=SEASON,
        week=WEEK,
        attempted="edges",
        outcome=outcome,
        detail=detail,
        inputs=inputs,
        results=inputs,
    )


def edge_receipts() -> tuple[CycleReceipt, ...]:
    """Published, then withheld before the last kickoff; the last receipt is after it."""
    withheld = "edge calculation blocked: model_edges_unvalidated"
    return (
        edge_receipt("2026-10-04T20:00:54+00:00", "done", "149 edges (top EV 27.1%)"),
        edge_receipt("2026-10-04T22:00:54+00:00", "blocked", withheld),
        edge_receipt("2026-10-06T01:00:00+00:00", "done", "3 edges (top EV 5.0%)"),
    )


def bet(
    source_bet_id: str,
    game_id: str | None,
    *,
    market_type: str = "moneyline",
    side: str | None = "home",
    odds: int = -150,
    stake: float = 1.0,
    status: str = "lost",
    paid: float | None = None,
    funding_type: str = "cash",
    placed_at: str = "2026-10-01T18:00:00+00:00",
) -> dict[str, object]:
    """Return one ledger row as an import writes it (no quote reference)."""
    row: dict[str, object] = dict.fromkeys(_BET_COLUMNS)
    pnl = None if status == "open" else (-stake if status == "lost" else (paid or 0.0) - stake)
    row |= {
        "bet_id": f"internal-{source_bet_id}",
        "source_bet_id": source_bet_id,
        "game_id": game_id,
        "description": source_bet_id,
        "placed_at": pd.Timestamp(placed_at),
        "market_type": market_type,
        "side": side,
        "odds": odds,
        "stake": stake,
        "book": "draftkings",
        "funding_type": funding_type,
        "paid_amount": paid,
        "status": status,
        "pnl": pnl,
    }
    return row


def ledger() -> DataFrame:
    """Return the whole synthetic ledger: the week's bets and one bet without a game."""
    return DataFrame(
        [
            bet("DK1", GAME_A, status="won", paid=1.66),
            bet("DK2", GAME_C, side="away", odds=120),
            bet("DK3", GAME_B, status="open"),
            bet(
                "DK4",
                GAME_D,
                market_type="same_game_parlay",
                side=None,
                odds=187500,
                stake=0.1,
                funding_type="bonus",
            ),
            bet("DK5", None, market_type="parlay", side=None, odds=211769),
        ],
        columns=list(_BET_COLUMNS),
    )


def market_inputs() -> MarketInputs:
    """Return the synthetic week's edge and wager evidence (no issuance, no quotes)."""
    bets = ledger()
    return MarketInputs(
        issuances=(),
        recommendations=(),
        receipts=edge_receipts(),
        bets=bets.loc[bets["game_id"].notna()].reset_index(drop=True),
        quotes=empty_quote_frame(),
    )
