# tests/fixtures/dk_payload_fixture.py

"""Realistic DraftKings API payload fixture for testing the odds pipeline.

Uses the actual 2026 Week 1 NFL schedule with made-up but realistic odds.

Usage:
    uv run python tests/fixtures/dk_payload_fixture.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 2026 Week 1 matchups with realistic odds
# ---------------------------------------------------------------------------

DK_PAYLOAD_FIXTURE: dict = {
    "events": [
        # Wed Sep 9
        {
            "id": "ev_001",
            "startEventDate": "2026-09-10T00:20:00Z",
            "participants": [
                {"venueRole": "Away", "name": "New England Patriots"},
                {"venueRole": "Home", "name": "Seattle Seahawks"},
            ],
        },
        # Thu Sep 10
        {
            "id": "ev_002",
            "startEventDate": "2026-09-11T00:35:00Z",
            "participants": [
                {"venueRole": "Away", "name": "San Francisco 49ers"},
                {"venueRole": "Home", "name": "Los Angeles Rams"},
            ],
        },
        # Sun Sep 13 1:00 PM ET
        {
            "id": "ev_003",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Chicago Bears"},
                {"venueRole": "Home", "name": "Carolina Panthers"},
            ],
        },
        {
            "id": "ev_004",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Tampa Bay Buccaneers"},
                {"venueRole": "Home", "name": "Cincinnati Bengals"},
            ],
        },
        {
            "id": "ev_005",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "New Orleans Saints"},
                {"venueRole": "Home", "name": "Detroit Lions"},
            ],
        },
        {
            "id": "ev_006",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Buffalo Bills"},
                {"venueRole": "Home", "name": "Houston Texans"},
            ],
        },
        {
            "id": "ev_007",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Baltimore Ravens"},
                {"venueRole": "Home", "name": "Indianapolis Colts"},
            ],
        },
        {
            "id": "ev_008",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Cleveland Browns"},
                {"venueRole": "Home", "name": "Jacksonville Jaguars"},
            ],
        },
        {
            "id": "ev_009",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Atlanta Falcons"},
                {"venueRole": "Home", "name": "Pittsburgh Steelers"},
            ],
        },
        {
            "id": "ev_010",
            "startEventDate": "2026-09-13T17:00:00Z",
            "participants": [
                {"venueRole": "Away", "name": "New York Jets"},
                {"venueRole": "Home", "name": "Tennessee Titans"},
            ],
        },
        # Sun Sep 13 4:25 PM ET
        {
            "id": "ev_011",
            "startEventDate": "2026-09-13T20:25:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Arizona Cardinals"},
                {"venueRole": "Home", "name": "Los Angeles Chargers"},
            ],
        },
        {
            "id": "ev_012",
            "startEventDate": "2026-09-13T20:25:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Miami Dolphins"},
                {"venueRole": "Home", "name": "Las Vegas Raiders"},
            ],
        },
        {
            "id": "ev_013",
            "startEventDate": "2026-09-13T20:25:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Green Bay Packers"},
                {"venueRole": "Home", "name": "Minnesota Vikings"},
            ],
        },
        {
            "id": "ev_014",
            "startEventDate": "2026-09-13T20:25:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Washington Commanders"},
                {"venueRole": "Home", "name": "Philadelphia Eagles"},
            ],
        },
        # Sun Sep 13 8:20 PM ET
        {
            "id": "ev_015",
            "startEventDate": "2026-09-14T00:20:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Dallas Cowboys"},
                {"venueRole": "Home", "name": "New York Giants"},
            ],
        },
        # Mon Sep 14 8:15 PM ET
        {
            "id": "ev_016",
            "startEventDate": "2026-09-15T00:15:00Z",
            "participants": [
                {"venueRole": "Away", "name": "Denver Broncos"},
                {"venueRole": "Home", "name": "Kansas City Chiefs"},
            ],
        },
    ],
    "markets": [],
    "selections": [],
}

# Build markets and selections programmatically to avoid repetition
_ODDS: list[tuple[str, int, int, float, float, float]] = [
    # ev_id, ml_home, ml_away, spread_home, total, spread_odds_home
    ("ev_001", -165, +140, -3.5, 44.5, -110),
    ("ev_002", -130, +110, -2.5, 47.0, -110),
    ("ev_003", -145, +125, -2.5, 41.0, -110),
    ("ev_004", -120, +100, -1.5, 45.5, -110),
    ("ev_005", -190, +160, -4.5, 46.0, -110),
    ("ev_006", -110, -110, 0.0, 48.5, -110),
    ("ev_007", -155, +130, -3.0, 43.5, -110),
    ("ev_008", +115, -135, +2.0, 42.0, -110),
    ("ev_009", -125, +105, -2.0, 43.0, -110),
    ("ev_010", +130, -150, +2.5, 40.5, -110),
    ("ev_011", -170, +145, -3.5, 46.0, -110),
    ("ev_012", +105, -125, +1.5, 41.5, -110),
    ("ev_013", -105, -115, -1.0, 45.0, -110),
    ("ev_014", -210, +175, -4.5, 45.5, -110),
    ("ev_015", -145, +125, -2.5, 44.0, -110),
    ("ev_016", -195, +165, -4.5, 48.5, -110),
]

for ev_id, ml_home, ml_away, spread_home, total, spread_odds in _ODDS:
    DK_PAYLOAD_FIXTURE["markets"].extend(
        [
            {"id": f"{ev_id}_ml", "eventId": ev_id, "name": "Moneyline"},
            {"id": f"{ev_id}_sp", "eventId": ev_id, "name": "Point Spread"},
            {"id": f"{ev_id}_tot", "eventId": ev_id, "name": "Total Points"},
        ]
    )
    DK_PAYLOAD_FIXTURE["selections"].extend(
        [
            {
                "marketId": f"{ev_id}_ml",
                "outcomeType": "home",
                "displayOdds": {"oddsAmerican": str(ml_home)},
            },
            {
                "marketId": f"{ev_id}_ml",
                "outcomeType": "away",
                "displayOdds": {"oddsAmerican": str(ml_away)},
            },
            {
                "marketId": f"{ev_id}_sp",
                "outcomeType": "home",
                "line": spread_home,
                "displayOdds": {"oddsAmerican": str(spread_odds)},
            },
            {
                "marketId": f"{ev_id}_sp",
                "outcomeType": "away",
                "line": -spread_home,
                "displayOdds": {"oddsAmerican": str(spread_odds)},
            },
            {
                "marketId": f"{ev_id}_tot",
                "outcomeType": "over",
                "total": total,
                "displayOdds": {"oddsAmerican": "-110"},
            },
            {
                "marketId": f"{ev_id}_tot",
                "outcomeType": "under",
                "total": total,
                "displayOdds": {"oddsAmerican": "-110"},
            },
        ]
    )


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

    from gridiron_edge.ingest.odds.draftkings import (
        _event_rows_to_team_rows,
        _extract_game_lines,
    )
    from gridiron_edge.ingest.odds.store import (
        append_to_odds_ledger,
        load_current_odds,
        wide_to_long,
        write_current_odds_snapshot,
    )

    print("=" * 60)
    print("DK Odds Pipeline Test - 2026 Week 1 (fixture)")
    print("=" * 60)

    print("\n[1/4] Parsing fixture through collector...")
    df_wide = _extract_game_lines(DK_PAYLOAD_FIXTURE)
    df_wide = _event_rows_to_team_rows(df_wide)
    print(f"      {len(df_wide)} rows ({len(df_wide) // 2} games)")
    print(
        df_wide[["team", "opponent", "location", "moneyline", "spread_value"]].to_string(
            index=False
        )
    )

    print("\n[2/4] Converting to long format...")
    df_long = wide_to_long(df_wide, sportsbook="draftkings", season="2026-2027", week=1)
    print(f"      {len(df_long)} rows ({df_long['market'].value_counts().to_dict()})")

    print("\n[3/4] Writing ledger + snapshot...")
    ledger_path = append_to_odds_ledger(df_long)
    snapshot_path = write_current_odds_snapshot(df_long)
    print(f"      Ledger:   {ledger_path}")
    print(f"      Snapshot: {snapshot_path}")

    print("\n[4/4] Verifying moneyline snapshot...")
    ml = load_current_odds(market="moneyline")
    print(ml[["away_team", "home_team", "side", "odds"]].to_string(index=False))

    print("\n✓ Done. Now run:")
    print("  uv run gridiron output predictions --year 2026-2027 --week 1")
