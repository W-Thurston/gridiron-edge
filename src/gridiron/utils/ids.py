from __future__ import annotations

import re

# Map some common alternates to your preferred 3-letter codes (e.g., KC -> KAN)
TEAM_CANON = {
    "KC": "KAN",
    "KCC": "KAN",
    "KAN": "KAN",
    "JAC": "JAX",
    "JAX": "JAX",
    "LA": "LAR",
    "LAR": "LAR",
    "STL": "STL",
    "SD": "SDG",
    "SDG": "SDG",  # legacy
    "TB": "TAM",
    "TAM": "TAM",
    "GB": "GNB",
    "GNB": "GNB",
    "NO": "NOR",
    "NOR": "NOR",
    "SF": "SFO",
    "SFO": "SFO",
    "NE": "NWE",
    "NWE": "NWE",
    "LV": "LVR",
    "LVR": "LVR",
    "OAK": "OAK",
    "WSH": "WAS",
    "WAS": "WAS",
    "WFT": "WAS",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "HOU": "HOU",
    "IND": "IND",
    "MIA": "MIA",
    "MIN": "MIN",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "TEN": "TEN",
}


def canon_team(code: str) -> str:
    code = code.strip().upper()
    return TEAM_CANON.get(code, code)


def human_game_id(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{week:02d}_{canon_team(away)}_{canon_team(home)}"


def parse_human_game_id(game_id: str) -> tuple[int, int, str, str]:
    m = re.match(r"^(\d{4})_(\d{2})_([A-Z]{2,3})_([A-Z]{2,3})$", game_id)
    if not m:
        raise ValueError(f"Bad game_id: {game_id}")
    season, week, away, home = m.groups()
    return int(season), int(week), away, home
