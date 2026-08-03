# src/gridiron_edge/sim/_types.py

"""Simulation constants, configuration dataclasses, and data containers.

This module is a pure-data leaf - no I/O, no pandas operations, no numba.
Any module in the sim package that needs to know the shape of the data
(e.g. N_TEAMS, SimulationConfig, TeamIndex) imports from here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Final

import numpy as np

from gridiron_edge.core.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from logging import Logger

logger: Logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

N_TEAMS: Final[int] = 32
N_WEEKS_REG: Final[int] = 18
N_PLAYOFF_ROUNDS: Final[int] = 4

# Game outcome encodings used in ScheduleArrays.result
UNPLAYED: Final[np.int8] = np.int8(-1)
AWAY_WIN: Final[np.int8] = np.int8(0)
HOME_WIN: Final[np.int8] = np.int8(1)
TIE: Final[np.int8] = np.int8(2)

# Playoff round indices - duplicated in playoffs.py (numba constraint).
# season.py asserts these stay in sync at import time.
ROUND_WC: Final[int] = 0
ROUND_DIV: Final[int] = 1
ROUND_CONF: Final[int] = 2
ROUND_SB: Final[int] = 3

DIV_CODES: Final[dict[str, int]] = {
    "AFC East": 0,
    "AFC North": 1,
    "AFC South": 2,
    "AFC West": 3,
    "NFC East": 4,
    "NFC North": 5,
    "NFC South": 6,
    "NFC West": 7,
}

CONF_CODES: Final[dict[str, int]] = {"AFC": 0, "NFC": 1}

DIV_CODE_TO_LABEL: Final[dict[int, str]] = {v: k for k, v in DIV_CODES.items()}


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for Monte Carlo simulation.

    Attributes:
        n_sims: Number of Monte Carlo iterations.
        k_factor: Elo K-factor. Should match the tuned optimum from
            ``gridiron evaluate tune elo``.
        divisor: Elo win-probability divisor. Should match the value used
            to build the Elo state table so simulated probabilities are
            consistent with historical predictions.
        p_tie: Per-game tie probability.
        base_seed: Base random seed (each sim uses ``base_seed + sim_idx``).
    """

    n_sims: int = 10_000
    k_factor: float = 20.0
    divisor: float = 480.0
    p_tie: float = 0.01
    base_seed: int = 1337


@dataclass(frozen=True)
class SimPaths:
    """Resolved file paths for simulation inputs and outputs.

    Integrates with gridiron_edge.core.settings and datasets.registry
    rather than accepting raw Path arguments.
    """

    data_cleaned: Path
    data_output: Path
    logo_dir: Path

    @classmethod
    def from_settings(cls) -> SimPaths:
        """Build SimPaths from the package settings (repo_root-relative)."""
        s = get_settings()
        return cls(
            data_cleaned=s.data_cleaned,
            data_output=s.data_output,
            logo_dir=s.repo_root / "data" / "images" / "Team Logos",
        )

    @property
    def schedule_file(self) -> Path:
        """Absolute path to the rich upcoming schedule Parquet."""
        return self.data_cleaned / "NFL_upcoming_schedule_rich.parquet"

    @property
    def wk_by_wk_file(self) -> Path:
        """Absolute path to the cleaned week-by-week historical games CSV."""
        return self.data_cleaned / "NFL_wk_by_wk_cleaned.csv"

    @property
    def mapping_file(self) -> Path:
        """Absolute path to the unified team metadata CSV.

        Contains long/short names, city/name split, conf/div,
        primary/secondary colors. Consumed by
        ``sim.season.load_long_to_short_mapping``, which reads the
        NFL_LONG_NAME + NFL_SHORT_NAME columns.
        """
        return self.data_cleaned / "NFL_team_metadata.csv"

    @property
    def elo_file(self) -> Path:
        """Absolute path to the Elo ratings state table CSV."""
        return self.data_cleaned / "NFL_Team_Elo.csv"

    @property
    def conf_div_file(self) -> Path:
        """Absolute path to the unified team metadata CSV.

        Points to the same file as ``mapping_file``. Consumed by
        ``sim.season.build_conf_div_arrays_from_csv``, which reads
        the conf + div columns.
        """
        return self.data_cleaned / "NFL_team_metadata.csv"

    @property
    def output_temp_dir(self) -> Path:
        """Absolute path to the temporary simulation output directory."""
        return self.data_output / "temp"

    @property
    def output_images_dir(self) -> Path:
        """Absolute path to the output images directory."""
        return self.data_output.parent / "images"

    def validate(self) -> None:
        """Raise FileNotFoundError if any required input file is missing."""
        required = [
            self.schedule_file,
            self.wk_by_wk_file,
            self.mapping_file,
            self.elo_file,
            self.conf_div_file,
        ]
        missing = [f for f in required if not f.exists()]
        if missing:
            msg = f"Missing required files: {', '.join(str(f) for f in missing)}"
            raise FileNotFoundError(msg)
        if not self.logo_dir.exists():
            logger.warning("Logo directory not found: %s", self.logo_dir)


# ============================================================================
# DATA CONTAINERS
# ============================================================================


@dataclass(frozen=True)
class TeamIndex:
    """Mapping between team names and integer indices."""

    short_names: list[str]
    short_to_id: dict[str, int]
    long_to_short: dict[str, str]


@dataclass(frozen=True)
class ScheduleArrays:
    """Numpy arrays representing the season schedule."""

    week: np.ndarray
    home: np.ndarray
    away: np.ndarray
    result: np.ndarray
    week_offsets: np.ndarray


@dataclass(frozen=True)
class SimulationResults:
    """Aggregated results from regular season + playoff simulations."""

    # Playoff simulation outputs
    pts_total_by_sim: np.ndarray  # shape (n_sims, 32) - standings points per sim
    po_win_counts: np.ndarray  # shape (32, 4) - playoff round wins
    make_playoffs_counts: np.ndarray  # shape (32,) - playoff appearances
    bye_counts: np.ndarray  # shape (32,) - first-round byes
    reg_win_counts: np.ndarray  # shape (32, 18) - win count per team per week

    # Actuals through final_actual_week - needed by build_viz_table_df
    pts_total_actual: np.ndarray  # shape (32,) - actual standings points
    gp_played_actual: np.ndarray  # shape (32,) - games played
    gp_total: np.ndarray  # shape (32,) - total scheduled games
    div_id: np.ndarray  # shape (32,) - division id per team


# ============================================================================
# UTILITIES
# ============================================================================


@contextmanager
def _log_phase(name: str) -> Iterator[None]:
    """Log a phase banner + elapsed time."""
    logger.info("-" * 72)
    logger.info("%s...", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("%s complete (%.2fs).", name, dt)


def format_record(pts: int, gp: int) -> str:
    """Convert points and games played to W-L(-T) record string."""
    ties = pts % 2
    wins = (pts - ties) // 2
    losses = gp - wins - ties
    return f"{wins}-{losses}" if ties == 0 else f"{wins}-{losses}-{ties}"
