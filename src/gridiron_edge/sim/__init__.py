# src/gridiron_edge/sim/__init__.py

"""NFL Season Monte Carlo Simulation package.

Public API — import from here rather than from submodules directly.
"""

from gridiron_edge.sim._types import (
    SimPaths,
    SimulationConfig,
    SimulationResults,
    TeamIndex,
    format_record,
)
from gridiron_edge.sim.season import run_full_simulation

__all__: list[str] = [
    "SimPaths",
    "SimulationConfig",
    "SimulationResults",
    "TeamIndex",
    "format_record",
    "run_full_simulation",
]
