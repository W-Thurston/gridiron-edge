"""Shared orchestration for the standard operational simulation."""

from __future__ import annotations

from gridiron_edge.sim import (
    SimPaths,
    SimulationConfig,
    run_full_simulation,
)

DEFAULT_N_SIMS = 10_000
DEFAULT_K_FACTOR = 20.0
DEFAULT_DIVISOR = 480.0
DEFAULT_P_TIE = 0.01
DEFAULT_SEED = 1337


def build_standard_simulation_config() -> SimulationConfig:
    """Return the standard operational season-simulation configuration."""
    return SimulationConfig(
        n_sims=DEFAULT_N_SIMS,
        k_factor=DEFAULT_K_FACTOR,
        divisor=DEFAULT_DIVISOR,
        p_tie=DEFAULT_P_TIE,
        base_seed=DEFAULT_SEED,
    )


def run_standard_simulation(
    *,
    render: bool,
) -> None:
    """Run the standard season simulation and persist all derived outputs."""
    run_full_simulation(
        paths=SimPaths.from_settings(),
        config=build_standard_simulation_config(),
        render=render,
    )
