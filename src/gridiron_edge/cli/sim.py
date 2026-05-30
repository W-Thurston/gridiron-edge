# src/gridiron_edge/cli/sim.py
"""CLI commands for season simulation."""

from __future__ import annotations

import typer

sim_app = typer.Typer(help="Monte Carlo season + playoff simulation.", no_args_is_help=True)


@sim_app.command("run")
def sim_run(
    *,
    n_sims: int = typer.Option(10_000, help="Number of Monte Carlo simulations."),
    # UPDATE: set this to the tuned K optimum after running 'gridiron evaluate tune elo'.
    # elo_v2 grid search found K=20 winning; verify after each season's re-tune.
    k_factor: float = typer.Option(20.0, help="Elo K-factor."),
    # UPDATE: set this to the tuned divisor after running 'gridiron evaluate tune elo'.
    # elo_v2 found 350 winning; elo_v1 default is 480.
    divisor: float = typer.Option(480.0, help="Elo win-probability divisor."),
    p_tie: float = typer.Option(0.01, help="Probability of a tie game."),
    seed: int = typer.Option(1337, help="Base random seed."),
    render: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Render playoff probability table PNG after simulation.",
    ),
) -> None:
    """Run Monte Carlo season + playoff simulation, write CSVs, and optionally render viz."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.sim import SimPaths, SimulationConfig, run_full_simulation

    config = SimulationConfig(
        n_sims=n_sims,
        k_factor=k_factor,
        divisor=divisor,
        p_tie=p_tie,
        base_seed=seed,
    )
    subtitle = f"{n_sims:,} simulations  ·  seed {seed}"
    if render:
        subtitle += "  ·  +render"
    console.header("sim run", subtitle=subtitle)

    with step(f"Simulate season + playoffs ({n_sims:,} sims)"):
        paths = SimPaths.from_settings()
        run_full_simulation(paths=paths, config=config, render=render)

    console.summary()
