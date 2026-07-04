"""CLI commands for season simulation."""

from __future__ import annotations

import typer

sim_app = typer.Typer(help="Monte Carlo season + playoff simulation.", no_args_is_help=True)


@sim_app.command("run")
def sim_run(
    *,
    n_sims: int = typer.Option(10_000, help="Number of Monte Carlo simulations."),
    # UPDATE: set this to the tuned K optimum after running 'gridiron evaluate tune elo'.
    # flat-K grid search found K=20 winning; verify after each season's re-tune.
    k_factor: float = typer.Option(20.0, help="Elo K-factor."),
    # UPDATE: set this to the tuned divisor after running 'gridiron evaluate tune elo'.
    # flat-K found 350 winning; Elo's original default is 480.
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


@sim_app.command("compute-percentiles")
def sim_compute_percentiles() -> None:
    """Compute team percentile rankings from existing Elo state and projections.

    Reads:
        data/cleaned/NFL_Team_Elo.csv
        data/output/temp/projections_summary.csv

    Writes:
        data/output/rankings/percentiles/percentiles_{season}_wk{NN}.parquet

    Standalone alternative to `gridiron sim run` when only percentiles need
    to be refreshed (e.g., after tweaking Elo state without re-running the
    full simulation).
    """
    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_teams_long_short
    from gridiron_edge.evaluation.percentiles import (
        compute_team_percentiles,
        write_team_percentiles,
    )

    settings = get_settings()
    repo = settings.repo_root

    console.header("sim compute-percentiles")

    with step("Load Elo state and projections") as s:
        elo_path = repo / "data" / "cleaned" / "NFL_Team_Elo.csv"
        proj_path = repo / "data" / "output" / "temp" / "projections_summary.csv"

        if not elo_path.exists():
            typer.echo(f"Elo state not found at {elo_path}")
            raise typer.Exit(1)
        if not proj_path.exists():
            typer.echo(f"Projections not found at {proj_path}")
            typer.echo("Run `gridiron sim run` first.")
            raise typer.Exit(1)

        elo = pd.read_csv(elo_path)
        proj = pd.read_csv(proj_path)
        mapping_df = load_teams_long_short(repo)
        long_to_short = dict(
            zip(
                mapping_df["NFL_LONG_NAME"],
                mapping_df["NFL_SHORT_NAME"],
                strict=True,
            )
        )

        s.set_detail(f"{len(elo)} elo rows, {len(proj)} projection rows")

    with step("Compute percentiles") as s:
        df = compute_team_percentiles(
            elo_state=elo,
            projections=proj,
            long_to_short=long_to_short,
        )
        s.set_detail(f"{len(df)} teams ranked")

    if df.empty:
        typer.echo("No percentiles computed (empty inputs).")
        raise typer.Exit(1)

    with step("Persist artifact") as s:
        season = str(df.iloc[0]["season"])
        week = int(df.iloc[0]["week"])
        path = write_team_percentiles(df, season=season, week=week, repo=repo)
        s.set_detail(str(path))

    console.summary()
