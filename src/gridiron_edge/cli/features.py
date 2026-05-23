# src/gridiron_edge/cli/features.py
"""CLI commands for feature engineering."""

from __future__ import annotations

import typer

features_app = typer.Typer(help="Build feature tables and modeling matrices.", no_args_is_help=True)


@features_app.command("model-inputs")
def builder_model_inputs(
    *,
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Rebuild all modeling rows vs append new weeks.",
    ),
) -> None:
    """Build modeling base and full feature files."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.features.pipeline import build_model_inputs

    mode = "full rebuild" if all_years else "incremental"
    console.header("features model-inputs", subtitle=mode)

    with step(f"Build model inputs ({mode})"):
        build_model_inputs(all_years=all_years)

    console.summary()
