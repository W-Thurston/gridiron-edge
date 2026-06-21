# src/gridiron_edge/cli/main.py
"""Gridiron Edge CLI — main entrypoint.

Assembles the top-level Typer app from sub-module apps and registers
the run-data-pipeline command. All sub-apps are lazy-loaded via their
own modules so that --help renders quickly and imports only happen when
a command is actually invoked.
"""

from __future__ import annotations

from typing import Annotated

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli.betting import betting_app
from gridiron_edge.cli.edges import edges_app
from gridiron_edge.cli.evaluate import evaluate_app
from gridiron_edge.cli.features import features_app
from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.cli.models import models_app
from gridiron_edge.cli.output import output_app
from gridiron_edge.cli.post_week import post_week_cmd
from gridiron_edge.cli.props import props_app
from gridiron_edge.cli.ratings import ratings_app
from gridiron_edge.cli.sim import sim_app
from gridiron_edge.cli.transform import transform_app
from gridiron_edge.cli.weekly_predict import weekly_predict_cmd
from gridiron_edge.core.logging import setup_logging
from gridiron_edge.core.settings import ensure_data_dirs

_verbose_state: dict[str, bool] = {"verbose": False}


def _cli_startup(
    _ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--no-verbose",
            "-v",
            help="Verbose output: step details, row counts, file paths, and debug logs.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Initialise data directories, logging, and console verbosity."""
    from gridiron_edge.core.console import console

    _verbose_state["verbose"] = verbose
    console.set_verbose(verbose)
    ensure_data_dirs()
    setup_logging(verbose=verbose)


app = typer.Typer(
    name="gridiron",
    help="Gridiron Edge CLI: ingest, transform, features, ratings, output.",
    no_args_is_help=True,
    callback=_cli_startup,
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(transform_app, name="transform")
app.add_typer(features_app, name="features")
app.add_typer(ratings_app, name="ratings")
app.add_typer(output_app, name="output")
app.add_typer(sim_app, name="sim")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(models_app, name="models")
app.add_typer(edges_app, name="edges")
app.add_typer(betting_app, name="bet")
app.add_typer(props_app, name="props")
app.command("weekly-predict")(weekly_predict_cmd)
app.command("post-week")(post_week_cmd)


# ===========================================================================
# FULL PIPELINE
# ===========================================================================

# Canonical stage order for run-data-pipeline.
# Each string is a valid argument to --skip and --only.
ALL_STAGES: list[str] = [
    "fetch-games",
    "clean-games",
    "fetch-upcoming",
    "clean-upcoming",
    "fetch-weather",
    "fetch-odds",
    "build-epa",
    "build-elo",
    "build-features",
]

# ===========================================================================
# STAGE DEPENDENCY CHECKING
# ===========================================================================

# Map of (stage → file path tuple relative to repo root). The first path
# is the stage's input; the second is the stage's output. A warning is
# logged when the input is newer than the output, indicating the stage
# is operating on a result that's already stale relative to its source.
# Only the most common dependencies are tracked here — fetch-* stages
# have no checkable upstream within the pipeline itself.
_STAGE_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "clean-games": (
        "data/raw/games.parquet",
        "data/cleaned/games.csv",
    ),
    "clean-upcoming": (
        "data/raw/schedule_upcoming.parquet",
        "data/cleaned/schedule_upcoming.csv",
    ),
}


def _check_stage_staleness(
    *,
    active: set[str],
) -> None:
    """Warn when a downstream stage will operate on stale upstream data.

    If a downstream stage (e.g. ``clean-games``) is active and the
    upstream input file is newer than the existing output file, log a
    warning. Does not refuse to run — the user may have legitimate
    reasons for re-cleaning.

    Args:
        active: The set of stages that will run.
    """
    import logging
    from pathlib import Path

    from gridiron_edge.core.settings import get_settings

    repo_root: Path = get_settings().repo_root
    logger = logging.getLogger(__name__)

    for stage in active:
        if stage not in _STAGE_DEPENDENCIES:
            continue
        rel_input, rel_output = _STAGE_DEPENDENCIES[stage]
        input_path: Path = repo_root / rel_input
        output_path: Path = repo_root / rel_output

        if not input_path.exists():
            # No upstream input → no check possible. The stage may
            # produce its first run; downstream code will surface
            # errors if input is genuinely required.
            continue
        if not output_path.exists():
            # First run — nothing to compare against.
            continue

        input_mtime: float = input_path.stat().st_mtime
        output_mtime: float = output_path.stat().st_mtime
        if input_mtime > output_mtime:
            logger.warning(
                "Stage %r will run on upstream data older than its current "
                "output (%s newer than %s). Consider running upstream "
                "fetch stages first to refresh.",
                stage,
                rel_input,
                rel_output,
            )


_STAGES_STR: str = ", ".join(ALL_STAGES)
_SKIP_HELP: str = f"Stage(s) to skip. Repeatable. Valid stages: {_STAGES_STR}."
_ONLY_HELP: str = (
    f"Run only these stage(s) and skip all others. Repeatable. Valid stages: {_STAGES_STR}."
)


def _run_pipeline_stages(  # noqa: PLR0912, PLR0915
    *,
    active: set[str],
    all_years: bool,
    resolved_season: int,
    upcoming_target: int,
    season: int | None,
    season_year: str | None,
    owm_api_key: str | None,
    fit_elo_all_years: bool,
) -> None:
    """Execute each pipeline stage in order for the stages in ``active``.

    Separated from the CLI command function to keep branch and statement
    counts within ruff limits (PLR0912, PLR0915).
    """
    from pathlib import Path

    from gridiron_edge.core.console import step

    _check_stage_staleness(active=active)

    def runs(stage: str) -> bool:
        return stage in active

    with step("Fetch nflverse games", skip=not runs("fetch-games")) as s:
        if runs("fetch-games"):
            from gridiron_edge.ingest.nflverse import (
                fetch_nflverse_games,
                fetch_nflverse_games_refresh,
            )

            if all_years:
                path: Path = fetch_nflverse_games()
            elif season:
                path = fetch_nflverse_games(seasons=[resolved_season])
            else:
                path = fetch_nflverse_games_refresh()
            s.set_detail(path.name)

    with step("Clean games", skip=not runs("clean-games")) as s:
        if runs("clean-games"):
            # pyrefly: ignore [missing-module-attribute]
            from gridiron_edge.transform.clean import clean_nflverse_games

            path = clean_nflverse_games()
            s.set_detail(path.name)

    with step("Fetch upcoming schedule", skip=not runs("fetch-upcoming")) as s:
        if runs("fetch-upcoming"):
            from gridiron_edge.ingest.nflverse import fetch_nflverse_upcoming

            path = fetch_nflverse_upcoming(season=upcoming_target)
            s.set_detail(path.name)

    with step("Clean upcoming schedule", skip=not runs("clean-upcoming")) as s:
        if runs("clean-upcoming"):
            # pyrefly: ignore [missing-module-attribute]
            from gridiron_edge.transform.clean import clean_nflverse_upcoming

            path = clean_nflverse_upcoming()
            s.set_detail(path.name)

    with step("Fetch weather", skip=not runs("fetch-weather")) as s:
        if runs("fetch-weather"):
            from gridiron_edge.cli._shared import get_owm_api_key
            from gridiron_edge.ingest.weather import fetch_weather

            if not season_year:
                raise ValueError(
                    "fetch-weather requires --season-year (e.g. '2025-2026').",
                )
            key: str = get_owm_api_key(owm_api_key)
            fetch_weather(season_year=season_year, owm_api_key=key)
            s.set_detail(season_year)

    with step("Fetch DraftKings odds", skip=not runs("fetch-odds")):
        if runs("fetch-odds"):
            from gridiron_edge.ingest.odds import fetch_dk_odds

            fetch_dk_odds()

    with step("Build EPA features", skip=not runs("build-epa")) as s:
        if runs("build-epa"):
            from gridiron_edge.ingest.nflverse.pbp import fetch_pbp, fetch_pbp_refresh
            from gridiron_edge.transform.clean.epa import aggregate_epa

            if all_years:
                fetch_pbp()
            else:
                fetch_pbp_refresh()
            aggregate_epa()
            s.set_detail("done")

    with step("Fit Elo", skip=not runs("build-elo")) as s:
        if runs("build-elo"):
            # pyrefly: ignore [missing-module-attribute]
            from gridiron_edge.ratings.elo import fit_elo

            fit_elo(all_years=fit_elo_all_years)
            s.set_detail("full rebuild" if fit_elo_all_years else "incremental")

    with step("Build model inputs", skip=not runs("build-features")):
        if runs("build-features"):
            from gridiron_edge.features.pipeline import build_model_inputs

            build_model_inputs(all_years=all_years)


@app.command("run-data-pipeline")
def run_data_pipeline(
    *,
    season: int | None = typer.Option(
        None,
        help="Season year (e.g. 2025). Defaults to current season.",
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Fetch/build full history instead of current season only.",
    ),
    upcoming_season: int | None = typer.Option(
        None,
        help=(
            "Season to fetch upcoming schedule for. Defaults to --season or current "
            "season. Use when fetching upcoming games from a different season than "
            "completed games (e.g. --all-years --upcoming-season 2026)."
        ),
    ),
    fit_elo_all_years: bool = typer.Option(
        False,
        "--fit-elo-all-years/--no-fit-elo-all-years",
        help="When build-elo runs, rebuild full Elo history rather than incrementally.",
    ),
    season_year: str | None = typer.Option(
        None,
        help="Required when fetch-weather is active (e.g. '2025-2026').",
    ),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key or env var OWM_API_KEY.",
    ),
    skip: list[str] = typer.Option([], "--skip", help=_SKIP_HELP),  # noqa: B008
    only: list[str] = typer.Option([], "--only", help=_ONLY_HELP),  # noqa: B008
) -> None:
    r"""Run a full end-to-end data pipeline with per-stage control.

    \b
    Scenario 1 — weekly refresh (most common, no flags needed):
      gridiron run-data-pipeline

    \b
    Scenario 2 — specific season:
      gridiron run-data-pipeline --season 2025

    \b
    Scenario 3 — full history rebuild:
      gridiron run-data-pipeline --all-years --upcoming-season 2026 \
        --only build-elo --fit-elo-all-years

    \b
    Scenario 4 — skip weather and odds:
      gridiron run-data-pipeline --skip fetch-weather --skip fetch-odds

    \b
    Scenario 5 — features only:
      gridiron run-data-pipeline --only build-features
    """
    from gridiron_edge.core.console import console
    from gridiron_edge.core.settings import current_nfl_season

    # Validate stage names
    unknown = set(skip + only) - set(ALL_STAGES)
    if unknown:
        raise typer.BadParameter(
            f"Unknown stage(s): {', '.join(sorted(unknown))}. Valid stages: {_STAGES_STR}."
        )
    if skip and only:
        raise typer.BadParameter("--skip and --only are mutually exclusive.")

    active: set[str] = set(only) if only else set(ALL_STAGES) - set(skip)
    resolved_season: int = season or current_nfl_season()
    upcoming_target: int = upcoming_season or resolved_season

    if all_years and upcoming_season:
        mode: str = f"full history  \u00b7  upcoming season {upcoming_season}"
    elif all_years:
        mode = "full history rebuild"
    elif season:
        mode = f"season {resolved_season}"
    else:
        mode = f"weekly refresh  \u00b7  season {resolved_season}"

    console.header("run-data-pipeline", subtitle=mode)

    _run_pipeline_stages(
        active=active,
        all_years=all_years,
        resolved_season=resolved_season,
        upcoming_target=upcoming_target,
        season=season,
        season_year=season_year,
        owm_api_key=owm_api_key,
        fit_elo_all_years=fit_elo_all_years,
    )

    console.summary()


# ===========================================================================
# ENTRYPOINT
# ===========================================================================


def main() -> None:
    """Entry point for the Gridiron Edge CLI."""
    app()


if __name__ == "__main__":
    main()
