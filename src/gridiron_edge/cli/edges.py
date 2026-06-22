# src/gridiron_edge/cli/edges.py
"""CLI commands for betting edge analysis.

Provides two sub-commands:

    gridiron edges report   Weekly edge report (the Sunday artifact)
    gridiron edges clv      Historical closing line value analysis

Workstream 2 convention:
    These commands operate on win_prob predictions. The ``--model-type``
    option selects the algorithm (``random_forest``, ``xgboost``,
    ``logistic``, ``elo``); ``model_name="win_prob"`` is implied.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.core.settings import Settings
from gridiron_edge.models.game_prediction.post_process import get_total_std

edges_app = typer.Typer(help="Betting edge analysis.", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# Fallback total_std used when no trained total model artifact exists
# for the requested model_type (e.g. when the win_prob model is elo and
# there is no matching total_elo artifact).
_TOTAL_STD_FALLBACK: float = 13.0


# ---------------------------------------------------------------------------
# gridiron edges report
# ---------------------------------------------------------------------------


@edges_app.command()
def report(
    week: int = typer.Option(..., help="NFL week number."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
    model_type: str = typer.Option(
        "random_forest",
        help=(
            "Win-probability model algorithm to use. One of: random_forest, xgboost, logistic, elo."
        ),
    ),
    bankroll: float = typer.Option(1000.0, help="Current bankroll in dollars."),
    kelly_multiplier: float = typer.Option(
        0.25, help="Fraction of full Kelly (e.g. 0.25 for quarter-Kelly)."
    ),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for display (e.g. 0.02 for 2%%)."),
    output_format: str = typer.Option("table", "--format", help="Output format: 'table' or 'csv'."),
) -> None:
    """Generate a weekly betting edge report."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.ingest.odds.store import load_current_odds
    from gridiron_edge.market.recommendations import build_edge_report, rank_edges
    from gridiron_edge.models.game_prediction.post_process import get_margin_std

    console.header(f"Edge Report - {season} Week {week}")

    # ── Load predictions ──────────────────────────────────────────────
    with step("Loading predictions"):
        predictions: DataFrame = load_prediction_log(
            season=season,
            week=week,
            model_name="win_prob",
            model_type=model_type,
        )
    if predictions.empty:
        typer.echo(f"No predictions found for win_prob/{model_type} / {season} / week {week}.")
        raise typer.Exit()

    typer.echo(f"  {len(predictions)} prediction(s) loaded.")

    # ── Load odds ─────────────────────────────────────────────────────
    with step("Loading current odds"):
        odds: DataFrame | None = load_current_odds()
    if odds is None or odds.empty:
        typer.echo("No current odds available. Run 'gridiron ingest fetch-odds' first.")
        raise typer.Exit()

    typer.echo(f"  {len(odds)} odds row(s) loaded.")

    # ── Build edge report ─────────────────────────────────────────────
    margin_std: float = get_margin_std("win_prob", model_type)
    total_std: float = get_total_std(
        "total",
        model_type,
        default=_TOTAL_STD_FALLBACK,
    )

    with step("Computing edges"):
        edge_report: DataFrame = build_edge_report(
            predictions,
            odds,
            margin_std=margin_std,
            total_std=total_std,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )

    if edge_report.empty:
        typer.echo("No edges found (predictions did not match any odds).")
        raise typer.Exit()

    # ── Rank and filter ───────────────────────────────────────────────
    with step("Ranking edges"):
        ranked: DataFrame = rank_edges(edge_report, min_ev=min_ev)

    if ranked.empty:
        typer.echo(f"No edges above min_ev={min_ev:.1%}.")
        raise typer.Exit()

    # ── Output ────────────────────────────────────────────────────────
    if output_format == "csv":
        _write_csv(ranked, season, week)
    else:
        _render_edge_table(ranked)

    typer.echo(f"\n  {len(ranked)} edge(s) found.")


# ---------------------------------------------------------------------------
# gridiron edges clv
# ---------------------------------------------------------------------------


@edges_app.command()
def clv(
    season: str | None = typer.Option(None, help="Filter to NFL season label, e.g. '2026-2027'."),
    model_type: str = typer.Option(
        "random_forest",
        help=(
            "Win-probability model algorithm to use. One of: random_forest, xgboost, logistic, elo."
        ),
    ),
    min_ev: float = typer.Option(0.0, help="Minimum EV threshold for edges to include."),
) -> None:
    """Analyse historical closing line value."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.ingest.odds.store import load_odds_ledger
    from gridiron_edge.market.clv import build_clv_report, summarize_clv
    from gridiron_edge.market.recommendations import build_edge_report, rank_edges
    from gridiron_edge.models.game_prediction.post_process import get_margin_std

    console.header("Closing Line Value Analysis")

    # ── Load data ─────────────────────────────────────────────────────
    with step("Loading predictions"):
        predictions: DataFrame = load_prediction_log(
            season=season,
            model_name="win_prob",
            model_type=model_type,
        )
    if predictions.empty:
        typer.echo(f"No predictions found for win_prob/{model_type}.")
        raise typer.Exit()

    typer.echo(f"  {len(predictions)} prediction(s) loaded.")

    with step("Loading odds ledger"):
        odds_ledger: DataFrame = load_odds_ledger(season=season)
    if odds_ledger.empty:
        typer.echo("No odds ledger data available.")
        raise typer.Exit()

    typer.echo(f"  {len(odds_ledger)} odds row(s) loaded.")

    # ── Build edge report for historical games ────────────────────────
    margin_std: float = get_margin_std("win_prob", model_type)
    total_std: float = get_total_std(
        "total",
        model_type,
        default=_TOTAL_STD_FALLBACK,
    )

    with step("Building edge report"):
        edge_report: DataFrame = build_edge_report(
            predictions,
            odds_ledger,
            margin_std=margin_std,
            total_std=total_std,
        )
        ranked: DataFrame = rank_edges(edge_report, min_ev=min_ev)

    if ranked.empty:
        typer.echo("No positive-EV edges found in historical data.")
        raise typer.Exit()

    # ── Compute CLV ───────────────────────────────────────────────────
    with step("Computing CLV"):
        clv_report: DataFrame = build_clv_report(ranked, odds_ledger)
        stats: dict[str, float] = summarize_clv(clv_report)

    # ── Display summary ───────────────────────────────────────────────
    _render_clv_summary(stats)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_edge_table(ranked_df: DataFrame) -> None:
    """Render a rich console table of ranked edges."""
    # pyrefly: ignore [missing-import]
    from rich.console import Console

    # pyrefly: ignore [missing-import]
    from rich.table import Table

    console = Console()
    table = Table(
        title="Betting Edges (ranked by EV)",
        show_lines=True,
        header_style="bold cyan",
    )

    table.add_column("Game", style="dim")
    table.add_column("Away")
    table.add_column("Home")
    table.add_column("Market")
    table.add_column("Side")
    table.add_column("EV", justify="right")
    table.add_column("Strength")
    table.add_column("Kelly $", justify="right")
    table.add_column("Conf. Tier")

    max_rows = 20
    for i, (_, row) in enumerate(ranked_df.iterrows()):
        if i >= max_rows:
            break

        ev_pct: str = f"{row['ev']:.1%}"
        strength: str = row.get("edge_strength", "")
        if strength == "strong":
            ev_style = "[bold green]"
        elif strength == "moderate":
            ev_style = "[yellow]"
        else:
            ev_style = "[dim]"

        kelly_str: str = f"${row.get('kelly_stake', 0):.2f}"

        game_id: str = row.get("game_id", "")
        short_game: str = game_id.split("_", 1)[-1] if "_" in game_id else game_id

        table.add_row(
            short_game,
            str(row.get("away_team", "")),
            str(row.get("home_team", "")),
            str(row.get("market_type", "")),
            str(row.get("side", "")),
            f"{ev_style}{ev_pct}[/]",
            strength,
            kelly_str,
            str(row.get("confidence_tier", "")),
        )

    console.print(table)

    total: int = len(ranked_df)
    if total > max_rows:
        console.print(
            f"  [dim]... and {total - max_rows} more edge(s). "
            f"Use --format csv for full output.[/dim]"
        )


def _render_clv_summary(stats: dict[str, float]) -> None:
    """Render CLV summary statistics."""
    import math

    typer.echo("\n  CLV Summary")
    typer.echo("  " + "-" * 35)

    n: float = stats.get("n_edges", 0)
    if n == 0 or math.isnan(stats.get("mean_clv", float("nan"))):
        typer.echo("  No CLV data available.")
        return

    typer.echo(f"  Edges analysed:    {int(n)}")
    typer.echo(f"  Mean CLV:          {stats['mean_clv']:+.3f}")
    typer.echo(f"  Median CLV:        {stats['median_clv']:+.3f}")
    typer.echo(f"  % Positive CLV:    {stats['pct_positive_clv']:.1%}")


def _write_csv(
    ranked_df: DataFrame,
    season: str,
    week: int,
) -> None:
    """Write ranked edges to a CSV file."""
    from gridiron_edge.core.settings import get_settings

    settings: Settings = get_settings()
    out_dir: Path = settings.data_output / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename: str = f"edges_{season}_wk{week:02d}.csv"
    path: Path = out_dir / filename
    ranked_df.to_csv(path, index=False)
    typer.echo(f"  Saved to {path}")
