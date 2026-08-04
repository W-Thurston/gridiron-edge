# src/gridiron_edge/cli/edges.py
"""CLI commands for betting edge analysis.

Provides two sub-commands:

    gridiron edges report   Weekly edge report (the Sunday artifact)
    gridiron edges clv      Historical closing line value analysis

The weekly report consumes the explicitly selected persisted weekly
product. Historical CLV analysis retains its explicit win-model selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import resolve_win_prob_model_type
from gridiron_edge.core.settings import Settings
from gridiron_edge.models.game_prediction.post_process import get_total_std

if TYPE_CHECKING:
    from gridiron_edge.market.edge_diagnostics import EdgeDiagnostics

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
    *,
    week: int = typer.Option(..., help="NFL week number."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
    bankroll: float | None = typer.Option(
        None,
        help="Optional bankroll in dollars for Kelly stake sizing.",
    ),
    kelly_multiplier: float = typer.Option(
        0.25,
        help="Fraction of full Kelly (e.g. 0.25 for quarter-Kelly).",
    ),
    min_ev: float = typer.Option(
        0.0,
        help="Minimum EV threshold for display (e.g. 0.02 for 2%%).",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format: 'table' or 'csv'.",
    ),
) -> None:
    """Generate a weekly betting edge report from the selected product."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.market.edge_diagnostics import EdgeResultState
    from gridiron_edge.market.weekly_edge_service import (
        build_weekly_edge_result,
    )

    if output_format not in {"table", "csv"}:
        raise typer.BadParameter(
            "Output format must be 'table' or 'csv'.",
            param_hint="--format",
        )
    if bankroll is not None and bankroll < 0.0:
        raise typer.BadParameter(
            "Bankroll must be greater than or equal to 0.",
            param_hint="--bankroll",
        )
    if not 0.0 <= kelly_multiplier <= 1.0:
        raise typer.BadParameter(
            "Kelly multiplier must be in [0, 1].",
            param_hint="--kelly-multiplier",
        )
    if min_ev < 0.0:
        raise typer.BadParameter(
            "Minimum EV must be greater than or equal to 0.",
            param_hint="--min-ev",
        )

    console.header(f"Edge Report - {season} Week {week}")

    with step("Building weekly edge result"):
        result = build_weekly_edge_result(
            season=season,
            week=week,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
            min_ev=min_ev,
        )

    _remove_edge_csv(season, week)

    if result.rows.empty:
        typer.echo(_edge_result_message(result.diagnostics, min_ev=min_ev))
        if result.diagnostics.state is EdgeResultState.BLOCKED:
            raise typer.Exit(code=1)
        return

    if output_format == "csv":
        _write_csv(result.rows, season, week)
    else:
        _render_edge_table(result.rows)

    typer.echo(f"\n  {len(result.rows)} edge(s) found.")


# ---------------------------------------------------------------------------
# gridiron edges clv
# ---------------------------------------------------------------------------


@edges_app.command()
def clv(
    season: str | None = typer.Option(None, help="Filter to NFL season label, e.g. '2026-2027'."),
    model_type: str = typer.Option(
        "auto",
        help=(
            "Win-probability model algorithm to use. One of: random_forest, "
            "xgboost, logistic, elo. Defaults to 'auto', which resolves to "
            "the current champion from the manifest at "
            "data/output/champions/champions.json."
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

    resolved_model_type = resolve_win_prob_model_type(model_type)

    console.header(f"Closing Line Value Analysis  ·  model={resolved_model_type}")

    # ── Load data ─────────────────────────────────────────────────────
    with step("Loading predictions"):
        predictions: DataFrame = load_prediction_log(
            season=season,
            model_name="win_prob",
            model_type=resolved_model_type,
        )
    if predictions.empty:
        typer.echo(f"No predictions found for win_prob/{resolved_model_type}.")
        raise typer.Exit()

    typer.echo(f"  {len(predictions)} prediction(s) loaded.")

    with step("Loading odds ledger"):
        odds_ledger: DataFrame = load_odds_ledger(season=season)
    if odds_ledger.empty:
        typer.echo("No odds ledger data available.")
        raise typer.Exit()

    typer.echo(f"  {len(odds_ledger)} odds row(s) loaded.")

    # ── Build edge report for historical games ────────────────────────
    margin_std: float = get_margin_std("win_prob", resolved_model_type)
    total_std: float = get_total_std(
        "total",
        resolved_model_type,
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


def _edge_result_message(
    diagnostics: EdgeDiagnostics,
    *,
    min_ev: float,
) -> str:
    """Render one deterministic explanation for an empty edge result."""
    from gridiron_edge.market.edge_diagnostics import (
        EdgeDiagnosticBlocker,
        EdgeResultState,
    )

    blocker_messages = {
        EdgeDiagnosticBlocker.NO_PREDICTIONS: (
            "No current weekly product is selected for the requested season and week."
        ),
        EdgeDiagnosticBlocker.NO_MARKET_DATA: ("No current market snapshot is available."),
        EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE: (
            "The current market snapshot does not contain the requested season and week."
        ),
        EdgeDiagnosticBlocker.MARKET_STALE: (
            "The current market snapshot is stale under the supplied freshness policy."
        ),
        EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES: (
            "The weekly product and market snapshot have no matching game IDs."
        ),
        EdgeDiagnosticBlocker.INCOMPLETE_MARKETS: (
            "Matching games exist, but one or more market families are incomplete."
        ),
    }
    if diagnostics.blockers:
        return " ".join(blocker_messages[blocker] for blocker in diagnostics.blockers)
    if diagnostics.state is EdgeResultState.NO_CALCULABLE_EDGES:
        return "No calculable edges were produced from the available inputs."
    if diagnostics.state is EdgeResultState.NO_POSITIVE_EDGES:
        return "Calculated markets contained no positive expected-value edges."
    if diagnostics.state is EdgeResultState.POSITIVE_EDGES and diagnostics.filtered_edge_count == 0:
        return f"Positive edges were calculated, but none exceeded min_ev={min_ev:.1%}."
    return "No edge rows are available for the requested scope."


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

        kelly_value = row.get("kelly_stake")
        # pyrefly: ignore [bad-argument-type]
        kelly_str = "—" if pd.isna(kelly_value) else f"${float(kelly_value):.2f}"

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


def _edge_csv_path(season: str, week: int) -> Path:
    """Return the scope-specific standalone edge-report path."""
    from gridiron_edge.core.settings import get_settings

    settings: Settings = get_settings()
    return settings.data_output / "edges" / f"edges_{season}_wk{week:02d}.csv"


def _remove_edge_csv(season: str, week: int) -> None:
    """Remove a prior CSV so it cannot be mistaken for the current result."""
    _edge_csv_path(season, week).unlink(missing_ok=True)


def _write_csv(
    ranked_df: DataFrame,
    season: str,
    week: int,
) -> None:
    """Write ranked edges to a CSV file."""
    path = _edge_csv_path(season, week)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(path, index=False)
    typer.echo(f"  Saved to {path}")
