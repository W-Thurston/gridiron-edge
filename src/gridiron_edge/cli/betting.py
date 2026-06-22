# src/gridiron_edge/cli/betting.py
"""CLI commands for bet tracking and performance.

Wires together ``betting/ledger.py``, ``betting/bankroll.py``, and
``betting/performance.py``.  The CLI orchestrates the decoupled modules:
bet placement calls both ``log_bet`` and ``record_bet_placed``; settlement
calls both ``settle_bet`` and ``record_bet_settled``.

Registered as ``gridiron bet`` in ``cli/main.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

betting_app = typer.Typer(help="Bet tracking and performance.", no_args_is_help=True)


# ---------------------------------------------------------------------------
# gridiron bet log
# ---------------------------------------------------------------------------


@betting_app.command("log")
def log_cmd(
    game_id: str = typer.Option(..., help="Canonical game ID, e.g. 2026_01_KC_LAC"),
    market: str = typer.Option(..., help="Market type: moneyline, spread, or total"),
    side: str = typer.Option(..., help="Bet side: home, away, over, or under"),
    odds: int = typer.Option(..., help="American odds at bet time, e.g. -110"),
    stake: float = typer.Option(..., help="Dollar amount wagered"),
    book: str = typer.Option(..., help="Sportsbook name, e.g. draftkings"),
    line: float | None = typer.Option(None, help="Spread or total line (omit for ML)"),
    model_name: str | None = typer.Option(
        None,
        help="Model purpose used to identify the edge, e.g. win_prob or qb_pass_yards",
    ),
    model_type: str | None = typer.Option(
        None,
        help="Algorithm used to compute the edge, e.g. random_forest or elasticnet",
    ),
    model_prob: float | None = typer.Option(None, help="Model probability at bet time"),
    model_ev: float | None = typer.Option(None, help="Model EV at bet time"),
    edge_strength: str | None = typer.Option(None, help="Edge classification"),
    confidence_tier: str | None = typer.Option(None, help="Confidence tier"),
) -> None:
    """Record a new bet."""
    from gridiron_edge.betting.bankroll import current_balance, record_bet_placed
    from gridiron_edge.betting.ledger import log_bet

    bet_id: str = log_bet(
        game_id=game_id,
        market_type=market,
        side=side,
        odds=odds,
        stake=stake,
        book=book,
        line=line,
        model_name=model_name,
        model_type=model_type,
        model_prob=model_prob,
        model_ev=model_ev,
        edge_strength=edge_strength,
        confidence_tier=confidence_tier,
    )
    record_bet_placed(stake, bet_id=bet_id)

    balance: float = current_balance()
    typer.echo(f"Bet logged: {bet_id}")
    typer.echo(f"  {market} {side} {game_id} @ {odds:+d}  stake=${stake:.2f}  book={book}")
    typer.echo(f"  Balance: ${balance:.2f}")


# ---------------------------------------------------------------------------
# gridiron bet settle
# ---------------------------------------------------------------------------


@betting_app.command("settle")
def settle_cmd(
    bet_id: str = typer.Argument(..., help="UUID of the bet to settle"),
    result: str = typer.Argument(..., help="Result: won, lost, or push"),
    with_clv: bool = typer.Option(True, "--with-clv/--no-clv", help="Compute CLV from odds ledger"),
) -> None:
    """Settle an open bet."""
    import pandas as pd

    from gridiron_edge.betting.bankroll import current_balance, record_bet_settled
    from gridiron_edge.betting.ledger import settle_bet

    odds_ledger: pd.DataFrame | None = None
    if with_clv:
        try:
            from gridiron_edge.ingest.odds.store import load_odds_ledger

            odds_ledger = load_odds_ledger()
        except Exception:
            odds_ledger = None

    try:
        row: Series = settle_bet(bet_id, result, odds_ledger=odds_ledger)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    pnl = float(row["pnl"])
    stake = float(row["stake"])
    record_bet_settled(stake, pnl, bet_id=bet_id)

    balance: float = current_balance()
    pnl_str: str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    typer.echo(f"Settled: {bet_id} -> {result}")
    typer.echo(f"  PnL: {pnl_str}  Balance: ${balance:.2f}")


# ---------------------------------------------------------------------------
# gridiron bet list
# ---------------------------------------------------------------------------


@betting_app.command("list")
def list_cmd(
    status: str | None = typer.Option(None, help="Filter by status: open, won, lost, push"),
    market: str | None = typer.Option(None, help="Filter by market type"),
    limit: int = typer.Option(20, help="Max rows to display"),
) -> None:
    """Show bets from the ledger."""
    from gridiron_edge.betting.ledger import load_bets

    df: DataFrame = load_bets(status=status, market_type=market)
    if df.empty:
        typer.echo("No bets found.")
        return

    total: int = len(df)
    df = df.head(limit)
    _render_bet_table(df)
    if total > limit:
        typer.echo(f"  ... showing {limit} of {total} bets")
    else:
        typer.echo(f"  {total} bet(s)")


# ---------------------------------------------------------------------------
# gridiron bet summary
# ---------------------------------------------------------------------------


@betting_app.command("summary")
def summary_cmd(
    split_by: str | None = typer.Option(None, help="Split record/ROI by column"),
) -> None:
    """Show performance dashboard."""
    from gridiron_edge.betting.ledger import load_bets
    from gridiron_edge.betting.performance import record, roi, summary

    bets: DataFrame = load_bets()
    if bets.empty:
        typer.echo("No bets to summarise.")
        return

    stats: dict[str, Any] = summary(bets)
    _render_summary(stats)

    if split_by:
        typer.echo(f"\n  Record by {split_by}:")
        rec_df: DataFrame = record(bets, split_by=split_by)
        if not rec_df.empty:
            typer.echo(rec_df.to_string(index=False))

        typer.echo(f"\n  ROI by {split_by}:")
        roi_df: DataFrame = roi(bets, split_by=split_by)
        if not roi_df.empty:
            typer.echo(roi_df.to_string(index=False))


# ---------------------------------------------------------------------------
# gridiron bet balance
# ---------------------------------------------------------------------------


@betting_app.command("balance")
def balance_cmd(
    limit: int = typer.Option(10, help="Number of recent transactions to show"),
) -> None:
    """Show current bankroll balance and recent transactions."""
    from gridiron_edge.betting.bankroll import balance_history, current_balance

    balance: float = current_balance()
    typer.echo(f"  Current balance: ${balance:.2f}")

    history: DataFrame = balance_history()
    if history.empty:
        typer.echo("  No transactions yet.")
        return

    recent: DataFrame = history.tail(limit)
    typer.echo(f"\n  Recent transactions (last {min(limit, len(history))}):")
    for _, row in recent.iterrows():
        sign: Literal["-", "+"] = "+" if row["signed_amount"] >= 0 else "-"
        typer.echo(
            f"    {row['timestamp']:%Y-%m-%d %H:%M}  "
            f"{row['txn_type']:12s}  "
            f"{sign}${abs(row['signed_amount']):.2f}  "
            f"bal=${row['running_balance']:.2f}"
        )


# ---------------------------------------------------------------------------
# gridiron bet export
# ---------------------------------------------------------------------------


@betting_app.command("export")
def export_cmd(
    status: str | None = typer.Option(None, help="Filter by status"),
    output: str | None = typer.Option(None, help="Output CSV path"),
) -> None:
    """Export bets to CSV."""
    from gridiron_edge.betting.ledger import load_bets
    from gridiron_edge.core.settings import get_settings

    df: DataFrame = load_bets(status=status)
    if df.empty:
        typer.echo("No bets to export.")
        return

    if output is None:
        out_dir: Path = get_settings().repo_root / "data" / "output" / "bets"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path: Path = out_dir / "bets_export.csv"
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    typer.echo(f"Exported {len(df)} bet(s) to {out_path}")


# ---------------------------------------------------------------------------
# gridiron bet deposit
# ---------------------------------------------------------------------------


@betting_app.command("deposit")
def deposit_cmd(
    amount: float = typer.Argument(..., help="Amount to deposit"),
    note: str | None = typer.Option(None, help="Optional note"),
) -> None:
    """Add funds to the bankroll."""
    from gridiron_edge.betting.bankroll import current_balance, deposit

    try:
        deposit(amount, note=note)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    balance: float = current_balance()
    typer.echo(f"Deposited ${amount:.2f}  Balance: ${balance:.2f}")


# ---------------------------------------------------------------------------
# gridiron bet withdraw
# ---------------------------------------------------------------------------


@betting_app.command("withdraw")
def withdraw_cmd(
    amount: float = typer.Argument(..., help="Amount to withdraw"),
    note: str | None = typer.Option(None, help="Optional note"),
) -> None:
    """Remove funds from the bankroll."""
    from gridiron_edge.betting.bankroll import current_balance, withdraw

    try:
        withdraw(amount, note=note)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    balance: float = current_balance()
    typer.echo(f"Withdrew ${amount:.2f}  Balance: ${balance:.2f}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_bet_table(df) -> None:  # noqa: ANN001
    """Print a formatted table of bets."""
    typer.echo(
        f"  {'ID':8s}  {'Game':18s}  {'Market':10s}  {'Side':6s}  "
        f"{'Odds':>6s}  {'Stake':>8s}  {'Status':6s}  {'PnL':>8s}"
    )
    typer.echo("  " + "-" * 80)
    for _, row in df.iterrows():
        bet_id_short: str = str(row["bet_id"])[:8]
        pnl_str: str = (
            f"${row['pnl']:.2f}" if row.get("pnl") is not None and str(row["pnl"]) != "nan" else "—"
        )
        typer.echo(
            f"  {bet_id_short:8s}  {row['game_id']!s:18s}  "
            f"{row['market_type']!s:10s}  {row['side']!s:6s}  "
            f"{int(row['odds']):>+6d}  ${float(row['stake']):>7.2f}  "
            f"{row['status']!s:6s}  {pnl_str:>8s}"
        )


def _render_summary(stats: dict) -> None:
    """Print a formatted performance summary."""
    import math

    typer.echo("\n  === Betting Performance ===")
    typer.echo(f"  Record:  {stats['wins']}W - {stats['losses']}L - {stats['pushes']}P")

    wp = stats["win_pct"]
    wp_str: str = f"{wp:.1%}" if not math.isnan(wp) else "—"
    typer.echo(f"  Win %:   {wp_str}")

    typer.echo(f"  Staked:  ${stats['total_staked']:.2f}")

    pnl = stats["total_pnl"]
    pnl_sign: Literal["", "+"] = "+" if pnl >= 0 else ""
    typer.echo(f"  PnL:     {pnl_sign}${pnl:.2f}")

    roi_val = stats["roi_pct"]
    roi_str: str = f"{roi_val:+.1f}%" if not math.isnan(roi_val) else "—"
    typer.echo(f"  ROI:     {roi_str}")

    clv = stats["mean_clv"]
    clv_str: str = f"{clv:+.4f}" if not math.isnan(clv) else "—"
    typer.echo(f"  CLV:     {clv_str} (n={stats['n_clv_bets']})")

    ev_gap = stats["ev_vs_actual_gap"]
    ev_gap_str: str = f"{ev_gap:+.4f}" if not math.isnan(ev_gap) else "—"
    typer.echo(f"  EV gap:  {ev_gap_str} (n={stats['n_model_bets']})")

    health: str = stats["calibration_health"]
    health_indicator: str = {
        "healthy": "✓",
        "degraded": "⚠",
        "unknown": "—",
    }.get(health, "—")
    typer.echo(f"  Health:  {health_indicator} {health}")

    streak = stats["current_streak"]
    s_type = stats["current_streak_type"]
    streak_str: str = f"{abs(streak)}{s_type}" if s_type else "—"
    typer.echo(f"  Streak:  {streak_str}")
    typer.echo(f"  Best:    {stats['longest_win_streak']}W  Worst: {stats['longest_loss_streak']}L")
