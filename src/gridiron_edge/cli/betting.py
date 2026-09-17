# src/gridiron_edge/cli/betting.py
"""CLI commands for bet tracking, bankroll management, and history import."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

betting_app = typer.Typer(
    help="Bet tracking and performance.",
    no_args_is_help=True,
)


@betting_app.command("log")
def log_cmd(
    game_id: str = typer.Option(
        ...,
        help="Canonical game ID, e.g. 2026_01_KC_LAC",
    ),
    market: str = typer.Option(
        ...,
        help="Market type: moneyline, spread, or total",
    ),
    side: str = typer.Option(
        ...,
        help="Bet side: home, away, over, or under",
    ),
    odds: int = typer.Option(
        ...,
        help="American odds at bet time, e.g. -110",
    ),
    stake: float = typer.Option(..., help="Dollar amount wagered"),
    book: str = typer.Option(
        ...,
        help="Sportsbook name, e.g. draftkings",
    ),
    line: float | None = typer.Option(
        None,
        help="Spread or total line (omit for ML)",
    ),
    model_name: str | None = typer.Option(
        None,
        help=("Model purpose used to identify the edge, e.g. win_prob or qb_pass_yards"),
    ),
    model_type: str | None = typer.Option(
        None,
        help=("Algorithm used to compute the edge, e.g. random_forest or elasticnet"),
    ),
    model_prob: float | None = typer.Option(
        None,
        help="Model probability at bet time",
    ),
    model_ev: float | None = typer.Option(
        None,
        help="Model EV at bet time",
    ),
    edge_strength: str | None = typer.Option(
        None,
        help="Edge classification",
    ),
    confidence_tier: str | None = typer.Option(
        None,
        help="Confidence tier",
    ),
) -> None:
    """Record a new local wager."""
    from gridiron_edge.betting.bankroll import current_balance
    from gridiron_edge.betting.recording import RecordWagerCommand, record_wager
    from gridiron_edge.core.settings import get_settings

    repo = get_settings().repo_root
    try:
        recorded = record_wager(
            RecordWagerCommand(
                game_id=game_id,
                market_type=market,
                side=side,
                odds=odds,
                stake=stake,
                book=book,
                line=line,
                model_name=model_name,
                model_type=model_type,
                model_probability=model_prob,
                expected_value=model_ev,
                edge_strength=edge_strength,
                confidence_tier=confidence_tier,
            ),
            repo=repo,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    balance = current_balance(repo=repo)
    typer.echo(f"Bet logged: {recorded.bet_id}")
    typer.echo(f"  {market} {side} {game_id} @ {odds:+d}  stake=${stake:.2f}  book={book}")
    typer.echo(f"  Balance: ${balance:.2f}")


@betting_app.command("settle")
def settle_cmd(
    bet_id: str = typer.Argument(..., help="UUID of the bet to settle"),
    result: str = typer.Argument(..., help="Result: won, lost, or push"),
) -> None:
    """Settle an open bet."""
    from gridiron_edge.betting.bankroll import current_balance, record_bet_settled
    from gridiron_edge.betting.ledger import settle_bet
    from gridiron_edge.core.settings import get_settings

    repo = get_settings().repo_root
    try:
        row: Series = settle_bet(bet_id, result, repo=repo)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    pnl = float(row["pnl"])
    stake = float(row["stake"])
    settled_at = row["settled_at"]
    record_bet_settled(
        stake,
        pnl,
        bet_id=bet_id,
        settled_at=settled_at.to_pydatetime()
        if hasattr(settled_at, "to_pydatetime")
        else settled_at,
        repo=repo,
    )

    balance = current_balance(repo=repo)
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    typer.echo(f"Settled: {bet_id} -> {result}")
    typer.echo(f"  PnL: {pnl_str}  Balance: ${balance:.2f}")


@betting_app.command("import-history")
def import_history_cmd(
    bets: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Normalized historical bet CSV.",
        ),
    ],
    transactions: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Normalized historical bankroll transaction CSV.",
        ),
    ],
    replace_existing: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="Replace both current betting ledgers after validation.",
        ),
    ] = False,
) -> None:
    """Validate or publish normalized historical betting records."""
    from gridiron_edge.betting.history_import import import_history
    from gridiron_edge.core.settings import get_settings

    try:
        imported = import_history(
            bets_path=bets,
            transactions_path=transactions,
            replace_existing=replace_existing,
            repo=get_settings().repo_root,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if imported.published:
        typer.echo("Historical portfolio imported.")
    else:
        typer.echo("Historical portfolio validated; no files changed.")
    typer.echo(f"  Bets: {imported.bet_count}")
    typer.echo(
        f"  Record: {imported.won_bet_count} won, "
        f"{imported.lost_bet_count} lost, "
        f"{imported.open_bet_count} open"
    )
    typer.echo(f"  Transactions: {imported.transaction_count}")
    typer.echo(f"  Opening bankroll: ${imported.opening_balance:.2f}")
    typer.echo(f"  Cash wager outflow: ${imported.cash_wager_outflow:.2f}")
    typer.echo(f"  Settlement inflow: ${imported.settlement_inflow:.2f}")
    typer.echo(f"  Current bankroll: ${imported.final_bankroll:.2f}")
    typer.echo(f"  Bonus-funded stake: ${imported.bonus_funding:.2f}")
    typer.echo(f"  Unresolved funding: ${imported.unresolved_funding:.2f}")


@betting_app.command("list")
def list_cmd(
    status: str | None = typer.Option(
        None,
        help="Filter by status: open, won, lost, push",
    ),
    market: str | None = typer.Option(None, help="Filter by market type"),
    limit: int = typer.Option(20, help="Max rows to display"),
) -> None:
    """Show bets from the ledger."""
    from gridiron_edge.betting.ledger import load_bets
    from gridiron_edge.core.settings import get_settings

    frame = load_bets(
        status=status,
        market_type=market,
        repo=get_settings().repo_root,
    )
    if frame.empty:
        typer.echo("No bets found.")
        return
    total = len(frame)
    _render_bet_table(frame.head(limit))
    typer.echo(f"  ... showing {limit} of {total} bets" if total > limit else f"  {total} bet(s)")


@betting_app.command("summary")
def summary_cmd(
    split_by: str | None = typer.Option(
        None,
        help="Split record/ROI by column",
    ),
) -> None:
    """Show the performance dashboard."""
    from gridiron_edge.betting.ledger import load_bets
    from gridiron_edge.betting.performance import record, roi, summary
    from gridiron_edge.core.settings import get_settings

    bets = load_bets(repo=get_settings().repo_root)
    if bets.empty:
        typer.echo("No bets to summarise.")
        return
    stats: dict[str, Any] = summary(bets)
    _render_summary(stats)
    if split_by:
        typer.echo(f"\n  Record by {split_by}:")
        record_frame = record(bets, split_by=split_by)
        if not record_frame.empty:
            typer.echo(record_frame.to_string(index=False))
        typer.echo(f"\n  ROI by {split_by}:")
        roi_frame = roi(bets, split_by=split_by)
        if not roi_frame.empty:
            typer.echo(roi_frame.to_string(index=False))


@betting_app.command("balance")
def balance_cmd(
    limit: int = typer.Option(
        10,
        help="Number of recent transactions to show",
    ),
) -> None:
    """Show current bankroll balance and recent transactions."""
    from gridiron_edge.betting.bankroll import balance_history, current_balance
    from gridiron_edge.core.settings import get_settings

    repo = get_settings().repo_root
    typer.echo(f"  Current balance: ${current_balance(repo=repo):.2f}")
    history = balance_history(repo=repo)
    if history.empty:
        typer.echo("  No transactions yet.")
        return
    recent = history.tail(limit)
    typer.echo(f"\n  Recent transactions (last {min(limit, len(history))}):")
    for _, row in recent.iterrows():
        sign: Literal["-", "+"] = "+" if row["signed_amount"] >= 0 else "-"
        typer.echo(
            f"    {row['timestamp']:%Y-%m-%d %H:%M}  "
            f"{row['txn_type']:12s}  "
            f"{sign}${abs(row['signed_amount']):.2f}  "
            f"bal=${row['running_balance']:.2f}"
        )


@betting_app.command("export")
def export_cmd(
    status: str | None = typer.Option(None, help="Filter by status"),
    output: str | None = typer.Option(None, help="Output CSV path"),
) -> None:
    """Export bets to CSV."""
    from gridiron_edge.betting.ledger import load_bets
    from gridiron_edge.core.settings import get_settings

    settings = get_settings()
    frame = load_bets(status=status, repo=settings.repo_root)
    if frame.empty:
        typer.echo("No bets to export.")
        return
    if output is None:
        directory = settings.repo_root / "data" / "output" / "bets"
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "bets_export.csv"
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    typer.echo(f"Exported {len(frame)} bet(s) to {output_path}")


@betting_app.command("deposit")
def deposit_cmd(
    amount: float = typer.Argument(..., help="Amount to deposit"),
    note: str | None = typer.Option(None, help="Optional note"),
) -> None:
    """Add funds to the bankroll."""
    from gridiron_edge.betting.bankroll import current_balance, deposit
    from gridiron_edge.core.settings import get_settings

    repo = get_settings().repo_root
    try:
        deposit(amount, note=note, repo=repo)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Deposited ${amount:.2f}  Balance: ${current_balance(repo=repo):.2f}")


@betting_app.command("withdraw")
def withdraw_cmd(
    amount: float = typer.Argument(..., help="Amount to withdraw"),
    note: str | None = typer.Option(None, help="Optional note"),
) -> None:
    """Remove funds from the bankroll."""
    from gridiron_edge.betting.bankroll import current_balance, withdraw
    from gridiron_edge.core.settings import get_settings

    repo = get_settings().repo_root
    try:
        withdraw(amount, note=note, repo=repo)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Withdrew ${amount:.2f}  Balance: ${current_balance(repo=repo):.2f}")


def _render_bet_table(frame: DataFrame) -> None:
    """Print a formatted table of bets."""
    typer.echo(
        f"  {'ID':8s}  {'Bet':18s}  {'Market':16s}  {'Side':6s}  "
        f"{'Odds':>6s}  {'Stake':>8s}  {'Status':6s}  {'PnL':>8s}"
    )
    typer.echo("  " + "-" * 86)
    for _, row in frame.iterrows():
        bet_id = str(row["bet_id"])[:8]
        label = row.get("game_id")
        if label is None or str(label) == "nan":
            label = row.get("description")
        label_text = str(label) if label is not None else "-"
        side = row.get("side")
        side_text = "-" if side is None or str(side) == "nan" else str(side)
        pnl = row.get("pnl")
        pnl_text = "-" if pnl is None or str(pnl) == "nan" else f"${pnl:.2f}"
        typer.echo(
            f"  {bet_id:8s}  {label_text[:18]:18s}  "
            f"{str(row['market_type'])[:16]:16s}  {side_text[:6]:6s}  "
            f"{int(row['odds']):>+6d}  ${float(row['stake']):>7.2f}  "
            f"{row['status']!s:6s}  {pnl_text:>8s}"
        )


def _render_summary(stats: dict[str, Any]) -> None:
    """Print a formatted performance summary."""
    import math

    typer.echo("\n  === Betting Performance ===")
    typer.echo(f"  Record:  {stats['wins']}W - {stats['losses']}L - {stats['pushes']}P")
    win_pct = stats["win_pct"]
    typer.echo(f"  Win %:   {win_pct:.1%}" if not math.isnan(win_pct) else "  Win %:   -")
    typer.echo(f"  Staked:  ${stats['total_staked']:.2f}")
    pnl = stats["total_pnl"]
    typer.echo(f"  PnL:     {'+' if pnl >= 0 else ''}${pnl:.2f}")
    roi_value = stats["roi_pct"]
    roi_text = f"{roi_value:+.1f}%" if not math.isnan(roi_value) else "-"
    typer.echo(f"  ROI:     {roi_text}")
    clv = stats["mean_clv"]
    clv_text = f"{clv:+.4f}" if not math.isnan(clv) else "-"
    typer.echo(f"  CLV:     {clv_text} (n={stats['n_clv_bets']})")
    ev_gap = stats["ev_vs_actual_gap"]
    gap_text = f"{ev_gap:+.4f}" if not math.isnan(ev_gap) else "-"
    typer.echo(f"  EV gap:  {gap_text} (n={stats['n_model_bets']})")
    health = str(stats["calibration_health"])
    indicator: str = {
        "healthy": "✓",
        "degraded": "⚠",
        "unknown": "-",
    }.get(
        health,
        "-",
    )
    typer.echo(f"  Health:  {indicator} {health}")
    streak = int(stats["current_streak"])
    streak_type = str(stats["current_streak_type"])
    streak_text = f"{abs(streak)}{streak_type}" if streak_type else "-"
    typer.echo(f"  Streak:  {streak_text}")
    typer.echo(f"  Best:    {stats['longest_win_streak']}W  Worst: {stats['longest_loss_streak']}L")
