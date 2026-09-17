import { useState } from "react";
import {
  usePortfolioBets,
  usePortfolioCurve,
  usePortfolioSplits,
  usePortfolioSummary,
  usePortfolioTransactions,
} from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { BalanceCurve } from "../components/portfolio/BalanceCurve";
import { useAppState } from "../context/AppStateContext";
import { formatOdds } from "../utils/odds";
import { ErrorCard } from "../components/error/ErrorCard";

type SplitDimension = "market_type" | "funding_type" | "book";

export function Bankroll() {
  const summary = usePortfolioSummary();
  const bets = usePortfolioBets();
  const curve = usePortfolioCurve();
  const transactions = usePortfolioTransactions();
  const [splitDim, setSplitDim] = useState<SplitDimension>("market_type");
  const splits = usePortfolioSplits(splitDim);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <SummaryCard result={summary} />
        <RecordCard result={summary} />
      </div>
      <CurveCard result={curve} />
      <BetsCard result={bets} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <SplitsCard result={splits} dimension={splitDim} onDimensionChange={setSplitDim} />
        <TransactionsCard result={transactions} />
      </div>
    </div>
  );
}

function SummaryCard({ result }: { result: ReturnType<typeof usePortfolioSummary> }) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Available Bankroll</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && (
        <>
          <div className="mono tnum" style={{ fontSize: 28, marginBottom: 16 }}>
            ${result.data.bankroll?.toFixed(2) ?? "—"}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <MetricCell label="Total P&L" value={<span className={result.data.total_pnl != null && result.data.total_pnl >= 0 ? "pos" : "neg"}>{result.data.total_pnl != null ? `${result.data.total_pnl >= 0 ? "+" : ""}$${result.data.total_pnl.toFixed(2)}` : "—"}</span>} />
            <MetricCell label="ROI" value={<span className={result.data.roi_pct != null && result.data.roi_pct >= 0 ? "pos" : "neg"}>{result.data.roi_pct != null ? `${result.data.roi_pct >= 0 ? "+" : ""}${result.data.roi_pct.toFixed(1)}%` : "—"}</span>} />
            <MetricCell label="Total Staked" value={result.data.total_staked != null ? `$${result.data.total_staked.toFixed(2)}` : "—"} />
            <MetricCell label="Mean CLV" value={<FieldValueOrStatus value={result.data.mean_clv} formatter={(value) => value.toFixed(3)} status={result.data._meta?.field_status?.mean_clv as FieldStatus | undefined} />} />
          </div>
        </>
      )}
    </div>
  );
}

function RecordCard({ result }: { result: ReturnType<typeof usePortfolioSummary> }) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Record</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && (
        <>
          <div className="mono tnum" style={{ fontSize: 28, marginBottom: 16 }}>
            {result.data.wins ?? 0}-{result.data.losses ?? 0}{result.data.pushes && result.data.pushes > 0 ? `-${result.data.pushes}` : ""}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <MetricCell label="Win %" value={result.data.win_pct != null ? `${(result.data.win_pct * 100).toFixed(1)}%` : "—"} />
            <MetricCell label="Total Bets" value={result.data.total_bets?.toString() ?? "—"} />
            <MetricCell label="Open Bets" value={result.data.open_bets?.toString() ?? "—"} />
            <MetricCell label="Current Streak" value={<FieldValueOrStatus value={result.data.current_streak} formatter={(value) => `${value}`} status={result.data._meta?.field_status?.current_streak as FieldStatus | undefined} />} />
          </div>
        </>
      )}
    </div>
  );
}

function CurveCard({ result }: { result: ReturnType<typeof usePortfolioCurve> }) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Balance Curve</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && <BalanceCurve points={result.data.items ?? []} width={1100} height={140} />}
    </div>
  );
}

function BetsCard({ result }: { result: ReturnType<typeof usePortfolioBets> }) {
  const { state } = useAppState();
  const items = result.data?.items ?? [];
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Bets</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No bets yet.</div>}
      {items.length > 0 && (
        <table className="mono tnum" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
          <thead><tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
            <Header>Placed</Header><Header>Bet</Header><Header>Market</Header><Header>Side</Header><Header>Funding</Header>
            <Header right>Odds</Header><Header right>Stake</Header><Header>Status</Header><Header right>P&L</Header>
          </tr></thead>
          <tbody>{items.map((bet) => (
            <tr key={bet.bet_id ?? bet.source_bet_id} style={{ borderTop: "1px solid var(--line-soft)" }}>
              <Cell dim>{formatDateShort(bet.placed_at)}</Cell>
              <Cell dim>{bet.game_id ?? bet.description ?? "—"}</Cell>
              <Cell>{formatMarketType(bet.market_type)}</Cell>
              <Cell>{bet.side ?? "—"}{bet.line != null ? ` (${bet.line})` : ""}</Cell>
              <Cell>{formatFundingType(bet.funding_type)}</Cell>
              <Cell right>{bet.odds != null ? formatOdds(bet.odds, state.oddsFormat) : "—"}</Cell>
              <Cell right>{bet.stake != null ? `$${bet.stake.toFixed(2)}` : "—"}</Cell>
              <Cell><BetStatusPill status={bet.status} /></Cell>
              <td style={{ padding: "10px 0", textAlign: "right", color: bet.pnl == null ? "var(--ink-4)" : bet.pnl >= 0 ? "var(--pos)" : "var(--neg)" }}>
                {bet.pnl != null ? `${bet.pnl >= 0 ? "+" : ""}$${bet.pnl.toFixed(2)}` : "—"}
              </td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </div>
  );
}

function BetStatusPill({ status }: { status: string | null | undefined }) {
  if (!status) return <span className="dim mono">—</span>;
  const color = status === "won" ? "var(--pos)" : status === "lost" ? "var(--neg)" : status === "open" ? "var(--info)" : "var(--ink-3)";
  return <span className="mono upper" style={{ fontSize: 9, color, padding: "2px 6px", border: `1px solid ${color}`, borderRadius: 3 }}>{status}</span>;
}

function SplitsCard({ result, dimension, onDimensionChange }: { result: ReturnType<typeof usePortfolioSplits>; dimension: SplitDimension; onDimensionChange: (dimension: SplitDimension) => void }) {
  const tabs: { value: SplitDimension; label: string }[] = [
    { value: "market_type", label: "Market" },
    { value: "funding_type", label: "Funding" },
    { value: "book", label: "Book" },
  ];
  const items = result.data?.items ?? [];
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div className="upper dim" style={{ fontSize: 10 }}>Splits</div>
        <div style={{ display: "flex", gap: 8 }}>{tabs.map((tab) => <TabButton key={tab.value} label={tab.label} active={tab.value === dimension} onClick={() => onDimensionChange(tab.value)} />)}</div>
      </div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No split data.</div>}
      {items.length > 0 && (
        <table className="mono tnum" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
          <thead><tr style={{ color: "var(--ink-3)", textAlign: "left" }}><Header>Segment</Header><Header right>W-L-P</Header><Header right>Win %</Header><Header right>ROI</Header></tr></thead>
          <tbody>{items.map((split) => <tr key={split.dimension_value} style={{ borderTop: "1px solid var(--line-soft)" }}>
            <Cell dim>{formatSplitValue(dimension, split.dimension_value)}</Cell>
            <Cell right>{split.wins ?? 0}-{split.losses ?? 0}{split.pushes && split.pushes > 0 ? `-${split.pushes}` : ""}</Cell>
            <Cell right>{split.win_pct != null ? `${(split.win_pct * 100).toFixed(1)}%` : "—"}</Cell>
            <td style={{ padding: "10px 0", textAlign: "right", color: split.roi == null ? "var(--ink-4)" : split.roi >= 0 ? "var(--pos)" : "var(--neg)" }}>{split.roi != null ? `${split.roi >= 0 ? "+" : ""}${(split.roi * 100).toFixed(1)}%` : "—"}</td>
          </tr>)}</tbody>
        </table>
      )}
    </div>
  );
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return <button type="button" onClick={onClick} aria-pressed={active} style={{ background: "transparent", border: "none", padding: 0, paddingBottom: 2, cursor: "pointer", font: "inherit", fontSize: 11, color: active ? "var(--ink)" : "var(--ink-3)", borderBottom: active ? "2px solid var(--pos)" : "2px solid transparent" }}>{label}</button>;
}

function TransactionsCard({ result }: { result: ReturnType<typeof usePortfolioTransactions> }) {
  const items = result.data?.items ?? [];
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Transactions</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No transactions.</div>}
      {items.length > 0 && (
        <table className="mono tnum" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
          <thead><tr style={{ color: "var(--ink-3)", textAlign: "left" }}><Header>When</Header><Header>Type</Header><Header right>Amount</Header><Header right>Balance</Header></tr></thead>
          <tbody>{items.map((transaction) => {
            const inflow = transaction.txn_type === "deposit" || transaction.txn_type === "bet_settled";
            const sign = inflow ? "+" : "-";
            const color = inflow ? "var(--pos)" : transaction.txn_type === "bet_placed" ? "var(--neg)" : "var(--ink-2)";
            return <tr key={transaction.txn_id ?? transaction.source_transaction_id} style={{ borderTop: "1px solid var(--line-soft)" }}>
              <Cell dim>{formatDateShort(transaction.timestamp)}</Cell><Cell>{formatTransactionType(transaction.txn_type)}</Cell>
              <td style={{ padding: "10px 12px 10px 0", textAlign: "right", color }}>{transaction.amount != null ? `${sign}$${transaction.amount.toFixed(2)}` : "—"}</td>
              <Cell right>{transaction.balance_after != null ? `$${transaction.balance_after.toFixed(2)}` : "—"}</Cell>
            </tr>;
          })}</tbody>
        </table>
      )}
    </div>
  );
}

function Header({ children, right = false }: { children: React.ReactNode; right?: boolean }) {
  return <th style={{ padding: "8px 12px 8px 0", textAlign: right ? "right" : "left" }}>{children}</th>;
}

function Cell({ children, right = false, dim = false }: { children: React.ReactNode; right?: boolean; dim?: boolean }) {
  return <td style={{ padding: "10px 12px 10px 0", textAlign: right ? "right" : "left", color: dim ? "var(--ink-2)" : undefined }}>{children}</td>;
}

function MetricCell({ label, value }: { label: string; value: React.ReactNode }) {
  return <div><div className="upper dim2" style={{ fontSize: 9, marginBottom: 4 }}>{label}</div><div className="mono tnum" style={{ fontSize: 14 }}>{value}</div></div>;
}

function FieldValueOrStatus<T>({ value, formatter, status }: { value: T | null | undefined; formatter: (value: T) => string; status: FieldStatus | undefined }) {
  if (value != null && value !== "") return <>{formatter(value)}</>;
  if (!status) return <span className="dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

function formatMarketType(value: string | null | undefined): string {
  const labels: Record<string, string> = { moneyline: "Moneyline", spread: "Spread", total: "Total", player_prop: "Player Prop", parlay: "Parlay", same_game_parlay: "Same Game Parlay", special: "Special" };
  return value ? labels[value] ?? titleCase(value) : "—";
}

function formatFundingType(value: string | null | undefined): string {
  const labels: Record<string, string> = { cash: "Cash", bonus: "Bonus", unresolved: "Unresolved" };
  return value ? labels[value] ?? titleCase(value) : "—";
}

function formatTransactionType(value: string | null | undefined): string {
  return value ? titleCase(value) : "—";
}

function formatSplitValue(dimension: SplitDimension, value: string): string {
  if (dimension === "market_type") return formatMarketType(value);
  if (dimension === "funding_type") return formatFundingType(value);
  return value;
}

function titleCase(value: string): string {
  return value.split("_").map((part) => part ? `${part[0].toUpperCase()}${part.slice(1)}` : part).join(" ");
}

function formatDateShort(timestamp: string | null | undefined): string {
  if (!timestamp) return "—";
  return timestamp.split(".")[0].replace("T", " ");
}
