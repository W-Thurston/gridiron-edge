import { useMemo, useState } from "react";
import {
  usePortfolioBets,
  usePortfolioCurve,
  usePortfolioSplits,
  usePortfolioSummary,
  usePortfolioTransactions,
} from "../api/hooks";
import type { components } from "../api/schema";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { BalanceCurve } from "../components/portfolio/BalanceCurve";
import { useAppState } from "../context/AppStateContext";
import { formatOdds } from "../utils/odds";
import { ErrorCard } from "../components/error/ErrorCard";

type BetRow = components["schemas"]["BetRow"];
type SplitDimension = "market_type" | "funding_type" | "book";
type DateRange = "all" | "7d" | "30d";
type BetFilters = {
  search: string;
  status: string;
  market: string;
  funding: string;
  book: string;
  dateRange: DateRange;
};

const DEFAULT_FILTERS: BetFilters = {
  search: "",
  status: "all",
  market: "all",
  funding: "all",
  book: "all",
  dateRange: "all",
};
const TABLE_MAX_HEIGHT = 41 * 21;

export function Bankroll() {
  const summary = usePortfolioSummary();
  const bets = usePortfolioBets();
  const curve = usePortfolioCurve();
  const transactions = usePortfolioTransactions();
  const [recordMarket, setRecordMarket] = useState("all");
  const [splitDim, setSplitDim] = useState<SplitDimension>("market_type");
  const splits = usePortfolioSplits(splitDim);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <SummaryCard result={summary} curve={curve.data?.items ?? []} />
        <RecordCard
          summary={summary}
          bets={bets.data?.items ?? []}
          market={recordMarket}
          onMarketChange={setRecordMarket}
        />
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

function SummaryCard({ result, curve }: { result: ReturnType<typeof usePortfolioSummary>; curve: Array<{ bankroll: number }> }) {
  const start = curve[0]?.bankroll;
  const current = result.data?.bankroll;
  const change = start != null && current != null ? current - start : null;
  const changePct = change != null && start !== 0 ? (change / start) * 100 : null;
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Available Bankroll</div>
      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
      {result.data && (
        <>
          <div className="mono tnum" style={{ fontSize: 28, marginBottom: 5 }}>${current?.toFixed(2) ?? "—"}</div>
          {change != null && (
            <div className="mono dim2" style={{ fontSize: 10, marginBottom: 16 }}>
              Since start: <span className={change >= 0 ? "pos" : "neg"}>{formatSignedMoney(change)}{changePct != null ? ` (${formatSignedPercent(changePct)})` : ""}</span>
            </div>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <MetricCell label="Settled P&L" value={<span className={result.data.total_pnl != null && result.data.total_pnl >= 0 ? "pos" : "neg"}>{result.data.total_pnl != null ? formatSignedMoney(result.data.total_pnl) : "—"}</span>} />
            <MetricCell label="ROI" value={<span className={result.data.roi_pct != null && result.data.roi_pct >= 0 ? "pos" : "neg"}>{result.data.roi_pct != null ? formatSignedPercent(result.data.roi_pct) : "—"}</span>} />
            <MetricCell label="Settled Stake" value={result.data.total_staked != null ? `$${result.data.total_staked.toFixed(2)}` : "—"} />
            <MetricCell label="Mean CLV" value={<FieldValueOrStatus value={result.data.mean_clv} formatter={(value) => value.toFixed(3)} status={result.data._meta?.field_status?.mean_clv as FieldStatus | undefined} />} />
          </div>
          <p className="dim2" style={{ fontSize: 10, lineHeight: 1.45, margin: "16px 0 0" }}>
            Settled P&amp;L excludes open wagers. Available bankroll also reflects cash committed to open bets.
          </p>
        </>
      )}
    </div>
  );
}

function RecordCard({ summary, bets, market, onMarketChange }: { summary: ReturnType<typeof usePortfolioSummary>; bets: BetRow[]; market: string; onMarketChange: (market: string) => void }) {
  const marketOptions = useMemo(() => distinctValues(bets, "market_type"), [bets]);
  const scoped = market === "all" ? bets : bets.filter((bet) => bet.market_type === market);
  const record = summarizeRecord(scoped);
  const isAll = market === "all";
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 16 }}>
        <div className="upper dim" style={{ fontSize: 10 }}>Record</div>
        <FilterSelect label="Record market" value={market} onChange={onMarketChange}>
          <option value="all">All Markets</option>
          {marketOptions.map((value) => <option key={value} value={value}>{formatMarketType(value)}</option>)}
        </FilterSelect>
      </div>
      {summary.isLoading && <div className="dim">Loading…</div>}
      {summary.error && <ErrorCard error={summary.error} onRetry={() => summary.refetch()} />}
      {summary.data && (
        <>
          <div className="mono tnum" style={{ fontSize: 28, marginBottom: 16 }}>{record.wins}-{record.losses}{record.pushes > 0 ? `-${record.pushes}` : ""}</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <MetricCell label="Win %" value={record.winPct != null ? `${(record.winPct * 100).toFixed(1)}%` : "—"} />
            <MetricCell label="Bets in View" value={String(scoped.length)} />
            <MetricCell label="Open Bets" value={String(record.open)} />
            <MetricCell label="Current Streak" value={isAll ? <FieldValueOrStatus value={summary.data.current_streak} formatter={(value) => `${value}`} status={summary.data._meta?.field_status?.current_streak as FieldStatus | undefined} /> : record.streak ?? "—"} />
          </div>
        </>
      )}
    </div>
  );
}

function CurveCard({ result }: { result: ReturnType<typeof usePortfolioCurve> }) {
  return <div className="hm-card" style={{ padding: 24 }}>
    <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>Balance Curve</div>
    {result.isLoading && <div className="dim">Loading…</div>}
    {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
    {result.data && <BalanceCurve points={result.data.items ?? []} width={1100} height={150} />}
  </div>;
}

function BetsCard({ result }: { result: ReturnType<typeof usePortfolioBets> }) {
  const { state } = useAppState();
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [showMore, setShowMore] = useState(false);
  const items = useMemo(
    () => result.data?.items ?? [],
    [result.data?.items],
  );
  const markets = useMemo(() => distinctValues(items, "market_type"), [items]);
  const funding = useMemo(() => distinctValues(items, "funding_type"), [items]);
  const books = useMemo(() => distinctValues(items, "book"), [items]);
  const filtered = useMemo(() => filterBets(items, filters), [items, filters]);
  const activeCount = Object.entries(filters).filter(([key, value]) => value !== DEFAULT_FILTERS[key as keyof BetFilters]).length;
  const update = <K extends keyof BetFilters>(key: K, value: BetFilters[K]) => setFilters((current) => ({ ...current, [key]: value }));

  return <div className="hm-card" style={{ padding: 24 }}>
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 12 }}>
      <div><div className="upper dim" style={{ fontSize: 10 }}>Bets</div><div className="mono dim2" style={{ fontSize: 10, marginTop: 4 }}>{filtered.length} of {items.length} bets</div></div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", justifyContent: "flex-end" }}>
        <input aria-label="Search bets" type="search" placeholder="Search bets" value={filters.search} onChange={(event) => update("search", event.target.value)} style={controlStyle} />
        <FilterSelect label="Bet status" value={filters.status} onChange={(value) => update("status", value)}><option value="all">All Statuses</option>{["open", "won", "lost", "push"].map((value) => <option key={value} value={value}>{titleCase(value)}</option>)}</FilterSelect>
        <FilterSelect label="Bet market" value={filters.market} onChange={(value) => update("market", value)}><option value="all">All Markets</option>{markets.map((value) => <option key={value} value={value}>{formatMarketType(value)}</option>)}</FilterSelect>
        <button type="button" onClick={() => setShowMore((value) => !value)} aria-expanded={showMore} style={buttonStyle}>Filters{activeCount > 0 ? ` (${activeCount})` : ""}</button>
        {activeCount > 0 && <button type="button" onClick={() => setFilters(DEFAULT_FILTERS)} style={buttonStyle}>Clear</button>}
      </div>
    </div>
    {showMore && <div aria-label="Additional bet filters" style={{ display: "flex", gap: 8, padding: 10, marginBottom: 12, background: "var(--bg-1)", border: "1px solid var(--line-soft)", borderRadius: 5 }}>
      <FilterSelect label="Bet funding" value={filters.funding} onChange={(value) => update("funding", value)}><option value="all">All Funding</option>{funding.map((value) => <option key={value} value={value}>{formatFundingType(value)}</option>)}</FilterSelect>
      <FilterSelect label="Bet sportsbook" value={filters.book} onChange={(value) => update("book", value)}><option value="all">All Sportsbooks</option>{books.map((value) => <option key={value} value={value}>{value}</option>)}</FilterSelect>
      <FilterSelect label="Bet date range" value={filters.dateRange} onChange={(value) => update("dateRange", value as DateRange)}><option value="all">All Time</option><option value="7d">Last 7 Days</option><option value="30d">Last 30 Days</option></FilterSelect>
    </div>}
    {result.isLoading && <div className="dim">Loading…</div>}
    {result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}
    {result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No bets yet.</div>}
    {items.length > 0 && filtered.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No bets match the active filters.</div>}
    {filtered.length > 0 && <ScrollableTable label={`Bets, ${filtered.length} in view`}>
      <table className="mono tnum" style={tableStyle}><thead><tr style={{ color: "var(--ink-3)" }}><Header>Placed</Header><Header>Bet</Header><Header>Market</Header><Header>Side</Header><Header>Funding</Header><Header>Book</Header><Header right>Odds</Header><Header right>Stake</Header><Header>Status</Header><Header right>P&amp;L</Header></tr></thead>
      <tbody>{filtered.map((bet) => <tr key={bet.bet_id ?? bet.source_bet_id} style={{ borderTop: "1px solid var(--line-soft)" }}><Cell dim>{formatDateShort(bet.placed_at)}</Cell><Cell dim>{bet.game_id ?? bet.description ?? "—"}</Cell><Cell>{formatMarketType(bet.market_type)}</Cell><Cell>{bet.side ?? "—"}{bet.line != null ? ` (${bet.line})` : ""}</Cell><Cell>{formatFundingType(bet.funding_type)}</Cell><Cell>{bet.book ?? "—"}</Cell><Cell right>{bet.odds != null ? formatOdds(bet.odds, state.oddsFormat) : "—"}</Cell><Cell right>{bet.stake != null ? `$${bet.stake.toFixed(2)}` : "—"}</Cell><Cell><BetStatusPill status={bet.status} /></Cell><td style={{ padding: "10px 0", textAlign: "right", color: bet.pnl == null ? "var(--ink-4)" : bet.pnl >= 0 ? "var(--pos)" : "var(--neg)" }}>{bet.pnl != null ? formatSignedMoney(bet.pnl) : "—"}</td></tr>)}</tbody></table>
    </ScrollableTable>}
  </div>;
}

function BetStatusPill({ status }: { status: string | null | undefined }) {
  if (!status) return <span className="dim mono">—</span>;
  const color = status === "won" ? "var(--pos)" : status === "lost" ? "var(--neg)" : status === "open" ? "var(--info)" : "var(--ink-3)";
  return <span className="mono upper" style={{ fontSize: 9, color, padding: "2px 6px", border: `1px solid ${color}`, borderRadius: 3 }}>{status}</span>;
}

function SplitsCard({ result, dimension, onDimensionChange }: { result: ReturnType<typeof usePortfolioSplits>; dimension: SplitDimension; onDimensionChange: (dimension: SplitDimension) => void }) {
  const tabs: { value: SplitDimension; label: string }[] = [{ value: "market_type", label: "Market" }, { value: "funding_type", label: "Funding" }, { value: "book", label: "Book" }];
  const items = result.data?.items ?? [];
  return <div className="hm-card" style={{ padding: 24 }}><div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}><div className="upper dim" style={{ fontSize: 10 }}>Splits</div><div style={{ display: "flex", gap: 8 }}>{tabs.map((tab) => <TabButton key={tab.value} label={tab.label} active={tab.value === dimension} onClick={() => onDimensionChange(tab.value)} />)}</div></div>{result.isLoading && <div className="dim">Loading…</div>}{result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}{result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No split data.</div>}{items.length > 0 && <table className="mono tnum" style={tableStyle}><thead><tr style={{ color: "var(--ink-3)" }}><Header>Segment</Header><Header right>W-L-P</Header><Header right>Win %</Header><Header right>ROI</Header></tr></thead><tbody>{items.map((split) => <tr key={split.dimension_value} style={{ borderTop: "1px solid var(--line-soft)" }}><Cell dim>{formatSplitValue(dimension, split.dimension_value)}</Cell><Cell right>{split.wins ?? 0}-{split.losses ?? 0}{split.pushes && split.pushes > 0 ? `-${split.pushes}` : ""}</Cell><Cell right>{split.win_pct != null ? `${(split.win_pct * 100).toFixed(1)}%` : "—"}</Cell><td style={{ padding: "10px 0", textAlign: "right", color: split.roi == null ? "var(--ink-4)" : split.roi >= 0 ? "var(--pos)" : "var(--neg)" }}>{split.roi != null ? formatSignedPercent(split.roi * 100) : "—"}</td></tr>)}</tbody></table>}</div>;
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) { return <button type="button" onClick={onClick} aria-pressed={active} style={{ ...buttonStyle, color: active ? "var(--ink)" : "var(--ink-3)", borderBottom: active ? "2px solid var(--pos)" : "1px solid var(--line-soft)" }}>{label}</button>; }

function TransactionsCard({ result }: { result: ReturnType<typeof usePortfolioTransactions> }) {
  const items = result.data?.items ?? [];
  return <div className="hm-card" style={{ padding: 24 }}><div className="upper dim" style={{ fontSize: 10 }}>Transactions</div><div className="mono dim2" style={{ fontSize: 10, margin: "4px 0 16px" }}>{items.length} transactions</div>{result.isLoading && <div className="dim">Loading…</div>}{result.error && <ErrorCard error={result.error} onRetry={() => result.refetch()} />}{result.data && items.length === 0 && <div className="dim mono" style={{ fontSize: 12 }}>No transactions.</div>}{items.length > 0 && <ScrollableTable label={`Transactions, ${items.length} total`}><table className="mono tnum" style={tableStyle}><thead><tr style={{ color: "var(--ink-3)" }}><Header>When</Header><Header>Type</Header><Header right>Amount</Header><Header right>Balance</Header></tr></thead><tbody>{items.map((transaction) => { const inflow = transaction.txn_type === "deposit" || transaction.txn_type === "bet_settled"; const sign = inflow ? "+" : "-"; const color = inflow ? "var(--pos)" : transaction.txn_type === "bet_placed" ? "var(--neg)" : "var(--ink-2)"; return <tr key={transaction.txn_id ?? transaction.source_transaction_id} style={{ borderTop: "1px solid var(--line-soft)" }}><Cell dim>{formatDateShort(transaction.timestamp)}</Cell><Cell>{formatTransactionType(transaction.txn_type)}</Cell><td style={{ padding: "10px 12px 10px 0", textAlign: "right", color }}>{transaction.amount != null ? `${sign}$${transaction.amount.toFixed(2)}` : "—"}</td><Cell right>{transaction.balance_after != null ? `$${transaction.balance_after.toFixed(2)}` : "—"}</Cell></tr>; })}</tbody></table></ScrollableTable>}</div>;
}

function ScrollableTable({ label, children }: { label: string; children: React.ReactNode }) { return <div role="region" aria-label={label} tabIndex={0} style={{ maxHeight: TABLE_MAX_HEIGHT, overflowY: "auto", overflowX: "auto" }}>{children}</div>; }
function FilterSelect({ label, value, onChange, children }: { label: string; value: string; onChange: (value: string) => void; children: React.ReactNode }) { return <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)} style={controlStyle}>{children}</select>; }
function Header({ children, right = false }: { children: React.ReactNode; right?: boolean }) { return <th style={{ padding: "8px 12px 8px 0", textAlign: right ? "right" : "left", position: "sticky", top: 0, zIndex: 1, background: "var(--bg)" }}>{children}</th>; }
function Cell({ children, right = false, dim = false }: { children: React.ReactNode; right?: boolean; dim?: boolean }) { return <td style={{ padding: "10px 12px 10px 0", textAlign: right ? "right" : "left", color: dim ? "var(--ink-2)" : undefined, whiteSpace: "nowrap" }}>{children}</td>; }
function MetricCell({ label, value }: { label: string; value: React.ReactNode }) { return <div><div className="upper dim2" style={{ fontSize: 9, marginBottom: 4 }}>{label}</div><div className="mono tnum" style={{ fontSize: 14 }}>{value}</div></div>; }
function FieldValueOrStatus<T>({ value, formatter, status }: { value: T | null | undefined; formatter: (value: T) => string; status: FieldStatus | undefined }) { if (value != null && value !== "") return <>{formatter(value)}</>; if (!status) return <span className="dim2">—</span>; if (status === "pending") return <PendingField />; return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />; }

function summarizeRecord(bets: BetRow[]) {
  const settled = bets.filter((bet) => ["won", "lost", "push"].includes(bet.status ?? "")).sort((a, b) => (a.placed_at ?? "").localeCompare(b.placed_at ?? ""));
  const wins = settled.filter((bet) => bet.status === "won").length;
  const losses = settled.filter((bet) => bet.status === "lost").length;
  const pushes = settled.filter((bet) => bet.status === "push").length;
  let streak = 0;
  for (const bet of settled) { if (bet.status === "won") streak = streak > 0 ? streak + 1 : 1; else if (bet.status === "lost") streak = streak < 0 ? streak - 1 : -1; else streak = 0; }
  return { wins, losses, pushes, open: bets.filter((bet) => bet.status === "open").length, winPct: wins + losses > 0 ? wins / (wins + losses) : null, streak: streak > 0 ? `W${streak}` : streak < 0 ? `L${Math.abs(streak)}` : null };
}

function filterBets(bets: BetRow[], filters: BetFilters) {
  const now = Date.now();
  const cutoffDays = filters.dateRange === "7d" ? 7 : filters.dateRange === "30d" ? 30 : null;
  const query = filters.search.trim().toLowerCase();
  return bets.filter((bet) => {
    const searchable = [bet.game_id, bet.description, bet.book, bet.source_bet_id, bet.side].filter(Boolean).join(" ").toLowerCase();
    const placed = bet.placed_at ? Date.parse(bet.placed_at) : Number.NaN;
    return (!query || searchable.includes(query)) && (filters.status === "all" || bet.status === filters.status) && (filters.market === "all" || bet.market_type === filters.market) && (filters.funding === "all" || bet.funding_type === filters.funding) && (filters.book === "all" || bet.book === filters.book) && (cutoffDays == null || (Number.isFinite(placed) && placed >= now - cutoffDays * 86_400_000));
  });
}

function distinctValues<K extends keyof BetRow>(rows: BetRow[], key: K): string[] { return [...new Set(rows.map((row) => row[key]).filter((value): value is NonNullable<BetRow[K]> => value != null).map(String))].sort(); }
function formatMarketType(value: string | null | undefined): string { const labels: Record<string, string> = { moneyline: "Moneyline", spread: "Spread", total: "Total", player_prop: "Player Prop", parlay: "Parlay", same_game_parlay: "Same Game Parlay", special: "Special" }; return value ? labels[value] ?? titleCase(value) : "—"; }
function formatFundingType(value: string | null | undefined): string { const labels: Record<string, string> = { cash: "Cash", bonus: "Bonus", unresolved: "Unresolved" }; return value ? labels[value] ?? titleCase(value) : "—"; }
function formatTransactionType(value: string | null | undefined): string { return value ? titleCase(value) : "—"; }
function formatSplitValue(dimension: SplitDimension, value: string): string { if (dimension === "market_type") return formatMarketType(value); if (dimension === "funding_type") return formatFundingType(value); return value; }
function titleCase(value: string): string { return value.split("_").map((part) => part ? `${part[0].toUpperCase()}${part.slice(1)}` : part).join(" "); }
function formatDateShort(timestamp: string | null | undefined): string { if (!timestamp) return "—"; return timestamp.split(".")[0].replace("T", " "); }
function formatSignedMoney(value: number): string { return `${value >= 0 ? "+" : "-"}$${Math.abs(value).toFixed(2)}`; }
function formatSignedPercent(value: number): string { return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`; }

const controlStyle = { background: "var(--bg-1)", border: "1px solid var(--line-soft)", borderRadius: 4, color: "var(--ink)", font: "inherit", fontSize: 11, padding: "6px 8px" } as const;
const buttonStyle = { ...controlStyle, cursor: "pointer" } as const;
const tableStyle = { width: "100%", fontSize: 12, borderCollapse: "collapse" } as const;
