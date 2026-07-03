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

type SplitDimension = "market_type" | "confidence_tier" | "model_type";

export function Bankroll() {
  const summary = usePortfolioSummary();
  const bets = usePortfolioBets();
  const curve = usePortfolioCurve();
  const transactions = usePortfolioTransactions();
  const [splitDim, setSplitDim] = useState<SplitDimension>("market_type");
  const splits = usePortfolioSplits(splitDim);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Row 1: Summary + Record */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <SummaryCard result={summary} />
        <RecordCard result={summary} />
      </div>

      {/* Row 2: Balance curve — full width */}
      <CurveCard result={curve} />

      {/* Row 3: Open bets */}
      <BetsCard result={bets} />

      {/* Row 4: Splits + Transactions */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <SplitsCard
          result={splits}
          dimension={splitDim}
          onDimensionChange={setSplitDim}
        />
        <TransactionsCard result={transactions} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Summary card
// ---------------------------------------------------------------------------

function SummaryCard({
  result,
}: {
  result: ReturnType<typeof usePortfolioSummary>;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Bankroll
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (
        <>
          <div
            className="mono tnum"
            style={{ fontSize: 28, marginBottom: 16 }}
          >
            ${result.data.bankroll?.toFixed(2) ?? "—"}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
            }}
          >
            <MetricCell
              label="Total P&L"
              value={
                <span className={result.data.total_pnl != null && result.data.total_pnl >= 0 ? "pos" : "neg"}>
                  {result.data.total_pnl != null
                    ? `${result.data.total_pnl >= 0 ? "+" : ""}$${result.data.total_pnl.toFixed(2)}`
                    : "—"}
                </span>
              }
            />
            <MetricCell
              label="ROI"
              value={
                <span className={result.data.roi_pct != null && result.data.roi_pct >= 0 ? "pos" : "neg"}>
                  {result.data.roi_pct != null
                    ? `${result.data.roi_pct >= 0 ? "+" : ""}${result.data.roi_pct.toFixed(1)}%`
                    : "—"}
                </span>
              }
            />
            <MetricCell
              label="Total Staked"
              value={result.data.total_staked != null ? `$${result.data.total_staked.toFixed(2)}` : "—"}
            />
            <MetricCell
              label="Mean CLV"
              value={
                <FieldValueOrStatus
                  value={result.data.mean_clv}
                  formatter={(v) => v.toFixed(3)}
                  status={result.data._meta?.field_status?.mean_clv as FieldStatus | undefined}
                />
              }
            />
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Record card
// ---------------------------------------------------------------------------

function RecordCard({
  result,
}: {
  result: ReturnType<typeof usePortfolioSummary>;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Record
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (
        <>
          <div
            className="mono tnum"
            style={{ fontSize: 28, marginBottom: 16 }}
          >
            {result.data.wins ?? 0}-{result.data.losses ?? 0}
            {result.data.pushes && result.data.pushes > 0 ? `-${result.data.pushes}` : ""}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
            }}
          >
            <MetricCell
              label="Win %"
              value={
                result.data.win_pct != null
                  ? `${(result.data.win_pct * 100).toFixed(1)}%`
                  : "—"
              }
            />
            <MetricCell
              label="Total Bets"
              value={result.data.total_bets?.toString() ?? "—"}
            />
            <MetricCell
              label="Open Bets"
              value={result.data.open_bets?.toString() ?? "—"}
            />
            <MetricCell
              label="Current Streak"
              value={
                <FieldValueOrStatus
                  value={result.data.current_streak}
                  formatter={(v) => `${v}`}
                  status={result.data._meta?.field_status?.current_streak as FieldStatus | undefined}
                />
              }
            />
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Curve card
// ---------------------------------------------------------------------------

function CurveCard({
  result,
}: {
  result: ReturnType<typeof usePortfolioCurve>;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Balance Curve
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (
        <BalanceCurve
          points={result.data.items ?? []}
          width={1100}
          height={140}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Bets card
// ---------------------------------------------------------------------------

function BetsCard({
  result,
}: {
  result: ReturnType<typeof usePortfolioBets>;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Bets
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (result.data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No bets yet.
        </div>
      )}

      {result.data && (result.data.items ?? []).length > 0 && (
        <table
          className="mono tnum"
          style={{
            width: "100%",
            fontSize: 12,
            borderCollapse: "collapse",
          }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>Placed</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Game</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Market</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Side</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>Odds</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>Stake</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Status</th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {(result.data.items ?? []).map((bet) => (
              <tr
                key={bet.bet_id}
                style={{ borderTop: "1px solid var(--line-soft)" }}
              >
                <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}>
                  {formatDateShort(bet.placed_at)}
                </td>
                <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}>
                  {bet.game_id ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>{bet.market_type ?? "—"}</td>
                <td style={{ padding: "10px 12px 10px 0" }}>
                  {bet.side ?? "—"}
                  {bet.line != null ? ` (${bet.line})` : ""}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {bet.odds != null ? formatAmerican(bet.odds) : "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {bet.stake != null ? `$${bet.stake.toFixed(2)}` : "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0" }}>
                  <BetStatusPill status={bet.status} />
                </td>
                <td
                  style={{
                    padding: "10px 0",
                    textAlign: "right",
                    color:
                      bet.pnl == null
                        ? "var(--ink-4)"
                        : bet.pnl >= 0
                          ? "var(--pos)"
                          : "var(--neg)",
                  }}
                >
                  {bet.pnl != null
                    ? `${bet.pnl >= 0 ? "+" : ""}$${bet.pnl.toFixed(2)}`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function BetStatusPill({ status }: { status: string | null | undefined }) {
  if (!status) return <span className="dim mono">—</span>;

  const color =
    status === "won"
      ? "var(--pos)"
      : status === "lost"
        ? "var(--neg)"
        : status === "open"
          ? "var(--info)"
          : "var(--ink-3)";

  return (
    <span
      className="mono upper"
      style={{
        fontSize: 9,
        color,
        padding: "2px 6px",
        border: `1px solid ${color}`,
        borderRadius: 3,
      }}
    >
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Splits card (tabbed)
// ---------------------------------------------------------------------------

function SplitsCard({
  result,
  dimension,
  onDimensionChange,
}: {
  result: ReturnType<typeof usePortfolioSplits>;
  dimension: SplitDimension;
  onDimensionChange: (d: SplitDimension) => void;
}) {
  const tabs: { value: SplitDimension; label: string }[] = [
    { value: "market_type", label: "Market" },
    { value: "confidence_tier", label: "Confidence" },
    { value: "model_type", label: "Model" },
  ];

  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Splits
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {tabs.map((tab) => (
            <TabButton
              key={tab.value}
              label={tab.label}
              active={tab.value === dimension}
              onClick={() => onDimensionChange(tab.value)}
            />
          ))}
        </div>
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (result.data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No split data.
        </div>
      )}

      {result.data && (result.data.items ?? []).length > 0 && (
        <table
          className="mono tnum"
          style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>Segment</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>W-L-P</th>
              <th style={{ padding: "8px 12px 8px 0", textAlign: "right" }}>Win %</th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>ROI</th>
            </tr>
          </thead>
          <tbody>
            {(result.data.items ?? []).map((split, i) => (
              <tr
                key={i}
                style={{ borderTop: "1px solid var(--line-soft)" }}
              >
                <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}>
                  {split.dimension_value ?? "—"}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {split.wins ?? 0}-{split.losses ?? 0}
                  {split.pushes && split.pushes > 0 ? `-${split.pushes}` : ""}
                </td>
                <td style={{ padding: "10px 12px 10px 0", textAlign: "right" }}>
                  {split.win_pct != null
                    ? `${(split.win_pct * 100).toFixed(1)}%`
                    : "—"}
                </td>
                <td
                  style={{
                    padding: "10px 0",
                    textAlign: "right",
                    color:
                      split.roi == null
                        ? "var(--ink-4)"
                        : split.roi >= 0
                          ? "var(--pos)"
                          : "var(--neg)",
                  }}
                >
                  {split.roi != null
                    ? `${split.roi >= 0 ? "+" : ""}${(split.roi * 100).toFixed(1)}%`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function TabButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <span
      onClick={onClick}
      style={{
        fontSize: 11,
        color: active ? "var(--ink)" : "var(--ink-3)",
        borderBottom: active ? "2px solid var(--pos)" : "2px solid transparent",
        paddingBottom: 2,
        cursor: "pointer",
      }}
    >
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Transactions card
// ---------------------------------------------------------------------------

function TransactionsCard({
  result,
}: {
  result: ReturnType<typeof usePortfolioTransactions>;
}) {
  return (
    <div className="hm-card" style={{ padding: 24 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Transactions
      </div>

      {result.isLoading && <div className="dim">Loading…</div>}
      {result.error && (
        <div className="neg mono" style={{ fontSize: 12 }}>
          Error: {result.error.message}
        </div>
      )}

      {result.data && (result.data.items ?? []).length === 0 && (
        <div className="dim mono" style={{ fontSize: 12 }}>
          No transactions.
        </div>
      )}

      {result.data && (result.data.items ?? []).length > 0 && (
        <table
          className="mono tnum"
          style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}
        >
          <thead>
            <tr style={{ color: "var(--ink-3)", textAlign: "left" }}>
              <th style={{ padding: "8px 12px 8px 0" }}>When</th>
              <th style={{ padding: "8px 12px 8px 0" }}>Type</th>
              <th style={{ padding: "8px 0", textAlign: "right" }}>Amount</th>
            </tr>
          </thead>
          <tbody>
            {(result.data.items ?? []).map((txn) => {
              const sign = txn.txn_type === "deposit" || txn.txn_type === "bet_settled" ? "+" : "-";
              const color =
                txn.txn_type === "deposit" || txn.txn_type === "bet_settled"
                  ? "var(--pos)"
                  : txn.txn_type === "bet_placed"
                    ? "var(--neg)"
                    : "var(--ink-2)";
              return (
                <tr
                  key={txn.txn_id}
                  style={{ borderTop: "1px solid var(--line-soft)" }}
                >
                  <td style={{ padding: "10px 12px 10px 0", color: "var(--ink-2)" }}>
                    {formatDateShort(txn.timestamp)}
                  </td>
                  <td style={{ padding: "10px 12px 10px 0" }}>{txn.txn_type ?? "—"}</td>
                  <td
                    style={{
                      padding: "10px 0",
                      textAlign: "right",
                      color,
                    }}
                  >
                    {txn.amount != null
                      ? `${sign}$${txn.amount.toFixed(2)}`
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------

function MetricCell({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <div className="upper dim2" style={{ fontSize: 9, marginBottom: 4 }}>
        {label}
      </div>
      <div className="mono tnum" style={{ fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}

function FieldValueOrStatus<T>({
  value,
  formatter,
  status,
}: {
  value: T | null | undefined;
  formatter: (v: T) => string;
  status: FieldStatus | undefined;
}) {
  if (value != null && value !== "") return <>{formatter(value)}</>;
  if (!status) return <span className="dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

function formatDateShort(ts: string | null | undefined): string {
  if (!ts) return "—";
  // Timestamp format from API: "2026-06-21 20:10:01.160530+00:00"
  const trimmed = ts.split(".")[0].replace("T", " ");
  return trimmed;
}

function formatAmerican(odds: number): string {
  if (odds > 0) return `+${odds}`;
  return `${odds}`;
}
