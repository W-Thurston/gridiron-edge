import { usePortfolioCurve, usePortfolioSummary } from "../../api/hooks";
import { useNav } from "../../context/NavContext";
import { Spark } from "../primitives/Spark";

/**
 * Compact right-rail card showing recent model/bankroll performance.
 *
 * Data flow:
 * 1. Fetch /portfolio/summary (for all-time ROI) and /portfolio/curve
 *    (for recent trend) in parallel
 * 2. Slice curve to last 10 points for a compact sparkline
 * 3. Render sparkline + big number + "Open bankroll" CTA
 *
 * Currently shows all-time ROI (not windowed). Rolling 30d/7d windows
 * are a backend hygiene item (§9.7). Big number label is "All-Time" for
 * clarity.
 */
export function ModelPerformanceRail() {
  const summary = usePortfolioSummary();
  const curve = usePortfolioCurve();
  const { navigate } = useNav();

  const isLoading = summary.isLoading || curve.isLoading;
  const error = summary.error ?? curve.error;

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Model Performance
        </div>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Model Performance
        </div>
        <div className="dim mono" style={{ fontSize: 12 }}>
          Couldn't load performance data.
        </div>
      </div>
    );
  }

  const summaryData = summary.data;
  const curveData = curve.data;
  const points = curveData?.items ?? [];

  // Slice to last 10 points for compact display.
  const recentPoints = points.slice(-10);
  const sparkData = recentPoints.map((p) => p.bankroll ?? 0);

  const hasData = summaryData != null && (
    (summaryData.total_bets ?? 0) > 0 || sparkData.length > 0
  );

  if (!hasData) {
    return (
      <div className="hm-card" style={{ padding: 20 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
          Model Performance
        </div>
        <EmptyState />
        <button
          type="button"
          onClick={() => navigate("/mybets")}
          style={{
            marginTop: 12,
            padding: "6px 12px",
            background: "transparent",
            color: "var(--ink-3)",
            border: "1px solid var(--line-soft)",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: "var(--f-sans)",
            cursor: "pointer",
            width: "100%",
          }}
        >
          Open bankroll →
        </button>
      </div>
    );
  }

  const roiPct = summaryData?.roi_pct ?? 0;
  const isPositive = roiPct >= 0;
  const roiColor = isPositive ? "var(--pos)" : "var(--neg)";
  const roiSign = isPositive ? "+" : "";

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 12,
        }}
      >
        <div className="upper dim" style={{ fontSize: 10 }}>
          Model Performance
        </div>
        <div className="mono dim2" style={{ fontSize: 10 }}>
          Recent
        </div>
      </div>

      {/* Sparkline */}
      {sparkData.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Spark
            data={sparkData}
            width={220}
            height={40}
            color="var(--pos)"
            strokeWidth={2}
          />
        </div>
      )}

      {/* Big number */}
      <div style={{ marginBottom: 4 }}>
        <div
          className="mono tnum"
          style={{
            fontSize: 32,
            color: roiColor,
            lineHeight: 1,
          }}
        >
          {roiSign}
          {roiPct.toFixed(1)}%
        </div>
        <div className="upper dim2" style={{ fontSize: 9, marginTop: 4 }}>
          All-Time ROI
        </div>
      </div>

      {/* Secondary stats */}
      {summaryData && (
        <div
          className="mono dim"
          style={{
            fontSize: 11,
            marginTop: 12,
            display: "flex",
            gap: 12,
            paddingTop: 12,
            borderTop: "1px solid var(--line-soft)",
          }}
        >
          <span>
            {summaryData.wins ?? 0}-{summaryData.losses ?? 0}
            {(summaryData.pushes ?? 0) > 0 ? `-${summaryData.pushes}` : ""}
          </span>
          <span>·</span>
          <span>{summaryData.total_bets ?? 0} bets</span>
        </div>
      )}

      {/* CTA */}
      <button
        type="button"
        onClick={() => navigate("/mybets")}
        style={{
          marginTop: 16,
          padding: "6px 12px",
          background: "transparent",
          color: "var(--ink-2)",
          border: "1px solid var(--line-soft)",
          borderRadius: 4,
          fontSize: 11,
          fontFamily: "var(--f-sans)",
          cursor: "pointer",
          width: "100%",
        }}
      >
        Open bankroll →
      </button>
    </div>
  );
}

function EmptyState() {
  return (
    <div style={{ padding: 12, textAlign: "center" }}>
      <div className="dim mono" style={{ fontSize: 12, marginBottom: 6 }}>
        No performance data yet.
      </div>
      <div className="mono dim2" style={{ fontSize: 11 }}>
        Log bets with `gridiron bet log` to see model performance.
      </div>
    </div>
  );
}
