import { useHistoricalModelPerformance } from "../../api/hooks";
import { useNav } from "../../context/NavContext";
import { ExplainTooltip, type ExplainTooltipSection } from "../primitives/ExplainTooltip";

/**
 * Compact historical validation snapshot for the dashboard.
 * Detailed series and methodology belong on the dedicated performance page.
 */
export function ModelPerformanceRail() {
  const summary = useHistoricalModelPerformance();
  const { navigate } = useNav();

  if (summary.isLoading) return <StateCard message="Loading…" />;
  if (summary.error) return <StateCard message="Couldn't load performance data." />;
  if (!summary.data) {
    return <StateCard message="No historical performance report selected." />;
  }

  const report = summary.data;

  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <div>
          <div className="upper dim" style={{ fontSize: 10 }}>
            Model Performance
          </div>
          <div className="mono dim2" style={{ fontSize: 9, marginTop: 3 }}>
            Walk-forward validation · current champions
          </div>
        </div>
        <div className="mono dim2" style={{ fontSize: 9, textAlign: "right" }}>
          {report.first_season}
          <br />
          through {report.last_season}
        </div>
      </div>

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        <PerformanceBlock
          label="Moneyline"
          headline={`${(report.moneyline.accuracy * 100).toFixed(1)}% accuracy`}
          record={`${report.moneyline.wins.toLocaleString()} W · ${report.moneyline.losses.toLocaleString()} L · ${report.moneyline.evaluated_count.toLocaleString()} games`}
          evidence={`${report.moneyline.net_wins.toLocaleString()} more correct than incorrect`}
          explanation={[
            {
              label: "Sample",
              text: "Moneyline accuracy uses historical games with an eligible Moneyline forecast and a completed winner. Each eligible game is graded correct or incorrect.",
            },
          ]}
        />
        <PerformanceBlock
          label="Total"
          headline={`${(report.total.hit_rate_excluding_pushes * 100).toFixed(1)}% hit rate`}
          record={`${report.total.wins.toLocaleString()} W · ${report.total.losses.toLocaleString()} L · ${report.total.pushes.toLocaleString()} P · ${report.total.decision_count.toLocaleString()} decisions`}
          evidence={`${report.total.net_wins.toLocaleString()} more wins than losses`}
          explanation={[
            {
              label: "Sample",
              text: "Total hit rate uses eligible historical Total decisions. Wins and losses determine the hit rate; pushes are recorded but excluded from the percentage.",
            },
            {
              label: "Why counts differ",
              text: "Moneyline and Total use independent eligibility rules, so their sample counts may differ.",
            },
          ]}
        />
      </div>

      <div
        style={{
          marginTop: 14,
          paddingTop: 11,
          borderTop: "1px solid var(--line-soft)",
        }}
      >
        <div className="upper dim" style={{ fontSize: 9 }}>
          Spread
        </div>
        <div className="mono dim2" style={{ fontSize: 10, marginTop: 3 }}>
          Historical validation pending
        </div>
        <button
          type="button"
          onClick={() => navigate("/performance", { scope: "backtest" })}
          style={{
            width: "100%", marginTop: 12, padding: "7px 10px",
            background: "transparent", color: "var(--ink-2)",
            border: "1px solid var(--line-soft)", borderRadius: 4,
            fontFamily: "var(--f-sans)", fontSize: 10, cursor: "pointer",
          }}
        >
          View full performance →
        </button>
      </div>
    </div>
  );
}

function PerformanceBlock({
  label,
  headline,
  record,
  evidence,
  explanation,
}: {
  label: string;
  headline: string;
  record: string;
  evidence: string;
  explanation: ExplainTooltipSection[];
}) {
  return (
    <div
      style={{
        padding: 12,
        border: "1px solid var(--line-soft)",
        borderRadius: 5,
      }}
    >
      <ExplainTooltip
        accessibleLabel={`Explain ${label} performance sample`}
        title={`${label} performance`}
        sections={explanation}
        className="model-edge-header-explanation"
      >
        <span className="upper dim" style={{ fontSize: 9 }}>
          {label} <span aria-hidden="true">ⓘ</span>
        </span>
      </ExplainTooltip>
      <div className="mono tnum" style={{ fontSize: 22, marginTop: 5 }}>
        {headline}
      </div>
      <div className="mono dim" style={{ fontSize: 10, marginTop: 6 }}>
        {record}
      </div>
      <div className="mono dim2" style={{ fontSize: 9, marginTop: 4 }}>
        {evidence}
      </div>
    </div>
  );
}

function StateCard({ message }: { message: string }) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div className="upper dim" style={{ fontSize: 10, marginBottom: 16 }}>
        Model Performance
      </div>
      <div className="dim mono" style={{ fontSize: 12 }}>
        {message}
      </div>
    </div>
  );
}
