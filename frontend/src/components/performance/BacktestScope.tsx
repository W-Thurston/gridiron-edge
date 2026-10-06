import { useHistoricalModelPerformance, useHistoricalModelPerformanceSeries } from "../../api/hooks";
import { ErrorCard } from "../error/ErrorCard";
import { PerformanceChart } from "../model-performance/PerformanceChart";
import { CompareChart } from "./CompareChart";
import { MethodDetails } from "./MethodDetails";
import { SectionHead } from "./SectionHead";
import { SegmentedControl } from "./SegmentedControl";
import { RecordBar, StatCard } from "./StatCard";
import { BACKTEST_COMPARISON } from "./compareData";
import type { CompareModelId } from "./compareData";
import { fixed, formatTimestamp, percent, signedCount, signedFixed } from "./format";

export type BacktestMarket = "moneyline" | "total" | "spread";
export type BacktestTrend = "net-wins" | "accuracy" | "units";

export type BacktestState = {
  market: BacktestMarket;
  trend: BacktestTrend;
  models: CompareModelId[];
  metric: string | undefined;
};

type Report = NonNullable<ReturnType<typeof useHistoricalModelPerformance>["data"]>;
type Point = NonNullable<ReturnType<typeof useHistoricalModelPerformanceSeries>["data"]>["items"][number];

const MARKETS = [
  { value: "moneyline", label: "Moneyline" },
  { value: "total", label: "Total" },
  { value: "spread", label: "Spread" },
] as const;

/** Walk-forward replay of the current champion models over past seasons. */
export function BacktestScope({
  state, onChange, onOpenScope,
}: {
  state: BacktestState;
  onChange: (next: Partial<BacktestState>) => void;
  onOpenScope: (scope: "week" | "season") => void;
}) {
  const summary = useHistoricalModelPerformance();
  const series = useHistoricalModelPerformanceSeries();

  if (summary.isLoading || series.isLoading) {
    return <div className="hm-card" style={{ padding: 24 }}>Loading model performance…</div>;
  }
  if (summary.error) return <ErrorCard error={summary.error} onRetry={() => summary.refetch()} />;
  if (series.error) return <ErrorCard error={series.error} onRetry={() => series.refetch()} />;
  if (!summary.data || !series.data) {
    return (
      <section className="hm-card pf-empty">
        <div className="pf-eyebrow">Backtest</div>
        <h2>No historical report selected</h2>
      </section>
    );
  }

  const report = summary.data;
  const { market } = state;
  const trend: BacktestTrend = market === "moneyline" && state.trend === "units" ? "net-wins" : state.trend;
  const chooseMarket = (value: BacktestMarket) =>
    onChange({ market: value, ...(value !== "total" && state.trend === "units" ? { trend: "net-wins" } : {}) });

  return (
    <>
      <header className="pf-hero">
        <div>
          <div className="pf-badge-row">
            <span className="pf-badge pf-badge-shadow">Backtest · walk-forward</span>
          </div>
          <h1>{report.first_season}–{report.last_season} <span>· current champion models</span></h1>
        </div>
        <SegmentedControl label="Market" options={MARKETS} value={market} onChange={chooseMarket} />
      </header>
      <p className="pf-caveat">
        These results replay the current models over past seasons, training only on data available before each
        season. They are not a record of published picks.
      </p>

      {market === "spread" ? (
        <section className="hm-card pf-empty">
          <div className="pf-eyebrow">Spread</div>
          <h2>Historical validation is not available</h2>
          <p>Leakage-safe historical Spread calibration is pending. No synthetic results are shown.</p>
          <p>
            Live Spread results against the closing line are in the{" "}
            <button type="button" className="pf-link" style={{ color: "var(--info)" }} onClick={() => onOpenScope("week")}>Week</button>
            {" "}and{" "}
            <button type="button" className="pf-link" style={{ color: "var(--info)" }} onClick={() => onOpenScope("season")}>Season</button>
            {" "}views.
          </p>
        </section>
      ) : (
        <>
          <StatGrid market={market} report={report} />
          <section className="hm-card pf-card" aria-labelledby="backtest-trend">
            <SectionHead
              eyebrow="Historical trend"
              title={trend === "net-wins" ? "Cumulative net wins" : trend === "accuracy" ? "Rolling accuracy" : "Cumulative hypothetical units"}
              description="Persisted series · no browser-side metric reconstruction"
              id="backtest-trend"
            >
              <SegmentedControl
                label="Trend metric"
                options={[
                  { value: "net-wins", label: "Net wins" },
                  { value: "accuracy", label: "Accuracy" },
                  ...(market === "total" ? [{ value: "units" as const, label: "Hypothetical units" }] : []),
                ]}
                value={trend}
                onChange={(value) => onChange({ trend: value })}
              />
            </SectionHead>
            <PerformanceChart
              points={chartPoints(market, trend, series.data.items)}
              valueKind={trend === "accuracy" ? "percentage" : trend === "units" ? "units" : "number"}
              color={trend === "units" ? "var(--neg)" : "var(--pos)"}
            />
            <ChartNote market={market} trend={trend} report={report} />
          </section>
          <CompareChart
            eyebrow="Compare · season by season"
            title={`${market === "moneyline" ? "Moneyline" : "Total"} · champion against other models`}
            source={BACKTEST_COMPARISON}
            selected={state.models}
            onSelectedChange={(models) => onChange({ models })}
            metricKey={state.metric}
            onMetricChange={(metric) => onChange({ metric })}
          />
        </>
      )}

      <MethodDetails>
        <div>
          <span>Season-by-season walk-forward evaluation of the explicitly selected current champion runs.</span>
          <span>Moneyline units are unavailable because historical Moneyline prices were not retained.</span>
          <span>Total units use historical consensus lines, one unit per decision, and assumed -110 pricing.</span>
          <span>Observed historical odds are a future replacement for the assumed-price return layer.</span>
        </div>
        <div className="pf-method-mono">
          <span>Moneyline · {report.moneyline.model_type} · {report.moneyline.run_id}</span>
          <span>Total · {report.total.model_type} · {report.total.run_id}</span>
          <span>Report {report.report_id} · selected {formatTimestamp(report.selected_at)}</span>
        </div>
      </MethodDetails>
    </>
  );
}

function StatGrid({ market, report }: { market: "moneyline" | "total"; report: Report }) {
  if (market === "moneyline") {
    const ml = report.moneyline;
    return (
      <div className="pf-stats">
        <StatCard label="Accuracy" value={percent(ml.accuracy)} tone="pos" sub={`${ml.wins.toLocaleString()} W · ${ml.losses.toLocaleString()} L`}>
          <RecordBar wins={ml.wins} losses={ml.losses} />
        </StatCard>
        <StatCard label="Net correct" value={signedCount(ml.net_wins)} sub={`${ml.evaluated_count.toLocaleString()} games evaluated`} />
        <StatCard label="Brier" value={fixed(ml.brier, 4)} sub={`Log loss ${fixed(ml.log_loss, 4)}`} />
        <StatCard label="Hypothetical units" value="—" tone="dim" sub="Historical Moneyline prices were not retained" />
      </div>
    );
  }
  const total = report.total;
  return (
    <div className="pf-stats">
      <StatCard
        label="Hit rate"
        value={percent(total.hit_rate_excluding_pushes)}
        tone="pos"
        sub={`${total.wins.toLocaleString()} W · ${total.losses.toLocaleString()} L · ${total.pushes.toLocaleString()} P`}
      >
        <RecordBar wins={total.wins} losses={total.losses} pushes={total.pushes} />
      </StatCard>
      <StatCard label="Net wins" value={signedCount(total.net_wins)} sub={`${total.decision_count.toLocaleString()} decisions · pushes count zero`} />
      <StatCard label="MAE" value={fixed(total.mae, 2)} sub={`RMSE ${fixed(total.rmse, 2)} · bias ${signedFixed(total.bias, 2)}`} />
      <StatCard
        label="Hypothetical units"
        value={`${signedFixed(total.net_units, 2)}u`}
        sub={`ROI ${percent(total.roi_per_unit_staked)} per unit · assumed ${total.assumed_american_price ?? -110}`}
      />
    </div>
  );
}

function ChartNote({ market, trend, report }: { market: "moneyline" | "total"; trend: BacktestTrend; report: Report }) {
  let text = "Each correct prediction adds one and each incorrect prediction subtracts one. Pushes contribute zero.";
  if (trend === "accuracy") text = `Trailing ${report.rolling_decision_window} graded decisions.`;
  if (trend === "units") text = report.total.methodology;
  if (market === "moneyline" && trend === "net-wins") text += " Sportsbook pricing is ignored.";
  return <div className="pf-note">{text}</div>;
}

function chartPoints(market: "moneyline" | "total", trend: BacktestTrend, points: Point[]) {
  return points.map((point) => ({
    game_date: point.game_date,
    season: point.season,
    week: point.week,
    value: market === "moneyline"
      ? trend === "accuracy" ? point.moneyline_rolling_accuracy_100 : point.moneyline_cumulative_net_wins
      : trend === "accuracy" ? point.total_rolling_accuracy_100 : trend === "units" ? point.total_cumulative_units : point.total_cumulative_net_wins,
  }));
}
